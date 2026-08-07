#!/usr/bin/env python3
"""GEPA train: compile a FactorioProgrammer offline (no Factorio).

Pair with ``examples/08_gepa_run.py`` to roll out the saved module.
See ``docs/GEPA_STARTER.md``.

Usage:
    uv run python examples/07_gepa_train.py --dry-run
    uv run python examples/07_gepa_train.py --auto light
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from factorio_gym.agent import AgentConfig, FactorioProgrammer, build_lm, strip_code_fences
from factorio_gym.env import load_project_env
from factorio_gym.trainset import TRAIN_DEMOS, VAL_DEMOS

DEFAULT_SAVE = ".fle/gepa_factorio_agent.json"


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Rich-feedback metric — GEPA needs textual critique, not score alone."""
    program = strip_code_fences(getattr(pred, "program", "") or "")
    notes: list[str] = []
    score = 0.0

    if not program.strip():
        return dspy.Prediction(
            score=0.0,
            feedback="Empty program. Emit executable FLE Python with print().",
        )

    if "print(" in program:
        score += 0.25
    else:
        notes.append("Add print(...) so the REPL returns an observation.")

    if any(t in program for t in ("nearest(", "place_entity(", "insert_item(", "move_to(", "get_entities(")):
        score += 0.25
    else:
        notes.append("Call an FLE tool (nearest / move_to / place_entity / insert_item / get_entities).")

    if "Resource." in program or "Prototype." in program:
        score += 0.25
    else:
        notes.append("Use Resource.* / Prototype.* enums — never string names like nearest(\"iron-ore\").")

    bad_string_api = 'nearest("' in program or "nearest('" in program
    if "```" not in program and not bad_string_api:
        score += 0.25
    else:
        notes.append("No markdown fences; never nearest(\"iron-ore\"). Prefer nearest(Resource.IronOre).")

    gold_prog = getattr(gold, "program", "") or ""
    if "move_to(" in gold_prog and "move_to(" not in program:
        notes.append("Gold uses move_to before placement — include move_to when the target is far.")
        score = min(score, 0.75)

    if score >= 0.99:
        feedback = "Good FLE program: enums, tools, and print() present."
    else:
        feedback = " ".join(notes) or "Improve FLE API usage."
        feedback += f" Expected style resembles: {gold_prog[:180]!r}"

    return dspy.Prediction(score=score, feedback=feedback)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Construct GEPA only (no LM calls)")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Task LM")
    parser.add_argument(
        "--reflection-model",
        default="openai/gpt-4o",
        help="Stronger LM for GEPA reflection (temperature=1.0)",
    )
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--log-dir", default=".fle/gepa_logs")
    args = parser.parse_args()

    load_project_env()
    student = dspy.Predict(FactorioProgrammer)

    reflection_lm = dspy.LM(args.reflection_model, temperature=1.0, max_tokens=8000)
    optimizer = dspy.GEPA(
        metric=gepa_metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=2,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        track_stats=True,
        log_dir=args.log_dir,
        seed=0,
    )

    if args.dry_run:
        print("GEPA constructed OK (dry-run).")
        print(f"train={len(TRAIN_DEMOS)} val={len(VAL_DEMOS)} auto={args.auto}")
        print(f"task_lm={args.model} reflection_lm={args.reflection_model}")
        return 0

    dspy.configure(lm=build_lm(AgentConfig(model=args.model, max_tokens=1024)))
    print(f"GEPA train: train={len(TRAIN_DEMOS)} val={len(VAL_DEMOS)} auto={args.auto}")
    compiled = optimizer.compile(student, trainset=TRAIN_DEMOS, valset=VAL_DEMOS)

    sample = VAL_DEMOS[0]
    pred = compiled(
        goal=sample.goal,
        observation=sample.observation,
        inventory_hint=sample.inventory_hint,
    )
    scored = gepa_metric(sample, pred)
    print("\n--- sample val output ---")
    print(strip_code_fences(pred.program))
    print(f"score={scored.score:.2f}")
    print(f"feedback={scored.feedback}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(save_path))
    meta = {
        "optimizer": "gepa",
        "auto": args.auto,
        "model": args.model,
        "reflection_model": args.reflection_model,
        "train_size": len(TRAIN_DEMOS),
        "val_size": len(VAL_DEMOS),
        "path": str(save_path),
        "log_dir": args.log_dir,
    }
    save_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved → {save_path}")
    print("Next: uv run python examples/08_gepa_run.py --program", save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
