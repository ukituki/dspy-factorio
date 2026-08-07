#!/usr/bin/env python3
"""BootstrapFewShot train: compile a FactorioProgrammer offline (no Factorio).

Train-only. For a reflective optimizer with a matching run script, see
``examples/07_gepa_train.py`` + ``examples/08_gepa_run.py``.

Usage:
    uv run python examples/05_optimize_agent.py
    uv run python examples/05_optimize_agent.py --save .fle/optimized_factorio_agent.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy
from dspy.teleprompt import BootstrapFewShot

from factorio_gym.agent import (
    AgentConfig,
    FactorioProgrammer,
    build_lm,
    strip_code_fences,
)
from factorio_gym.env import load_project_env
from factorio_gym.trainset import TRAIN_DEMOS as SEED_DEMOS


def program_metric(example, pred, trace=None) -> float:
    """Prefer real FLE enums; penalize stringly-typed tool calls."""
    program = strip_code_fences(getattr(pred, "program", "") or "")
    if not program.strip():
        return 0.0
    score = 0.0
    if "print(" in program:
        score += 0.25
    if any(
        tok in program
        for tok in ("nearest(", "place_entity(", "insert_item(", "get_entities(", "move_to(")
    ):
        score += 0.25
    if "Resource." in program or "Prototype." in program:
        score += 0.25
    if "```" not in program and 'nearest("' not in program and "nearest('" not in program:
        score += 0.25
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--save", default=".fle/optimized_factorio_agent.json")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=2)
    parser.add_argument("--max-labeled-demos", type=int, default=2)
    args = parser.parse_args()

    load_project_env()
    lm = build_lm(AgentConfig(model=args.model))
    dspy.configure(lm=lm)

    student = dspy.Predict(FactorioProgrammer)
    optimizer = BootstrapFewShot(
        metric=program_metric,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )
    print(f"Optimizing with {len(SEED_DEMOS)} examples on {args.model}...")
    compiled = optimizer.compile(student, trainset=SEED_DEMOS)

    sample = SEED_DEMOS[0]
    pred = compiled(
        goal=sample.goal,
        observation=sample.observation,
        inventory_hint=sample.inventory_hint,
    )
    print("\n--- sample optimized output ---")
    print(strip_code_fences(pred.program))
    print(f"metric={program_metric(sample, pred):.2f}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(save_path))
    meta = {
        "model": args.model,
        "train_size": len(SEED_DEMOS),
        "metric": "program_metric",
        "path": str(save_path),
    }
    save_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved compiled module → {save_path}")
    print("To roll out, mirror examples/08_gepa_run.py with --program", save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
