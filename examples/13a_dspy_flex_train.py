#!/usr/bin/env python3
"""Flex train (13a): learn a Flex program from online Factorio play.

Flow:
  1. Play the drill milestone with baseline ``dspy.Flex`` (Factorio cluster).
  2. Keep healthy (obs → program) steps as a trainset learned from play.
  3. Compile with ``dspy.GEPA`` under a *tiny* metric budget (default 24 calls —
     not ``auto=light``, which can be 300+ for Flex).
  4. Save artifact for ``examples/13b_dspy_flex_run.py``.

Pair with intro ``12_dspy_flex_miner.py`` and run ``13b_dspy_flex_run.py``.
See ``docs/FLEX_STARTER.md``.

Requires: OPENAI_API_KEY, Deno, Factorio cluster (unless ``--dry-run``).

Usage:
    uv run python examples/13a_dspy_flex_train.py --dry-run
    uv run python examples/13a_dspy_flex_train.py
    uv run python examples/13a_dspy_flex_train.py --max-metric-calls 24
    # larger budget only when you mean it:
    uv run python examples/13a_dspy_flex_train.py --auto light
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from dspy_factorio.agent import (
    API_HINT,
    AgentConfig,
    FactorioProgrammer,
    build_lm,
    propose_program,
    strip_code_fences,
)
from dspy_factorio.env import (
    describe_env,
    get_environment_info,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    step_code,
)
from dspy_factorio.flex_drill import (
    BOOTSTRAP,
    DEFAULT_FLEX_PROGRAM,
    DEFAULT_PLAY_DEMOS,
    DRILL_GOAL,
    FLEX_VAL_DEMOS,
    build_flex_agent,
    flex_gepa_metric,
    looks_like_fueled_drill,
    make_play_example,
    step_looks_healthy,
)

def collect_play_demos(
    *,
    env,
    agent: dspy.Module,
    goal: str,
    episodes: int,
    steps: int,
) -> list[dspy.Example]:
    """Roll out baseline Flex in Factorio; keep healthy steps as gold programs."""
    demos: list[dspy.Example] = []

    for ep in range(1, episodes + 1):
        print(f"\n######## play episode {ep}/{episodes} ########")
        reset_env(env)
        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("--- bootstrap ---")
        print(observation[:800])

        for step in range(1, steps + 1):
            print(f"\n=== play step {ep}.{step} ===")
            obs_before = observation
            program = propose_program(
                agent,
                goal=goal,
                observation=obs_before,
                inventory_hint=API_HINT,
            )
            print("--- program ---")
            print(program)
            obs, reward, terminated, truncated, _info = step_code(env, program)
            observation = obs_text(obs) or "(empty)"
            print("--- output ---")
            print(observation[:800])
            print(f"reward={reward} done={terminated or truncated}")

            if step_looks_healthy(observation):
                demos.append(
                    make_play_example(
                        goal=goal,
                        observation=obs_before,
                        inventory_hint=API_HINT,
                        program=program,
                    )
                )
                print(f"kept demo #{len(demos)}")
            else:
                print("dropped step (unhealthy Factorio output)")

            if looks_like_fueled_drill(observation, program):
                print("episode success: fueled drill heuristic")
                break
            if terminated or truncated:
                break

    return demos


def demos_to_jsonable(demos: list[dspy.Example]) -> list[dict]:
    rows = []
    for ex in demos:
        rows.append(
            {
                "goal": ex.goal,
                "observation": ex.observation,
                "inventory_hint": ex.inventory_hint,
                "program": ex.program,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Factorio play + GEPA LM calls; print the planned flow",
    )
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Task LM (play + compile)")
    parser.add_argument(
        "--reflection-model",
        default="openai/gpt-4o",
        help="Stronger LM for GEPA code proposals (temperature=1.0)",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Online play episodes")
    parser.add_argument("--steps", type=int, default=6, help="Max steps per play episode")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="GEPA metric-call cap (default 24 if no --auto / --max-full-evals)",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="GEPA auto budget (expensive for Flex — prefer the default tiny cap)",
    )
    parser.add_argument(
        "--max-full-evals",
        type=int,
        default=None,
        help="Alternative GEPA budget: full train+val evals",
    )
    parser.add_argument("--save", default=DEFAULT_FLEX_PROGRAM, help="Compiled Flex JSON")
    parser.add_argument(
        "--demos-out",
        default=DEFAULT_PLAY_DEMOS,
        help="Where to write demos collected from play",
    )
    parser.add_argument("--log-dir", default=".fle/flex_play_gepa_logs")
    parser.add_argument(
        "--min-demos",
        type=int,
        default=3,
        help="Abort compile if fewer healthy play demos than this",
    )
    args = parser.parse_args()

    n_budget = sum(
        x is not None for x in (args.auto, args.max_full_evals, args.max_metric_calls)
    )
    if n_budget == 0:
        args.max_metric_calls = 24
    elif n_budget > 1:
        parser.error("Use exactly one of --max-metric-calls, --auto, --max-full-evals")

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = (
        f"{DRILL_GOAL} "
        f"(Full env description: {info.get('description') or args.env_id})"
    )

    budget_desc = (
        f"auto={args.auto}"
        if args.auto
        else (
            f"max_full_evals={args.max_full_evals}"
            if args.max_full_evals
            else f"max_metric_calls={args.max_metric_calls}"
        )
    )
    print(describe_env(args.env_id))
    print(
        f"Flex train 13a | play episodes={args.episodes} steps={args.steps} "
        f"| {budget_desc} | model={args.model}"
    )

    if args.dry_run:
        print("Dry-run OK.")
        print("Would: play Factorio with baseline Flex → keep healthy steps → GEPA on Flex.")
        print(f"Would save demos → {args.demos_out}")
        print(f"Would save Flex  → {args.save}")
        print(f"valset=FLEX_VAL_DEMOS ({len(FLEX_VAL_DEMOS)}) | budget={budget_desc}")
        print("Tip: default --max-metric-calls 24 shows learning without 300+ rollouts.")
        return 0

    # --- Phase 1: online play -------------------------------------------------
    play_agent = build_flex_agent(AgentConfig(model=args.model))
    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        play_demos = collect_play_demos(
            env=env,
            agent=play_agent,
            goal=goal,
            episodes=args.episodes,
            steps=args.steps,
        )
    finally:
        env.close()

    demos_path = Path(args.demos_out)
    demos_path.parent.mkdir(parents=True, exist_ok=True)
    demos_path.write_text(json.dumps(demos_to_jsonable(play_demos), indent=2))
    print(f"\nCollected {len(play_demos)} play demos → {demos_path}")

    if len(play_demos) < args.min_demos:
        print(
            f"Need at least --min-demos {args.min_demos} healthy steps "
            f"(got {len(play_demos)}). Re-run with more --episodes/--steps "
            "or a stronger --model."
        )
        return 1

    # Fresh Flex student (play agent is only for demo collection).
    student = dspy.Flex(FactorioProgrammer)
    baseline_src = student.module_src
    print("\n=== baseline module_src (head) ===")
    print("\n".join(baseline_src.splitlines()[:12]))

    reflection_lm = dspy.LM(args.reflection_model, temperature=1.0, max_tokens=8000)
    gepa_kwargs: dict = {
        "metric": flex_gepa_metric,
        "reflection_lm": reflection_lm,
        "reflection_minibatch_size": min(2, len(play_demos)),
        "candidate_selection_strategy": "pareto",
        "skip_perfect_score": True,
        "track_stats": True,
        "log_dir": args.log_dir,
        "seed": 0,
    }
    if args.auto is not None:
        gepa_kwargs["auto"] = args.auto
    elif args.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = args.max_full_evals
    else:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls

    optimizer = dspy.GEPA(**gepa_kwargs)

    print(
        f"\nGEPA compile on play trainset={len(play_demos)} "
        f"valset={len(FLEX_VAL_DEMOS)} | {budget_desc}"
    )
    dspy.configure(lm=build_lm(AgentConfig(model=args.model, max_tokens=1024)))
    compiled = optimizer.compile(student, trainset=play_demos, valset=FLEX_VAL_DEMOS)

    sample = FLEX_VAL_DEMOS[0]
    pred = compiled(
        goal=sample.goal,
        observation=sample.observation,
        inventory_hint=sample.inventory_hint,
    )
    scored = flex_gepa_metric(sample, pred)
    print("\n=== sample val (fuel-after-place) ===")
    print(strip_code_fences(pred.program))
    print(f"score={scored.score:.2f}")
    print(f"feedback={scored.feedback}")
    print("\n=== module_src after GEPA (head) ===")
    print("\n".join(compiled.module_src.splitlines()[:40]))
    if compiled.module_src == baseline_src:
        print(
            "(module_src unchanged — budget exhausted or baseline already strong; "
            "demos from play are still saved for inspection.)"
        )
    else:
        print("(module_src changed — Flex structure/instructions were rewritten.)")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(save_path))
    meta = {
        "optimizer": "gepa+flex",
        "source": "online_play",
        "budget": budget_desc,
        "max_metric_calls": args.max_metric_calls,
        "auto": args.auto,
        "max_full_evals": args.max_full_evals,
        "model": args.model,
        "reflection_model": args.reflection_model,
        "play_episodes": args.episodes,
        "play_steps": args.steps,
        "play_demos": len(play_demos),
        "val_size": len(FLEX_VAL_DEMOS),
        "demos_path": str(demos_path),
        "path": str(save_path),
        "log_dir": args.log_dir,
        "module_src_changed": compiled.module_src != baseline_src,
    }
    save_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved Flex → {save_path}")
    print("Next: uv run python examples/13b_dspy_flex_run.py --program", save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
