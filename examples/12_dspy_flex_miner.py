#!/usr/bin/env python3
"""Intro: dspy.Flex places a fueled burner mining drill on iron ore.

Baseline Flex only — no GEPA. Same milestone as ``11_dspy_rlm_miner.py``.
Outer Python loop owns steps; Flex proposes one FLE program per step.

Next: learn from online play with ``13a_dspy_flex_train.py``, then roll out
with ``13b_dspy_flex_run.py``. See ``docs/FLEX_STARTER.md``.

Requires: OPENAI_API_KEY, Deno (`brew install deno`), Factorio cluster.

Usage:
    uv run python examples/12_dspy_flex_miner.py
    uv run python examples/12_dspy_flex_miner.py --verbose --steps 6
    uv run python examples/12_dspy_flex_miner.py --renders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorio_gym.agent import API_HINT, AgentConfig, propose_program
from factorio_gym.env import (
    describe_env,
    get_environment_info,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    save_render,
    step_code,
)
from factorio_gym.flex_drill import (
    BOOTSTRAP,
    DRILL_GOAL,
    build_flex_agent,
    looks_like_fueled_drill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Max outer-loop Factorio steps after bootstrap (default 6)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print Flex module_src head at start",
    )
    parser.add_argument(
        "--renders",
        action="store_true",
        help="Save a schematic map PNG after bootstrap and each step",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path(".fle/renders/flex_miner"),
        help="Directory for --renders PNGs",
    )
    parser.add_argument(
        "--render-zoom",
        type=float,
        default=0.25,
        help="Zoom for --renders (simple mode). Default 0.25 ≈ spawn+iron in frame",
    )
    args = parser.parse_args()

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = (
        f"{DRILL_GOAL} "
        f"(Full env description: {info.get('description') or args.env_id})"
    )
    print(describe_env(args.env_id))
    print(f"Flex intro | model={args.model} | steps={args.steps} | baseline (no train)")

    agent = build_flex_agent(AgentConfig(model=args.model))
    if args.verbose:
        print("\n=== module_src (head) ===")
        print("\n".join(agent.module_src.splitlines()[:16]))

    env = make_env(args.env_id, run_idx=args.run_idx)
    success = False
    try:
        reset_env(env)
        if args.renders:
            path = save_render(
                env,
                args.render_dir / "step_00_reset.png",
                mode="simple",
                zoom=args.render_zoom,
                overview=True,
            )
            print(f"saved {path}")

        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("\n=== bootstrap ===")
        print(observation[:1200])
        if args.renders:
            path = save_render(
                env,
                args.render_dir / "step_01_bootstrap.png",
                mode="simple",
                zoom=args.render_zoom,
                overview=True,
            )
            print(f"saved {path}")

        for step in range(1, args.steps + 1):
            print(f"\n=== flex step {step}/{args.steps} ===")
            program = propose_program(
                agent,
                goal=goal,
                observation=observation,
                inventory_hint=API_HINT,
            )
            print("--- program ---")
            print(program)
            obs, reward, terminated, truncated, _info = step_code(env, program)
            observation = obs_text(obs) or "(empty)"
            print("--- output ---")
            print(observation[:1200])
            print(f"reward={reward} done={terminated or truncated}")
            if args.renders:
                path = save_render(
                    env,
                    args.render_dir / f"step_{step + 1:02d}.png",
                    mode="simple",
                    zoom=args.render_zoom,
                    overview=True,
                )
                print(f"saved {path}")

            if looks_like_fueled_drill(observation, program):
                success = True
                print("\n=== early stop: observation looks like a fueled drill ===")
                break
            if terminated or truncated:
                break
    finally:
        env.close()

    print("\n=== Flex intro result ===")
    print(f"success={success}")
    print("Next: uv run python examples/13a_dspy_flex_train.py")
    if args.renders:
        print(f"Done. Open PNGs under {args.render_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
