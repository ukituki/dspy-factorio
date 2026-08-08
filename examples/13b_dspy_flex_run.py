#!/usr/bin/env python3
"""Flex run (13b): roll out a Flex program learned from online play.

Pair with ``examples/13a_dspy_flex_train.py`` (play → GEPA → save).
Does not optimize. Same drill milestone as intro ``12`` / RLM ``11``.

See ``docs/FLEX_STARTER.md``.

Usage:
    uv run python examples/13b_dspy_flex_run.py --steps 6
    uv run python examples/13b_dspy_flex_run.py --program .fle/flex_from_play.json --verbose
    uv run python examples/13b_dspy_flex_run.py --renders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_factorio.agent import API_HINT, AgentConfig, propose_program
from dspy_factorio.env import (
    describe_env,
    get_environment_info,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    save_render,
    step_code,
)
from dspy_factorio.flex_drill import (
    BOOTSTRAP,
    DEFAULT_FLEX_PROGRAM,
    DRILL_GOAL,
    build_flex_agent,
    looks_like_fueled_drill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program",
        default=DEFAULT_FLEX_PROGRAM,
        help="Compiled Flex module from 13a (default: .fle/flex_from_play.json)",
    )
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print loaded module_src head before the rollout",
    )
    parser.add_argument(
        "--renders",
        action="store_true",
        help="Save a schematic map PNG after bootstrap and each step",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path(".fle/renders/flex_from_play"),
    )
    parser.add_argument("--render-zoom", type=float, default=0.25)
    args = parser.parse_args()

    path = Path(args.program)
    if not path.is_file():
        print(f"Missing compiled Flex program: {path}")
        print("Train first: uv run python examples/13a_dspy_flex_train.py")
        return 1

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = (
        f"{DRILL_GOAL} "
        f"(Full env description: {info.get('description') or args.env_id})"
    )
    print(describe_env(args.env_id))
    print(f"Flex run 13b | program={path} | model={args.model} | steps={args.steps}")

    agent = build_flex_agent(AgentConfig(model=args.model), path)
    if args.verbose:
        print("\n=== module_src (head) ===")
        print("\n".join(agent.module_src.splitlines()[:40]))

    env = make_env(args.env_id, run_idx=args.run_idx)
    success = False
    try:
        reset_env(env)
        if args.renders:
            path_png = save_render(
                env,
                args.render_dir / "step_00_reset.png",
                mode="simple",
                zoom=args.render_zoom,
                overview=True,
            )
            print(f"saved {path_png}")

        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("\n=== bootstrap ===")
        print(observation[:1200])
        if args.renders:
            path_png = save_render(
                env,
                args.render_dir / "step_01_bootstrap.png",
                mode="simple",
                zoom=args.render_zoom,
                overview=True,
            )
            print(f"saved {path_png}")

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
                path_png = save_render(
                    env,
                    args.render_dir / f"step_{step + 1:02d}.png",
                    mode="simple",
                    zoom=args.render_zoom,
                    overview=True,
                )
                print(f"saved {path_png}")

            if looks_like_fueled_drill(observation, program):
                success = True
                print("\n=== early stop: observation looks like a fueled drill ===")
                break
            if terminated or truncated:
                break
    finally:
        env.close()

    print("\n=== Flex run result ===")
    print(f"success={success}")
    print(f"program={path}")
    if args.renders:
        print(f"Done. Open PNGs under {args.render_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
