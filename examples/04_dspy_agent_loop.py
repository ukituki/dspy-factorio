#!/usr/bin/env python3
"""Intro: DSPy agent loop (baseline Predict → Factorio step).

First contact with the online agent — no optimization, no compiled modules.
Requires OPENAI_API_KEY and a running Factorio cluster.

Usage:
    uv run python examples/04_dspy_agent_loop.py --steps 5
    uv run python examples/04_dspy_agent_loop.py --steps 8 --no-renders
    open .fle/renders/dspy_agent/<session>/step_02.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_factorio.agent import (
    API_HINT,
    AgentConfig,
    build_agent,
    propose_program,
)
from dspy_factorio.env import (
    describe_env,
    get_environment_info,
    load_project_env,
    make_env,
    new_render_session_dir,
    obs_text,
    reset_env,
    save_render,
    step_code,
)

BOOTSTRAP = """
print(inspect_inventory())
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
""".strip()


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.rsplit("/", 1)[-1])[:32]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--renders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save schematic map PNGs after bootstrap and each agent step (default: on)",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path(".fle/renders/dspy_agent"),
        help="Parent folder for session PNGs (default: .fle/renders/dspy_agent)",
    )
    parser.add_argument(
        "--render-mode",
        choices=("simple", "sprites"),
        default="simple",
        help="simple=schematic (default); sprites needs `fle sprites`",
    )
    parser.add_argument(
        "--render-zoom",
        type=float,
        default=0.25,
        help=(
            "Zoom for renders (simple mode). Values < 1 zoom out. "
            "Default 0.25 ≈ 80-tile radius so spawn and iron (~y=70) share a frame"
        ),
    )
    args = parser.parse_args()

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = info.get("description") or args.env_id
    print(describe_env(args.env_id))
    print(f"Intro agent (baseline Predict) | model={args.model} | steps={args.steps}")

    render_dir = None
    if args.renders:
        tag = f"{args.env_id}_{_model_slug(args.model)}"
        render_dir = new_render_session_dir(args.render_dir, tag=tag)
        print(
            f"Renders → {render_dir.resolve()} "
            f"(mode={args.render_mode}, zoom={args.render_zoom})"
        )

    def dump(name: str) -> None:
        if render_dir is None:
            return
        path = save_render(
            env,
            render_dir / name,
            mode=args.render_mode,
            zoom=args.render_zoom if args.render_mode == "simple" else None,
            overview=args.render_mode == "simple",
        )
        print(f"saved {path}")

    agent = build_agent(AgentConfig(model=args.model))
    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        dump("step_00_reset.png")

        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("\n=== bootstrap ===")
        print(observation[:1200])
        dump("step_01_bootstrap.png")

        for step in range(1, args.steps + 1):
            print(f"\n=== agent step {step}/{args.steps} ===")
            program = propose_program(
                agent,
                goal=goal,
                observation=observation,
                inventory_hint=API_HINT,
            )
            print("--- proposed program ---")
            print(program)
            obs, reward, terminated, truncated, _info = step_code(env, program)
            observation = obs_text(obs) or "(empty)"
            print("--- factorio output ---")
            print(observation[:1200])
            print(f"reward={reward} done={terminated or truncated}")
            dump(f"step_{step + 1:02d}.png")
            if terminated or truncated:
                break
    finally:
        env.close()

    if render_dir is not None:
        print(f"\nDone. Open PNGs under {render_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
