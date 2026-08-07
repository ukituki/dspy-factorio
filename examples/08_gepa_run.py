#!/usr/bin/env python3
"""GEPA run: load a compiled module and roll out in Factorio.

Pair with ``examples/07_gepa_train.py`` (train/save). Does not optimize.
See ``docs/GEPA_STARTER.md``.

Usage:
    uv run python examples/08_gepa_run.py --steps 3
    uv run python examples/08_gepa_run.py --program .fle/gepa_factorio_agent.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorio_gym.agent import (
    API_HINT,
    AgentConfig,
    load_agent,
    propose_program,
)
from factorio_gym.env import (
    describe_env,
    get_environment_info,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    step_code,
)

DEFAULT_PROGRAM = ".fle/gepa_factorio_agent.json"

BOOTSTRAP = """
print(inspect_inventory())
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", default=DEFAULT_PROGRAM, help="Compiled DSPy module from 07_gepa_train")
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    args = parser.parse_args()

    path = Path(args.program)
    if not path.is_file():
        print(f"Missing compiled program: {path}")
        print("Train first: uv run python examples/07_gepa_train.py --auto light")
        return 1

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = info.get("description") or args.env_id
    print(describe_env(args.env_id))
    print(f"GEPA run | program={path} | model={args.model} | steps={args.steps}")

    agent = load_agent(path, AgentConfig(model=args.model))
    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("\n=== bootstrap ===")
        print(observation[:800])

        for step in range(1, args.steps + 1):
            print(f"\n=== step {step}/{args.steps} ===")
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
            print(observation[:800])
            print(f"reward={reward} done={terminated or truncated}")
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
