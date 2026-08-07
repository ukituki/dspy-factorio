#!/usr/bin/env python3
"""Hello World — connect to Factorio and print nearest iron ore.

Requires a running cluster:
    uv run fle cluster start -n 1

Usage:
    uv run python examples/01_hello_world.py
    uv run python examples/01_hello_world.py --env-id iron_plate_throughput
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorio_gym.env import (
    describe_env,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    step_code,
)


HELLO_CODE = """
# Locate the nearest iron ore patch and print its position.
pos = nearest(Resource.IronOre)
print(f"Nearest iron ore: {pos}")
print(f"Inventory: {inspect_inventory()}")
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    args = parser.parse_args()

    load_project_env()
    print(describe_env(args.env_id))
    print("Creating environment (first connect injects Lua tools — can take a minute)...")

    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        obs, _info = reset_env(env)
        print("Reset OK. Observation keys:", sorted(obs.keys()))

        obs, reward, terminated, truncated, info = step_code(env, HELLO_CODE)
        print("--- program output ---")
        print(obs_text(obs) or "(no raw_text)")
        print("---")
        print(f"reward={reward} terminated={terminated} truncated={truncated}")
        if "output_game_state" in info:
            print("info contains output_game_state")
    finally:
        env.close()

    print("Hello World complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
