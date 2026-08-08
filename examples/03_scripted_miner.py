#!/usr/bin/env python3
"""Scripted scenario: place a burner mining drill on iron ore.

This is a deterministic baseline (no LLM) useful for validating the env and
for collecting trajectories before optimizing an AI policy.

Usage:
    uv run python examples/03_scripted_miner.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_factorio.env import (
    describe_env,
    load_project_env,
    make_env,
    obs_text,
    reset_env,
    step_code,
)

# Multi-step scripted plan. Each string is one REPL action.
STEPS = [
    """
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
print(inspect_inventory())
""".strip(),
    """
# Move close enough, then place a burner mining drill facing north.
iron = nearest(Resource.IronOre)
move_to(iron)
drill = place_entity(
    entity=Prototype.BurnerMiningDrill,
    position=iron,
    direction=Direction.NORTH,
)
print(drill)
""".strip(),
    """
# Fuel the drill if we have coal; otherwise report inventory.
entities = get_entities()
print(entities)
drill = next((e for e in entities if e.name == "burner-mining-drill"), None)
if drill is not None:
    insert_item(Prototype.Coal, drill, quantity=5)
    print(drill)
print(inspect_inventory())
""".strip(),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    args = parser.parse_args()

    load_project_env()
    print(describe_env(args.env_id))
    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        for i, code in enumerate(STEPS, start=1):
            print(f"\n=== step {i}/{len(STEPS)} ===")
            print(code)
            obs, reward, terminated, truncated, _info = step_code(env, code)
            print("--- output ---")
            print(obs_text(obs) or "(no raw_text)")
            print(f"reward={reward} done={terminated or truncated}")
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
