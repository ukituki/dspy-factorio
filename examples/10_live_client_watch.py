#!/usr/bin/env python3
"""Watch agent actions live in the Factorio desktop client.

Primary visualization path for this workspace: join the Docker Factorio
server from the game client, then run a slow scripted scenario so you can see
placements happen.

Requires:
    - `uv run fle cluster start -n 1`
    - Factorio desktop client ≈ 2.0.73 (match the Docker image)

Usage:
    # Fix whitelist + print connect instructions, then run the miner slowly
    uv run python examples/10_live_client_watch.py

    # Only patch the cluster so you can join (no scenario)
    uv run python examples/10_live_client_watch.py --prepare-only

    # Skip recreate if you already prepared
    uv run python examples/10_live_client_watch.py --no-recreate --pause 8
"""

from __future__ import annotations

import argparse
import sys
import time
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
from dspy_factorio.live_client import connect_instructions, prepare_live_client

STEPS = [
    """
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
print(inspect_inventory())
""".strip(),
    """
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
    parser.add_argument(
        "--pause",
        type=float,
        default=5.0,
        help="Seconds to wait after each step so you can watch in-client",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only make the server joinable; do not run the scenario",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do not docker compose recreate (use if already prepared)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume the cluster is already joinable",
    )
    args = parser.parse_args()

    if not args.skip_prepare:
        info = prepare_live_client(recreate=not args.no_recreate)
        print("Live-client prepare:")
        for key in (
            "connect",
            "server_version",
            "whitelist_disabled",
            "recreated",
            "created_files",
        ):
            print(f"  {key}: {info[key]}")
        print()

    print(connect_instructions())
    if args.prepare_only:
        return 0

    input("Connect the Factorio client, then press Enter to start the scenario… ")

    load_project_env()
    print(describe_env(args.env_id))
    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        print(f"Reset done — watching pause {args.pause}s…")
        time.sleep(args.pause)

        for i, code in enumerate(STEPS, start=1):
            print(f"\n=== step {i}/{len(STEPS)} ===")
            print(code)
            obs, reward, terminated, truncated, _info = step_code(env, code)
            print("--- output ---")
            print(obs_text(obs) or "(no raw_text)")
            print(f"reward={reward} done={terminated or truncated}")
            print(f"(pause {args.pause}s — look at the Factorio client)")
            time.sleep(args.pause)
            if terminated or truncated:
                break
    finally:
        env.close()

    print("\nScenario finished. Leave the client connected for further agent runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
