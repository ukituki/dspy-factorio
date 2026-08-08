#!/usr/bin/env python3
"""List registered FLE gym environments.

Usage:
    uv run python examples/02_list_environments.py
    uv run python examples/02_list_environments.py --throughput
    uv run python examples/02_list_environments.py --search iron
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_factorio.env import describe_env, list_envs, load_project_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--search", default="")
    args = parser.parse_args()

    load_project_env()
    env_ids = list_envs(throughput_only=args.throughput)
    needle = args.search.lower().strip()
    if needle:
        env_ids = [e for e in env_ids if needle in e.lower() or needle in describe_env(e).lower()]

    print(f"Found {len(env_ids)} environments:\n")
    for env_id in env_ids:
        print(f"  - {describe_env(env_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
