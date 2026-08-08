#!/usr/bin/env python3
"""Thin wrapper around `fle inspect-eval` with sane local defaults.

Usage:
    uv run python examples/06_run_inspect_eval.py
    uv run python examples/06_run_inspect_eval.py --env-id iron_plate_throughput --steps 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import dotenv_values

from dspy_factorio.env import load_project_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--steps", type=int, default=8, help="Trajectory length")
    parser.add_argument("--epochs", type=int, default=1, help="Use 1 with a single Factorio container")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    load_project_env()
    vals = dotenv_values(".env")
    key = vals.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key or key in {"XXX", ""}:
        print("OPENAI_API_KEY missing in .env", file=sys.stderr)
        return 1
    os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("FACTORIO_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("FACTORIO_SERVER_PORT", "27000")

    cmd = [
        "uv",
        "run",
        "fle",
        "inspect-eval",
        "--env-id",
        args.env_id,
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--epochs",
        str(args.epochs),
        "--trajectory-length",
        str(args.steps),
        "--max-connections",
        "1",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
