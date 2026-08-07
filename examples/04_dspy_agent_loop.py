#!/usr/bin/env python3
"""LLM agent loop: observation → DSPy program → Factorio step.

Requires OPENAI_API_KEY in `.env` and a running Factorio cluster.

Usage:
    uv run python examples/04_dspy_agent_loop.py --steps 5
    uv run python examples/04_dspy_agent_loop.py --env-id iron_plate_throughput --model openai/gpt-4o-mini
    uv run python examples/04_dspy_agent_loop.py --load .fle/optimized_factorio_agent.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from factorio_gym.agent import (
    API_HINT,
    AgentConfig,
    FactorioProgrammer,
    build_agent,
    build_lm,
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

# First REPL action: give the LM a real inventory + iron position, not an empty reset.
BOOTSTRAP = """
print(inspect_inventory())
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--load",
        default="",
        help="Optional path to a DSPy module saved by examples/05_optimize_agent.py",
    )
    args = parser.parse_args()

    load_project_env()
    info = get_environment_info(args.env_id) or {}
    goal = info.get("description") or args.env_id
    print(describe_env(args.env_id))
    print(f"Model: {args.model} | steps={args.steps}")

    if args.load:
        dspy.configure(lm=build_lm(AgentConfig(model=args.model)))
        agent = dspy.Predict(FactorioProgrammer)
        agent.load(args.load)
        print(f"Loaded optimized module from {args.load}")
    else:
        agent = build_agent(AgentConfig(model=args.model))

    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        obs, _, _, _, _ = step_code(env, BOOTSTRAP)
        observation = obs_text(obs) or "Environment reset. No prior output."
        print("\n=== bootstrap ===")
        print(observation[:1200])

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
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
