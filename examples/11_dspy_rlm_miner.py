#!/usr/bin/env python3
"""Intro: dspy.RLM places a fueled burner mining drill on iron ore.

Unlike example 04 (outer loop: Predict → step → Predict…), here one RLM call
owns the multi-step exploration. The LLM writes Python in a Deno/Pyodide REPL
and calls ``run_factorio(code)`` to execute FLE programs against the live game.

Requires: OPENAI_API_KEY, Deno (`brew install deno`), Factorio cluster.

Usage:
    uv run python examples/11_dspy_rlm_miner.py
    uv run python examples/11_dspy_rlm_miner.py --verbose --max-iters 12
    uv run python examples/11_dspy_rlm_miner.py --renders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from dspy_factorio.agent import API_HINT, AgentConfig, build_lm
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

# Compact API card injected as an RLM input variable (not stuffed into the
# system prompt). The REPL can print / slice it; keep it short for tutorials.
API_DOCS = f"""
TWO SANDBOXES (do not mix them):
- Outer RLM REPL (where you write code now): ONLY run_factorio, print, llm_query, SUBMIT.
  nearest / Resource / Prototype / move_to / place_entity do NOT exist here.
- Inner Factorio (only inside the STRING you pass to run_factorio): those FLE names ARE builtins.
  No imports. No `game.` prefix. No `def main` / `def run` wrappers — top-level statements only.

CORRECT shape (copy this pattern):
out = run_factorio('''
iron = nearest(Resource.IronOre)
print(iron)
move_to(iron)
iron = nearest(Resource.IronOre)
drill = next((e for e in get_entities() if e.name == "burner-mining-drill"), None)
if drill is None:
    drill = place_entity(entity=Prototype.BurnerMiningDrill, position=iron, direction=Direction.NORTH)
insert_item(Prototype.Coal, drill, quantity=5)
print(drill)
print(get_entities())
''')
print(out)

WRONG (causes NameError in the outer REPL): nearest(Resource.IronOre) at top level.
WRONG: from fle.env import *, game.nearest(...), def main(game): ...
Do NOT use llm_query to invent FLE imports — trust this card.

CRITICAL: Factorio state does NOT carry across run_factorio calls. Re-fetch iron/drill every time.

{API_HINT}

If a drill already exists, do NOT place again — get_entities() + insert_item.
Always print() inside the Factorio string. Fix Exceptions from the last out — do not repeat them.
""".strip()


class PlaceBurnerDrill(dspy.Signature):
    """Place and fuel one burner mining drill on iron ore.

    You are in the outer RLM Python REPL. Factorio tools are NOT in scope here.
    The ONLY way to touch the game is:

        out = run_factorio('''...FLE python...''')
        print(out)

    Put nearest/Resource/Prototype/move_to/place_entity/insert_item ONLY inside that
    string. Never call them at top level. Never import fle. Never wrap in def main/run.
    Do not use llm_query to rediscover the API — use the `api_docs` variable.

    Loop: run_factorio → read out → fix errors → when a fueled burner-mining-drill is
    visible in out, SUBMIT(summary=..., success=True).

    Each run_factorio is a fresh Factorio program (re-bind iron/drill every call).
    Prefer one combined place+fuel program after move_to.
    """

    goal: str = dspy.InputField(desc="Human-readable task goal")
    api_docs: str = dspy.InputField(
        desc="FLE API card; print(api_docs) if needed. Names apply ONLY inside run_factorio strings."
    )
    summary: str = dspy.OutputField(desc="What you did and evidence from Factorio output")
    success: bool = dspy.OutputField(desc="True iff a fueled burner drill was placed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="iron_ore_throughput")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--sub-model",
        default=None,
        help="Cheaper model for llm_query inside RLM (default: same as --model)",
    )
    parser.add_argument("--max-iters", type=int, default=12)
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=5,
        help="Cap llm_query* (default 5 — high values invite hallucinated FLE APIs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every RLM REPL thought/code/stdout step",
    )
    parser.add_argument(
        "--renders",
        action="store_true",
        help="Save a schematic map PNG after each run_factorio call",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path(".fle/renders/rlm_miner"),
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
    # Tutorial goal is narrower than the full throughput quota — teach RLM first.
    goal = (
        "Place one BurnerMiningDrill on nearest iron ore, move_to first if needed, "
        "insert 5 coal, and print entities/inventory as proof. "
        f"(Full env description: {info.get('description') or args.env_id})"
    )
    print(describe_env(args.env_id))
    print(
        f"RLM miner | model={args.model} | max_iters={args.max_iters} "
        f"| max_llm_calls={args.max_llm_calls}"
    )

    env = make_env(args.env_id, run_idx=args.run_idx)
    step_count = {"n": 0}

    def run_factorio(code: str) -> str:
        """Execute one FLE Python program in Factorio; return stdout/stderr text.

        Args:
            code: Executable Factorio Learning Environment Python (no markdown).
        """
        step_count["n"] += 1
        n = step_count["n"]
        print(f"\n=== run_factorio #{n} ===")
        print(code.strip())
        obs, reward, terminated, truncated, _info = step_code(env, code)
        text = obs_text(obs) or "(empty)"
        print("--- factorio output ---")
        print(text[:1500])
        print(f"reward={reward} done={terminated or truncated}")
        if args.renders:
            path = save_render(
                env,
                args.render_dir / f"step_{n:02d}.png",
                mode="simple",
                zoom=args.render_zoom,
                overview=True,
            )
            print(f"saved {path}")
        # Keep tool returns small so the RLM REPL stays focused.
        return (
            f"step={n} reward={reward} done={terminated or truncated}\n"
            f"{text[:4000]}"
        )

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

        lm = build_lm(AgentConfig(model=args.model))
        dspy.configure(lm=lm)
        sub_lm = (
            build_lm(AgentConfig(model=args.sub_model))
            if args.sub_model
            else None
        )

        rlm = dspy.RLM(
            PlaceBurnerDrill,
            max_iters=args.max_iters,
            max_llm_calls=args.max_llm_calls,
            verbose=args.verbose,
            tools=[run_factorio],
            sub_lm=sub_lm,
        )

        print("\n=== RLM start ===")
        result = rlm(goal=goal, api_docs=API_DOCS)
        print("\n=== RLM result ===")
        print(f"success={result.success}")
        print(f"summary={result.summary}")
        print(f"factorio_steps={step_count['n']}")
    finally:
        env.close()

    if args.renders:
        print(f"\nDone. Open PNGs under {args.render_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
