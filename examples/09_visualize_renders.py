#!/usr/bin/env python3
"""Visualize agent actions by saving a map PNG after each step.

Uses FLE's schematic renderer by default (no sprite download). For Factorio-like
pixels: ``uv run fle sprites`` then pass ``--mode sprites``.

Requires a running cluster:
    uv run fle cluster start -n 1

Usage:
    uv run python examples/09_visualize_renders.py
    uv run python examples/09_visualize_renders.py --mode sprites --out .fle/renders/sprites
    open .fle/renders/simple/step_02.png
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
    save_render,
    step_code,
)

# Same deterministic plan as examples/03_scripted_miner.py.
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
        "--mode",
        choices=("simple", "sprites"),
        default="simple",
        help="simple=schematic (default); sprites needs `fle sprites`",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: .fle/renders/<mode>)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.25,
        help="Zoom for simple mode (<1 = wider view). Default 0.25 ≈ 80 tiles",
    )
    args = parser.parse_args()
    out_dir = args.out or Path(".fle/renders") / args.mode

    load_project_env()
    print(describe_env(args.env_id))
    zoom = args.zoom if args.mode == "simple" else None
    print(f"Renders → {out_dir.resolve()} (mode={args.mode}, zoom={zoom})")

    env = make_env(args.env_id, run_idx=args.run_idx)
    try:
        reset_env(env)
        path = save_render(
            env, out_dir / "step_00_reset.png", mode=args.mode, zoom=zoom, overview=True
        )
        print(f"saved {path}")

        for i, code in enumerate(STEPS, start=1):
            print(f"\n=== step {i}/{len(STEPS)} ===")
            print(code)
            obs, reward, terminated, truncated, _info = step_code(env, code)
            print("--- output ---")
            print(obs_text(obs) or "(no raw_text)")
            print(f"reward={reward} done={terminated or truncated}")

            path = save_render(
                env,
                out_dir / f"step_{i:02d}.png",
                mode=args.mode,
                zoom=zoom,
                overview=args.mode == "simple",
            )
            print(f"saved {path}")

            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"\nDone. Open PNGs under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
