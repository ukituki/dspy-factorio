"""Offline training / validation examples for DSPy teleprompting.

Used only by train scripts (``examples/05_optimize_agent.py``,
``examples/07_gepa_train.py``). Intro/run rollouts never import this module.

GEPA needs a **separate** valset — do not reuse TRAIN_DEMOS for both.
"""

from __future__ import annotations

import dspy


def _ex(
    *,
    goal: str,
    observation: str,
    inventory_hint: str,
    program: str,
) -> dspy.Example:
    return dspy.Example(
        goal=goal,
        observation=observation,
        inventory_hint=inventory_hint,
        program=program,
    ).with_inputs("goal", "observation", "inventory_hint")


# Prefer a larger train set for GEPA; keep validation small but representative.
TRAIN_DEMOS = [
    _ex(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="Environment reset. No prior output.",
        inventory_hint="burner-mining-drill=50, coal=500, transport-belt=500",
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "print(f'iron={iron}')\n"
            "move_to(iron)\n"
            "print(inspect_inventory())"
        ),
    ),
    _ex(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="iron=Position(x=15.5, y=70.5)\nInventory(... burner-mining-drill=50, coal=500 ...)",
        inventory_hint=(
            "player is at origin; iron is far — must move_to(iron) before placing "
            "(never move_to(pos=...))"
        ),
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "move_to(iron)\n"
            "drill = place_entity(\n"
            "    entity=Prototype.BurnerMiningDrill,\n"
            "    position=iron,\n"
            "    direction=Direction.NORTH,\n"
            ")\n"
            "print(drill)"
        ),
    ),
    _ex(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="BurnerMiningDrill(status=NO_FUEL, warnings=['out of fuel'])",
        inventory_hint="coal=500",
        program=(
            "entities = get_entities()\n"
            "print(entities)\n"
            "drill = next(e for e in entities if e.name == 'burner-mining-drill')\n"
            "insert_item(Prototype.Coal, drill, quantity=5)\n"
            "sleep(5)\n"
            "print(get_entities())"
        ),
    ),
    _ex(
        goal="Find iron ore and report position.",
        observation="(empty)",
        inventory_hint="(none)",
        program=(
            "pos = nearest(Resource.IronOre)\n"
            "print(f'iron={pos}')\n"
            "print(inspect_inventory())"
        ),
    ),
    _ex(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="Exception: Could not place burner-mining-drill — too far away",
        inventory_hint="must move_to before place_entity",
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "move_to(iron)\n"
            "print(f'moved to {iron}')"
        ),
    ),
]

# Held-out situations for Pareto selection (disjoint from TRAIN_DEMOS).
VAL_DEMOS = [
    _ex(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="entity already exists at the target position",
        inventory_hint="drills already placed; fuel or add chest instead of placing again",
        program=(
            "entities = get_entities()\n"
            "drill = next(e for e in entities if e.name == 'burner-mining-drill')\n"
            "insert_item(Prototype.Coal, drill, quantity=5)\n"
            "print(drill)"
        ),
    ),
    _ex(
        goal="Find iron ore and report position.",
        observation="Environment reset. No prior output.",
        inventory_hint="(none)",
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "print(f'iron={iron}')"
        ),
    ),
]

# Back-compat alias used by BootstrapFewShot example.
SEED_DEMOS = TRAIN_DEMOS
