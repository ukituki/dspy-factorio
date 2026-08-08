"""Shared helpers for Flex drill tutorials (examples 12 / 13a / 13b).

Narrow milestone: place + fuel one burner mining drill (same as RLM starter).
"""

from __future__ import annotations

from pathlib import Path

import dspy

from factorio_gym.agent import (
    AgentConfig,
    FactorioProgrammer,
    build_lm,
    strip_code_fences,
)

BOOTSTRAP = """
print(inspect_inventory())
iron = nearest(Resource.IronOre)
print(f"iron={iron}")
""".strip()

DRILL_GOAL = (
    "Place one BurnerMiningDrill on nearest iron ore, move_to first if needed, "
    "insert 5 coal, and print entities/inventory as proof. "
    "One small verifiable step per program. Stop once a fueled drill exists."
)

DEFAULT_FLEX_PROGRAM = ".fle/flex_from_play.json"
DEFAULT_PLAY_DEMOS = ".fle/flex_play_demos.json"

# Tiny held-out set for 13a Pareto — drill milestone, not full factory quota.
# Designed so a sloppy proposer (place-only, no fuel / no move_to) scores < 1.0.
FLEX_VAL_DEMOS = [
    dspy.Example(
        goal=DRILL_GOAL,
        observation="BurnerMiningDrill(status=NO_FUEL, warnings=['out of fuel'])",
        inventory_hint="coal=500; drill already placed — fuel it, do not place again",
        program=(
            "entities = get_entities()\n"
            "drill = next(e for e in entities if e.name == 'burner-mining-drill')\n"
            "insert_item(Prototype.Coal, drill, quantity=5)\n"
            "print(drill)\n"
            "print(get_entities())"
        ),
    ).with_inputs("goal", "observation", "inventory_hint"),
    dspy.Example(
        goal=DRILL_GOAL,
        observation="iron=Position(x=15.5, y=70.5)\nException: Could not place burner-mining-drill — too far away",
        inventory_hint="must move_to(iron) before place_entity (never move_to(pos=...))",
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "move_to(iron)\n"
            "print(f'moved to {iron}')"
        ),
    ).with_inputs("goal", "observation", "inventory_hint"),
]


def looks_like_fueled_drill(observation: str, program: str = "") -> bool:
    """Cheap success heuristic from Factorio stdout + the program that produced it.

    Requires fueling evidence. A print like ``Placed Burner Mining Drill`` alone
    must NOT count (that false-positive aborted play before ``insert_item``).
    """
    text = (observation or "").lower()
    compact = text.replace(" ", "").replace("_", "").replace("-", "")
    prog = program or ""

    has_entity = (
        "burnerminingdrill(" in compact
        or "name='burner-mining-drill'" in text
        or 'name="burner-mining-drill"' in text
        or "burner-mining-drill" in text
    )
    fueled_prog = "insert_item" in prog and "Coal" in prog

    if "no_fuel" in text or "out of fuel" in text:
        return False
    # Prefer explicit fueling in the program that produced this observation.
    if fueled_prog:
        return True
    if has_entity and "working" in text:
        return True
    return False


def step_looks_healthy(observation: str) -> bool:
    """True if Factorio stdout does not look like a hard failure."""
    text = observation or ""
    lowered = text.lower()
    if "exception" in lowered or "traceback" in lowered:
        return False
    if "typeerror" in lowered or "nameerror" in lowered:
        return False
    return bool(text.strip()) and text.strip() != "(empty)"


def build_flex_agent(
    config: AgentConfig | None = None,
    program_path: str | Path | None = None,
) -> dspy.Module:
    """Baseline Flex, or load a saved ``module_src`` artifact."""
    dspy.configure(lm=build_lm(config))
    agent = dspy.Flex(FactorioProgrammer)
    if program_path is not None:
        agent.load(str(program_path))
    return agent


def make_play_example(
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
        program=strip_code_fences(program),
    ).with_inputs("goal", "observation", "inventory_hint")


def flex_gepa_metric(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
    program_trace=None,
):
    """Rich-feedback metric for Flex+GEPA (needs critique, not score alone)."""
    program = strip_code_fences(getattr(pred, "program", "") or "")
    notes: list[str] = []
    score = 0.0

    if not program.strip():
        return dspy.Prediction(
            score=0.0,
            feedback="Empty program. Emit executable FLE Python with print().",
        )

    if "print(" in program:
        score += 0.25
    else:
        notes.append("Add print(...) so the REPL returns an observation.")

    if any(
        t in program
        for t in (
            "nearest(",
            "place_entity(",
            "insert_item(",
            "move_to(",
            "get_entities(",
        )
    ):
        score += 0.25
    else:
        notes.append(
            "Call an FLE tool (nearest / move_to / place_entity / insert_item / get_entities)."
        )

    if "Resource." in program or "Prototype." in program:
        score += 0.25
    else:
        notes.append(
            'Use Resource.* / Prototype.* enums — never string names like nearest("iron-ore").'
        )

    bad_string_api = 'nearest("' in program or "nearest('" in program
    if "```" not in program and not bad_string_api:
        score += 0.25
    else:
        notes.append(
            'No markdown fences; never nearest("iron-ore"). Prefer nearest(Resource.IronOre).'
        )

    if "move_to(pos=" in program or "move_to(pos =" in program:
        notes.append(
            "move_to takes Position positionally or as position=... — NEVER move_to(pos=...). "
            "Use: iron = nearest(Resource.IronOre); move_to(iron)"
        )
        score = min(score, 0.5)

    gold_prog = getattr(gold, "program", "") or ""
    goal = (getattr(gold, "goal", "") or "").lower()
    obs = (getattr(gold, "observation", "") or "").lower()

    # Drill milestone: observations that need fueling must emit insert_item.
    needs_fuel = (
        "no_fuel" in obs
        or "out of fuel" in obs
        or "insert" in goal
        or "coal" in goal
        or "fuel" in goal
    )
    if needs_fuel and "insert_item" not in program:
        notes.append(
            "Drill goal / NO_FUEL obs requires insert_item(Prototype.Coal, drill, quantity=5)."
        )
        score = min(score, 0.5)

    if "too far" in obs and "move_to(" not in program:
        notes.append("Observation says too far — call move_to(iron) before placing.")
        score = min(score, 0.5)

    if "move_to(" in gold_prog and "move_to(" not in program:
        notes.append(
            "Gold uses move_to before placement — include move_to when the target is far."
        )
        score = min(score, 0.75)

    if "insert_item" in gold_prog and "insert_item" not in program:
        notes.append("Gold fuels the drill — include insert_item(Prototype.Coal, …).")
        score = min(score, 0.5)

    if program_trace is not None and len(program_trace) > 4:
        notes.append(
            f"program_trace has {len(program_trace)} LM calls — prefer fewer predictors / more Python."
        )
        score = max(0.0, score - 0.05 * (len(program_trace) - 4))

    if score >= 0.99 and not notes:
        feedback = "Good FLE program: enums, tools, and print() present."
    else:
        feedback = " ".join(notes) or "Improve FLE API usage."
        feedback += f" Expected style resembles: {gold_prog[:180]!r}"

    return dspy.Prediction(score=score, feedback=feedback)
