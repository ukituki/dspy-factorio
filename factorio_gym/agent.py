"""Minimal DSPy agent that emits Factorio Python programs from observations."""

from __future__ import annotations

from dataclasses import dataclass

import dspy
from dspy.teleprompt import LabeledFewShot

# Short API reminder passed as inventory_hint when the caller has nothing better.
API_HINT = (
    "Use enums + kwargs only. nearest(Resource.IronOre) — never nearest(\"iron-ore\"). "
    "move_to(pos) before place_entity. "
    "place_entity(entity=Prototype.BurnerMiningDrill, position=pos, direction=Direction.NORTH). "
    "insert_item(Prototype.Coal, drill, quantity=5). Always print()."
)

# Seed demos shared by the online loop and offline optimizer.
SEED_DEMOS = [
    dspy.Example(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="Environment reset. No prior output.",
        inventory_hint="burner-mining-drill=50, coal=500, transport-belt=500",
        program=(
            "iron = nearest(Resource.IronOre)\n"
            "print(f'iron={iron}')\n"
            "move_to(iron)\n"
            "print(inspect_inventory())"
        ),
    ).with_inputs("goal", "observation", "inventory_hint"),
    dspy.Example(
        goal="Create an automatic iron-ore factory that produces 16 iron-ore per 60 seconds.",
        observation="iron=Position(x=15.5, y=70.5)\nInventory(... burner-mining-drill=50, coal=500 ...)",
        inventory_hint="player is at origin; iron is far — must move_to before placing",
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
    ).with_inputs("goal", "observation", "inventory_hint"),
    dspy.Example(
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
    ).with_inputs("goal", "observation", "inventory_hint"),
    dspy.Example(
        goal="Find iron ore and report position.",
        observation="(empty)",
        inventory_hint="(none)",
        program=(
            "pos = nearest(Resource.IronOre)\n"
            "print(f'iron={pos}')\n"
            "print(inspect_inventory())"
        ),
    ).with_inputs("goal", "observation", "inventory_hint"),
]


class FactorioProgrammer(dspy.Signature):
    """Write one short Factorio Learning Environment (FLE) Python program.

    You are inside the FLE REPL. Emit executable Python only — no markdown.

    Hard API rules (wrong form crashes):
    - Resources: nearest(Resource.IronOre) / Resource.Coal — NEVER nearest("iron-ore")
    - Entities: Prototype.BurnerMiningDrill, Prototype.WoodenChest, Prototype.TransportBelt,
      Prototype.Coal — NEVER place_entity("mining-drill", ...)
    - Placement: place_entity(entity=Prototype.X, position=pos, direction=Direction.NORTH)
    - Distance: call move_to(pos) before place_entity if the target is far from the player
    - Fuel: insert_item(Prototype.Coal, drill, quantity=5)
    - Belts: connect_entities(source, target, connection_type=Prototype.TransportBelt)
    - Inspect: get_entities(), inspect_inventory(), sleep(n)

    Policy: one small verifiable step per program. Always print() something useful.
    If the last observation shows an Exception, fix that exact error — do not repeat it.
    If drills already exist, fuel them / place a chest at drop_position / connect belts —
    do not place another drill on the same tile.
    """

    goal: str = dspy.InputField(desc="Task goal / quota description")
    observation: str = dspy.InputField(desc="Last stdout/stderr from Factorio REPL")
    inventory_hint: str = dspy.InputField(desc="Inventory / API reminders for this step")
    program: str = dspy.OutputField(desc="Executable Factorio Python program only")


@dataclass
class AgentConfig:
    model: str = "openai/gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.2


def build_lm(config: AgentConfig | None = None) -> dspy.LM:
    cfg = config or AgentConfig()
    return dspy.LM(
        cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def build_agent(config: AgentConfig | None = None) -> dspy.Module:
    """Return a Predict module with labeled FLE API demos."""
    lm = build_lm(config)
    dspy.configure(lm=lm)
    student = dspy.Predict(FactorioProgrammer)
    return LabeledFewShot(k=len(SEED_DEMOS)).compile(student, trainset=SEED_DEMOS)


def strip_code_fences(text: str) -> str:
    """Remove accidental markdown fences from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop opening fence
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def propose_program(
    agent: dspy.Module,
    *,
    goal: str,
    observation: str,
    inventory_hint: str = "",
) -> str:
    result = agent(
        goal=goal,
        observation=observation or "(empty observation)",
        inventory_hint=inventory_hint or API_HINT,
    )
    return strip_code_fences(result.program)
