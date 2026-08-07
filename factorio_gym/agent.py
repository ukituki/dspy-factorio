"""Runtime DSPy agent: propose Factorio programs from observations.

Optimization (teleprompting / compiling demos) lives elsewhere — see
``examples/05_optimize_agent.py`` and ``factorio_gym.trainset``. Runtime only
builds or loads a module; it never compiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dspy


# Short API reminder passed as inventory_hint when the caller has nothing better.
API_HINT = (
    "Use enums + kwargs only. nearest(Resource.IronOre) — never nearest(\"iron-ore\"). "
    "move_to(pos) before place_entity. "
    "place_entity(entity=Prototype.BurnerMiningDrill, position=pos, direction=Direction.NORTH). "
    "insert_item(Prototype.Coal, drill, quantity=5). Always print()."
)


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
    model: str = "openai/gpt-5.1"
    max_tokens: int = 8024
    temperature: float = 0.2


def build_lm(config: AgentConfig | None = None) -> dspy.LM:
    cfg = config or AgentConfig()
    return dspy.LM(
        cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def build_agent(config: AgentConfig | None = None) -> dspy.Module:
    """Return an unoptimized Predict module for online rollouts."""
    dspy.configure(lm=build_lm(config))
    return dspy.Predict(FactorioProgrammer)


def load_agent(
    path: str | Path,
    config: AgentConfig | None = None,
) -> dspy.Module:
    """Load a compiled DSPy module previously saved by the optimizer."""
    dspy.configure(lm=build_lm(config))
    agent = dspy.Predict(FactorioProgrammer)
    agent.load(str(path))
    return agent


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
