"""Runtime DSPy agent: propose Factorio programs from observations.

Optimization (teleprompting / compiling demos) lives elsewhere — see
``examples/05_optimize_agent.py`` and ``dspy_factorio.trainset``. Runtime only
builds or loads a module; it never compiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dspy


# Short API reminder passed as inventory_hint when the caller has nothing better.
API_HINT = (
    "Use enums + kwargs only. nearest(Resource.IronOre) — never nearest(\"iron-ore\"). "
    "move_to(iron) or move_to(position=iron) — NEVER move_to(pos=...) "
    "(TypeError: unexpected keyword argument 'pos'). "
    "Call move_to before place_entity when far. "
    "place_entity(entity=Prototype.BurnerMiningDrill, position=iron, direction=Direction.NORTH). "
    "insert_item(Prototype.Coal, drill, quantity=5). Always print()."
)


class FactorioProgrammer(dspy.Signature):
    """Write one short Factorio Learning Environment (FLE) Python program.

    You are inside the FLE REPL. Emit executable Python only — no markdown.

    Hard API rules (wrong form crashes):
    - Resources: nearest(Resource.IronOre) / Resource.Coal — NEVER nearest("iron-ore")
    - Entities: Prototype.BurnerMiningDrill, Prototype.WoodenChest, Prototype.TransportBelt,
      Prototype.Coal — NEVER place_entity("mining-drill", ...)
    - Movement: move_to(iron) or move_to(position=iron). First arg is Position.
      NEVER move_to(pos=...) — that raises TypeError.
      Prefer: iron = nearest(Resource.IronOre); move_to(iron)
    - Placement: place_entity(entity=Prototype.X, position=pos, direction=Direction.NORTH)
    - Distance: call move_to(...) before place_entity if the target is far from the player
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


def _gpt5_fixed_temperature_only(model: str) -> bool:
    """LiteLLM rejects non-1 temperature on most gpt-5* models.

    gpt-5.1 supports custom temperature when reasoning_effort is unset/'none'.
    gpt-5 / gpt-5.5 / gpt-5-codex only accept temperature=1.
    """
    name = model.lower().rsplit("/", 1)[-1]
    if name.startswith("gpt-5.1"):
        return False
    return name.startswith("gpt-5")


def build_lm(config: AgentConfig | None = None) -> dspy.LM:
    cfg = config or AgentConfig()
    temperature = 1.0 if _gpt5_fixed_temperature_only(cfg.model) else cfg.temperature
    return dspy.LM(
        cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=temperature,
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
