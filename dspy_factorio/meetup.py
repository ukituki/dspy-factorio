"""Harness choices and usage accounting for the meetup notebook."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import dspy

from dspy_factorio.agent import FactorioProgrammer


class StateAwareProgrammer(dspy.Signature):
    """Advance the supplied Factorio goal with one observable state change.

    Use the supplied FLE API reminders. Emit top-level FLE Python without
    imports, markdown, or main wrappers. Re-fetch entities and positions in
    each program; do not assume Python variables survive between actions.
    If the observation contains an error, address its cause first. If a machine
    exists, inspect its fuel, inputs and output blockage before adding another.
    After acting, print fresh entities and inventory so the next turn can use
    game evidence. Do not claim production or success without measured evidence.
    State a brief expected result that can be checked against the game output.
    """

    goal: str = dspy.InputField(desc="Scenario goal and production quota")
    observation: str = dspy.InputField(desc="Latest game output, including errors")
    inventory_hint: str = dspy.InputField(desc="Available FLE API reminders")
    expected_result: str = dspy.OutputField(desc="One observable change to check after execution")
    program: str = dspy.OutputField(desc="One executable FLE Python program")


SIGNATURES: dict[str, type[dspy.Signature]] = {
    "Baseline": FactorioProgrammer,
    "State-aware": StateAwareProgrammer,
}

MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "openai/gpt-5.1",
    "openai/gpt-6-astra",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "Custom model",
]


def build_harness(module_name: str, signature_name: str) -> dspy.Module:
    signature = SIGNATURES[signature_name]
    if module_name == "Predict":
        return dspy.Predict(signature)
    if module_name == "ChainOfThought":
        return dspy.ChainOfThought(
            signature,
            rationale_field=dspy.OutputField(
                desc="Brief action rationale: what the observation implies, "
                "what to do next, and what game evidence to check."
            ),
        )
    raise ValueError(f"Unknown module: {module_name}")


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _token_count(usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def summarize_usage(
    history: Sequence[Mapping[str, Any]], *, incomplete: bool = False
) -> dict[str, Any]:
    """Count all recorded LM responses, including adapter retries.

    Cost is the estimate supplied in DSPy's LM history (via LiteLLM), not an
    invoice. Missing metadata and failed/unrecorded requests leave totals null;
    known subtotals are retained separately. Reasoning tokens are already part
    of completion/output tokens and must not be added again.
    """
    inputs, outputs, totals, costs = [], [], [], []
    for entry in history:
        raw = entry.get("usage")
        usage = raw if isinstance(raw, Mapping) else {}
        prompt = _token_count(usage, "prompt_tokens", "input_tokens")
        completion = _token_count(usage, "completion_tokens", "output_tokens")
        total = _token_count(usage, "total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        inputs.append(prompt)
        outputs.append(completion)
        totals.append(total)
        cost = entry.get("cost")
        costs.append(float(cost) if _number(cost) and cost >= 0 else None)

    def complete_sum(values: list) -> int | float | None:
        if incomplete or any(value is None for value in values):
            return None
        return sum(values)

    return {
        "lm_calls": len(history),
        "input_tokens": complete_sum(inputs),
        "output_tokens": complete_sum(outputs),
        "total_tokens": complete_sum(totals),
        "cost_usd": complete_sum(costs),
        "known_tokens": sum(value for value in totals if value is not None),
        "known_cost_usd": sum(value for value in costs if value is not None),
        "usage_complete": not incomplete and all(value is not None for value in totals),
        "cost_complete": not incomplete and all(value is not None for value in costs),
        "request_incomplete": incomplete,
        "cost_source": "DSPy / LiteLLM response-cost estimate",
    }


def peak_score(steps: Sequence[Mapping[str, Any]]) -> float | None:
    scores = [step["reward"] for step in steps if _number(step.get("reward"))]
    return max(scores) if scores else None



def final_score(steps: Sequence[Mapping[str, Any]]) -> float | None:
    """Last measured reward; pending actions have no measurement."""
    return next((step["reward"] for step in reversed(steps) if _number(step.get("reward"))), None)


def episode_success(steps: Sequence[Mapping[str, Any]]) -> bool | None:
    """Use explicit task verification; legacy done flags are ambiguous."""
    results = [step.get("task_success") for step in steps if isinstance(step.get("task_success"), bool)]
    return any(results) if results else None


def task_success(observation: Mapping[str, Any]) -> bool | None:
    verification = observation.get("task_verification")
    value = verification.get("success") if isinstance(verification, Mapping) else None
    return bool(value) if isinstance(value, (bool, int)) and value in (0, 1) else None


def scenario_guidance(scenario: str) -> tuple[str, int, str]:
    """Suggested tiers from TASKS_GUIDE recipe depth and AI_OPTIMIZATION curriculum.

    These are presentation guidance, not measured benchmark difficulty.
    """
    product = scenario.split("_throughput")[0]
    if product == "iron_ore":
        return "Easy", 16, "Mining only"
    if product in {"iron_plate", "steel_plate", "stone_wall"}:
        return "Medium", 32, "Mining → smelting"
    if product in {"iron_gear_wheel", "electronic_circuit", "inserter", "automation_science_pack"}:
        return "Medium", 64, "Smelting → assembly"
    if product in {"processing_unit", "low_density_structure", "production_science_pack", "utility_science_pack", "chemical_science_pack"}:
        return "Expert", 256, "Deep recipe chains + multiple production lines"
    if product in {"crude_oil", "petroleum_gas", "plastic_bar", "sulfur", "sufuric_acid", "sulfuric_acid", "battery", "advanced_circuit", "engine_unit", "logistics_science_pack", "military_science_pack", "piercing_round"}:
        return "Hard", 128, "Multiple production stages and/or fluid handling"
    return "Unrated", 64, "Default task horizon"

def comparison_row(episode: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Keep legacy episodes reviewable without inventing token/cost data."""
    usage = episode.get("usage") or {}
    signature = episode.get("signature", "Baseline")
    cost = usage.get("cost_usd")
    return {
        "Episode": index,
        "Scenario": episode.get("scenario", "iron_ore_throughput"),
        "Model": episode["model"],
        "Harness": f"{episode['module']} × {signature}",
        "Final score": final_score(episode["steps"]),
        "Max score": peak_score(episode["steps"]),
        "Success?": {True: "Yes", False: "No", None: "Unknown"}[episode_success(episode["steps"])],
        "Input tokens": usage.get("input_tokens"),
        "Output tokens": usage.get("output_tokens"),
        "Tokens": usage.get("total_tokens"),
        "Est. USD": round(cost, 6) if cost is not None else None,
        "Budget": episode.get("budget"),
        "Agent steps": sum(step["index"] > 0 for step in episode["steps"]),
        "Seconds": episode["elapsed"],
        "Stop reason": episode["status"],
    }
