"""Fixed and learned planning, evidence retrieval, and candidate ranking."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Literal

import dspy

from dspy_factorio.iron_domain import (
    BLOCKED,
    Candidate,
    Decision,
    Goal,
    WorldView,
    iron_drills,
    output_chest,
    plan_fixed,
    progress_label,
)

PROGRESS = {
    Goal.ESTABLISH: "An iron drill becomes fueled, or the FLE task succeeds.",
    Goal.OUTPUT: "The number of iron drills with an open chest output increases, or the FLE task succeeds.",
    Goal.SCALE: "Measured iron-ore throughput rises, or the FLE task succeeds.",
}


class ChooseIronSubgoal(dspy.Signature):
    """Choose the next subgoal for an automatic iron-ore factory.

    Use the measured game state and recent failures. Establish extraction before
    scaling. Restore output when an established drill is blocked or has no chest.
    If a choice stalled, pick a useful alternative. Choose one allowed subgoal.
    """

    world: dict = dspy.InputField()
    completion_conditions: dict = dspy.InputField()
    recent_outcomes: list[dict] = dspy.InputField()
    subgoal: Literal[
        "establish_extraction", "restore_output", "increase_throughput"
    ] = dspy.OutputField()
    rationale: str = dspy.OutputField(
        desc="Brief explanation tied to observed game evidence"
    )


@dataclass(frozen=True)
class Plan:
    goal: Goal
    rationale: str = ""


Planner = Callable[[WorldView, list[Decision]], Plan]


def fixed_planner(view: WorldView, history: list[Decision]) -> Plan:
    return Plan(plan_fixed(view))


class DSPyPlanner:
    def __init__(self, predictor=None):
        self.predictor = (
            predictor if predictor is not None else dspy.Predict(ChooseIronSubgoal)
        )

    def __call__(self, view: WorldView, history: list[Decision]) -> Plan:
        prediction = self.predictor(
            world=world_context(view),
            completion_conditions=PROGRESS,
            recent_outcomes=[
                {
                    "subgoal": entry.goal.value,
                    "status": entry.result.status,
                    "reason": (entry.result.reason or "")[:240],
                }
                for entry in history[-4:]
            ],
        )
        return Plan(Goal(prediction.subgoal), prediction.rationale)


def world_context(view: WorldView) -> dict:
    return view.model_dump(
        mode="json", exclude={"episode_id", "version"}, exclude_none=True
    )


def condition(view: WorldView) -> str:
    drills = iron_drills(view)
    if not drills:
        return "no_iron_drill"
    if not any(drill.fueled for drill in drills):
        return "needs_fuel"
    if any(e.status in BLOCKED or not output_chest(view, e) for e in drills):
        return "needs_output"
    return "has_working_layout"


@dataclass(frozen=True)
class Knowledge:
    findings: tuple[dict, ...] = ()

    @property
    def version(self) -> str:
        content = json.dumps(self.findings, sort_keys=True, separators=(",", ":"))
        return sha256(content.encode()).hexdigest()[:16]

    def retrieve(
        self, view: WorldView, goal: Goal, choices: tuple[Candidate, ...]
    ) -> tuple[dict, ...]:
        applicable = {choice.action.kind.value for choice in choices}
        current = condition(view)
        return tuple(
            fact
            for fact in self.findings
            if fact["goal"] == goal.value
            and fact["condition"] == current
            and fact["action_kind"] in applicable
        )


def learn(episodes: list[dict]) -> Knowledge:
    """Aggregate verified effects on exploration starts, with episode provenance."""
    supporting = defaultdict(list)
    for episode in episodes:
        if (
            episode.get("split") != "train"
            or episode.get("scenario") != "iron_ore_throughput"
        ):
            continue
        decisions = [
            Decision.model_validate(step["decision"])
            for step in episode["steps"]
            if "decision" in step
        ]
        for index, decision in enumerate(decisions):
            if decision.result.status not in {"applied", "failed", "unverified"}:
                continue
            chosen = next(c for c in decision.candidates if c.id == decision.selected)
            key = (
                str(decision.goal),
                condition(decision.before),
                str(chosen.action.kind),
                chosen.expected,
            )
            supporting[key].append(
                (
                    episode["id"],
                    decision.result.status == "applied",
                    progress_label(decisions, index, 3),
                )
            )
    return Knowledge(
        tuple(
            {
                "goal": goal,
                "condition": context,
                "action_kind": action,
                "expected_effect": effect,
                "support": sorted({episode for episode, _, _ in evidence}),
                "attempts": len(evidence),
                "verified_effects": sum(verified for _, verified, _ in evidence),
                "progress_within_three_steps": sum(
                    progress is True for _, _, progress in evidence
                ),
                "progress_outcomes_observed": sum(
                    progress is not None for _, _, progress in evidence
                ),
            }
            for (goal, context, action, effect), evidence in sorted(supporting.items())
        )
    )


@dataclass(frozen=True)
class Selection:
    candidate_id: str
    scores: dict[str, float] | None = None


Selector = Callable[
    [WorldView, Goal, tuple[Candidate, ...], tuple[dict, ...]], Selection
]


def scripted(
    view: WorldView, goal: Goal, choices: tuple[Candidate, ...], facts: tuple[dict, ...]
) -> Selection:
    return Selection(choices[0].id)


class JevSelector:
    """Rank candidates by their binary probability of making benchmark progress."""

    def __init__(self, client, model: str, horizon: int = 3):
        self.client = client
        self.model = model
        self.horizon = horizon

    def __call__(
        self,
        view: WorldView,
        goal: Goal,
        choices: tuple[Candidate, ...],
        facts: tuple[dict, ...],
    ) -> Selection:
        from typesafe_sdk import Noul

        # API-facing keys are short. Map them back to the exact episode-scoped IDs.
        keys = {f"action_{i}": choice for i, choice in enumerate(choices, start=1)}
        reply = self.client.system_one(
            model=self.model,
            state={
                "world": world_context(view),
                "subgoal": str(goal),
                "completion_condition": PROGRESS[goal],
                "past_observations": list(facts),
                "candidates": {
                    key: choice.action.model_dump(mode="json")
                    for key, choice in keys.items()
                },
            },
            questions={
                key: Noul(
                    instructions=(
                        f"If {key} is selected now and the same policy continues, will the completion "
                        f"condition be observed within {self.horizon} scored FLE steps including this action?"
                    )
                )
                for key in keys
            },
        )
        scores = {choice.id: reply.nouls[key].noul for key, choice in keys.items()}
        if any(not isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
            raise ValueError("Jev returned an invalid progress probability")
        return Selection(max(scores, key=scores.__getitem__), scores)


def summarize_knowledge(facts: tuple[dict, ...]) -> tuple[dict, ...]:
    """Expose support counts to the model while preserving provenance in the snapshot."""
    return tuple({**fact, "support": len(fact["support"])} for fact in facts)
