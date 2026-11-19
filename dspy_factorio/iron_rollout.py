"""Run a reproducible policy against a single iron-ore episode."""

from collections.abc import Callable
from typing import Protocol

from dspy_factorio.iron_domain import (
    RECOVERABLE,
    Candidate,
    Decision,
    Result,
    WorldView,
    candidates,
    iron_drills,
    reached_goal,
)
from dspy_factorio.iron_policy import (
    Knowledge,
    Planner,
    Selection,
    Selector,
    fixed_planner,
    scripted,
    summarize_knowledge,
)


class Game(Protocol):
    def execute(self, view: WorldView, candidate: Candidate) -> Result: ...


def run_typed(
    game: Game,
    start: WorldView,
    budget: int,
    *,
    milestone: bool = False,
    planner: Planner = fixed_planner,
    selector: Selector = scripted,
    selector_name: str = "scripted",
    knowledge: Knowledge = Knowledge(),
    on_step: Callable[[Decision, str], None] = lambda *_: None,
    first_action: int | None = None,
) -> str:
    view = start
    remaining = budget
    history: list[Decision] = []
    excluded: set[str] = set()
    plan = None
    plan_start = start
    plan_steps = 0
    while remaining > 0:
        if view.task_success:
            return "throughput verified"
        if milestone and any(drill.fueled for drill in iron_drills(view)):
            return "fueled drill verified"
        if plan is None:
            plan = planner(view, history)
            plan_start, plan_steps = view, 0
        eligible = candidates(view, plan.goal, frozenset(excluded), milestone=milestone)
        if not eligible:
            return "no eligible candidates"
        facts = knowledge.retrieve(view, plan.goal, eligible)
        if first_action is not None and not history:
            if not 1 <= first_action <= len(eligible):
                raise ValueError(
                    "Requested first action is outside the eligible candidate list"
                )
            selection = Selection(eligible[first_action - 1].id)
        else:
            selection = selector(view, plan.goal, eligible, summarize_knowledge(facts))
        chosen = next((c for c in eligible if c.id == selection.candidate_id), None)
        if chosen is None:
            raise ValueError(
                "Selector returned an ID outside the current observation's candidates"
            )
        # This runner is the only game writer. Scores never grant permission to
        # construct a different action or execute an ID from an earlier view.
        result = game.execute(view, chosen)
        scores = selection.scores or {}
        decision = Decision(
            before=view,
            goal=plan.goal,
            candidates=eligible,
            selected=chosen.id,
            result=result,
            selector=selector_name,
            knowledge_version=knowledge.version,
            knowledge=facts,
            progress_probability=scores.get(chosen.id),
            candidate_probabilities=scores,
        )
        history.append(decision)
        on_step(decision, plan.rationale)
        remaining -= result.fle_steps
        plan_steps += result.fle_steps
        if result.after is None or result.status == "unknown":
            return "game outcome unknown"
        view = result.after
        if view.task_success:
            return "throughput verified"
        if milestone and any(drill.fueled for drill in iron_drills(view)):
            return "fueled drill verified"
        if result.terminated or result.truncated:
            return "environment stopped"
        if result.status in RECOVERABLE:
            excluded.add(chosen.action.key())
        if (
            reached_goal(plan.goal, plan_start, view) is True
            or result.status in RECOVERABLE
            or plan_steps >= 4
        ):
            plan = None
    return "FLE step budget reached"
