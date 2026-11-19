import json
from pathlib import Path
from types import SimpleNamespace

import dspy
from dspy.utils.dummies import DummyLM
import pytest

from dspy_factorio.iron_domain import (
    Action,
    ActionKind,
    Decision,
    Goal,
    Point,
    Result,
    WorldView,
    candidates,
    plan_fixed,
    verify,
)
from dspy_factorio.iron_eval import calibrate, export_examples
from dspy_factorio.iron_fle import IronFLE, project
from dspy_factorio.iron_policy import DSPyPlanner, JevSelector, Selection, learn
from dspy_factorio.iron_rlcd import TASK, register
from dspy_factorio.iron_rollout import run_typed


@pytest.fixture
def samples():
    return json.loads(
        (Path(__file__).parent / "fixtures/iron_lab_views.json").read_text()
    )


def test_live_fixture_replays_candidates(samples):
    for fixture in samples:
        view = WorldView.model_validate(fixture["view"])
        assert fixture["milestone_candidates"] == [
            candidate.model_dump(mode="json")
            for candidate in candidates(view, plan_fixed(view), milestone=True)
        ]


def test_world_projection_preserves_missing_facts():
    obs = {
        "character_positions": [{"agent_idx": 0, "x": 0, "y": 0}],
        "inventory": [{"type": "coal", "quantity": 20}],
        "entities": [
            {
                "name": "burner-mining-drill",
                "position": {"x": 2, "y": 3},
                "status": "no_fuel",
                "resources": [{"name": "iron-ore"}],
            }
        ],
        "task_verification": {"success": 0, "meta": []},
    }
    state = project(obs, "episode", 2)
    assert state.count("coal") == 20
    assert state.entities[0].fueled is False
    assert state.throughput is None and state.task_success is None
    assert state.entities[0].drop is None


def test_rules_offer_movement_instead_of_distant_placement(samples):
    view = WorldView.model_validate(samples[0]["view"]).model_copy(
        update={"iron": Point(x=40, y=0)}
    )
    offers = candidates(view, Goal.ESTABLISH, milestone=True)
    assert [item.action.kind for item in offers] == [ActionKind.MOVE]
    assert offers[0].id != offers[0].action.candidate_id(
        view.model_copy(update={"version": 5})
    )


def test_candidate_references_stay_stable_when_one_is_excluded(samples):
    view = WorldView.model_validate(samples[1]["view"])
    choices = candidates(view, Goal.ESTABLISH)
    assert choices[0].action.kind == ActionKind.FUEL
    assert all(
        not (
            c.action.kind == ActionKind.DRILL
            and c.action.target == view.entities[0].position
        )
        for c in choices
    )
    second = candidates(view, Goal.ESTABLISH, frozenset({choices[0].action.key()}))
    assert choices[1] in second


def test_reset_uses_fle_spawn_reset_before_gym_reset(monkeypatch):
    calls = []
    instance = SimpleNamespace(reset=lambda **kwargs: calls.append(kwargs))
    game = IronFLE(SimpleNamespace(unwrapped=SimpleNamespace(instance=instance)))

    def gym_reset(_):
        calls.append("gym")
        return {"character_positions": [{"agent_idx": 0, "x": 0, "y": 0}]}, {}

    monkeypatch.setattr("dspy_factorio.iron_fle.reset_env", gym_reset)
    assert game.reset("renewed").position == Point(x=0, y=0)
    assert calls == [{"reset_position": True}, "gym"]


def test_adapter_rejects_stale_before_any_step(samples):
    view = WorldView.model_validate(samples[0]["view"])
    offer = candidates(view, Goal.ESTABLISH)[0]
    game = IronFLE(None)
    with pytest.raises(ValueError, match="stale"):
        game.execute(view, offer)


def test_adapter_does_not_retry_indeterminate_step(samples, monkeypatch):
    view = WorldView.model_validate(samples[0]["view"])
    offer = candidates(view, Goal.ESTABLISH)[0]
    game = IronFLE(None)
    game.current = view
    calls = []

    def timed_out(*args, **kwargs):
        calls.append(args)
        raise TimeoutError("receipt unavailable")

    monkeypatch.setattr("dspy_factorio.iron_fle.step_code", timed_out)
    result = game.execute(view, offer)
    assert len(calls) == result.fle_steps == 1
    assert result.after is None and result.status == "unknown"


def test_dspy_planner_uses_typed_output(samples):
    view = WorldView.model_validate(samples[1]["view"])
    lm = DummyLM(
        [{"subgoal": "restore_output", "rationale": "Clear a path to a chest"}]
    )
    with dspy.context(lm=lm):
        plan = DSPyPlanner()(view, [])
    assert plan.goal == Goal.OUTPUT
    assert plan.rationale == "Clear a path to a chest"


def favorable_transition(samples):
    before = WorldView.model_validate(samples[1]["view"])
    options = candidates(before, Goal.ESTABLISH)
    machine = before.entities[0].model_copy(update={"status": "working", "fuel": 19})
    after = before.model_copy(
        update={"version": before.version + 1, "entities": (machine,)}
    )
    assert verify(options[0].action, before, after) is True
    return Decision(
        before=before,
        goal=Goal.ESTABLISH,
        candidates=options,
        selected=options[0].id,
        result=Result(status="applied", after=after, fle_steps=1),
        selector="test-score-source",
        progress_probability=0.7,
        knowledge_version="test",
    )


def test_fueling_success_requires_observed_fuel(samples):
    view = WorldView.model_validate(samples[1]["view"])
    action = Action(kind=ActionKind.FUEL, entity=view.entities[0].ref, quantity=20)
    assert verify(action, view, view) is False


def test_rollout_rejects_a_model_supplied_unknown_id(samples):
    view = WorldView.model_validate(samples[0]["view"])
    with pytest.raises(ValueError, match="outside"):
        run_typed(None, view, 1, selector=lambda *_: Selection("invented"))


def episode_for(decision, split="train", fingerprint="start"):
    return {
        "id": fingerprint,
        "scenario": "iron_ore_throughput",
        "split": split,
        "start_group": fingerprint,
        "start_fingerprint": fingerprint,
        "horizon": 3,
        "steps": [{"decision": decision.model_dump(mode="json")}],
    }


def test_experience_learns_only_from_verified_training_actions(samples):
    decision = favorable_transition(samples)
    memory = learn([episode_for(decision), episode_for(decision, "test", "held-out")])
    assert len(memory.findings) == 1
    assert memory.findings[0]["support"] == ["start"]
    assert memory.retrieve(decision.before, decision.goal, decision.candidates)


def test_dataset_exports_only_observed_actions_and_drops_censored_rows(samples):
    decision = favorable_transition(samples)
    censored = decision.model_copy(
        update={
            "result": Result(status="unverified", after=decision.before, fle_steps=1)
        }
    )
    dataset = export_examples(
        [episode_for(decision), episode_for(censored, "test", "held-out")]
    )
    (item,) = dataset["items"]
    assert item["q"] == [1, 0] and item["probability"] == 0.7
    assert item["action"]["kind"] == "fuel_drill"
    with pytest.raises(ValueError, match="leakage"):
        export_examples([episode_for(decision), episode_for(decision, "test")])
    registry = {}
    register(registry, {"train": [item]})
    prompt = registry[TASK]["message"](item)
    assert "fuel_drill" in prompt and '"probability"' not in prompt
    assert registry[TASK]["labels"] == ["yes", "no"]


def test_calibration_fits_on_dev_and_reports_test_baselines():
    rows = [
        {"split": split, "probability": probability, "q": [label, 1 - label]}
        for split, probability, label in (
            ("dev", 0.1, 0),
            ("dev", 0.8, 1),
            ("test", 0.6, 1),
            ("test", 0.4, 0),
        )
    ]
    report = calibrate({"items": rows})
    assert report["test"]["count"] == 2
    assert report["test_scaled"]["brier"] < report["test_dev_base_rate"]["brier"]


def test_jev_scores_and_maps_candidates_with_the_sdk_contract(samples):
    pytest.importorskip("typesafe_sdk")
    decision = favorable_transition(samples)

    class Client:
        def system_one(self, **request):
            assert request["model"] == "jev-test"
            assert "3 scored FLE steps" in request["questions"]["action_1"].instructions
            return SimpleNamespace(
                nouls={
                    name: SimpleNamespace(noul=0.9 if name == "action_2" else 0.1)
                    for name in request["questions"]
                }
            )

    choice = JevSelector(Client(), "jev-test")(
        decision.before, decision.goal, decision.candidates, ()
    )
    assert choice.candidate_id == decision.candidates[1].id
