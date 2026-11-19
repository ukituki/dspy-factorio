"""Episode comparison and proper scoring for observed action outcomes."""

from collections import Counter
import json
from math import exp, log
from pathlib import Path

from dspy_factorio.iron_domain import Decision, progress_label
from dspy_factorio.iron_policy import PROGRESS, world_context

SPLITS = ("train", "dev", "test")


def load_runs(paths: list[Path]) -> list[dict]:
    files = [
        file
        for path in paths
        for file in (sorted(path.rglob("episode.json")) if path.is_dir() else [path])
    ]
    episodes = {}
    for file in files:
        episode = json.loads(file.read_text())
        if episode["id"] in episodes and episodes[episode["id"]] != episode:
            raise ValueError(f"Conflicting recordings for {episode['id']}")
        episodes[episode["id"]] = episode
    return list(episodes.values())


def summarize(episode: dict) -> dict:
    throughputs = [
        step["throughput"]
        for step in episode["steps"]
        if step.get("throughput") is not None
    ]
    return {
        "harness": episode["module"],
        "planner": episode.get("planner"),
        "start": episode["start"],
        "start_fingerprint": episode["start_fingerprint"],
        "knowledge_version": episode.get("knowledge_version"),
        "budget": episode["budget"],
        "fle_steps": sum(s["fle_steps"] for s in episode["steps"]),
        "preparation_steps": episode.get("preparation_steps", 0),
        "throughput": throughputs[-1] if throughputs else None,
        "peak_throughput": max(throughputs, default=None),
        "task_success": episode.get("success"),
        "outcomes": dict(
            Counter(step.get("outcome", "observed") for step in episode["steps"])
        ),
        "stop": episode["status"],
        "replay": str(Path(episode["directory"]) / "replay.html"),
    }


def export_examples(episodes: list[dict], horizon: int = 3) -> dict:
    """Label the executed action only, under its recorded continuation policy."""
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    groups: dict[str, str] = {}
    items = []
    for episode in episodes:
        if not any("decision" in step for step in episode["steps"]):
            continue
        split, group = episode["split"], episode["start_group"]
        if split not in SPLITS or not group:
            raise ValueError(f"Invalid split or start group in {episode['id']}")
        # Explicit map grouping protects transfer splits. The observed fingerprint
        # additionally catches duplicate starts recorded under a different label.
        keys = [f"group:{group}", f"start:{episode['start_fingerprint']}"]
        if episode.get("map_seed") is not None:
            keys.append(f"map:{episode['map_seed']}")
        for key in keys:
            if groups.setdefault(key, split) != split:
                raise ValueError(
                    f"Evaluation leakage: {key} occurs in multiple data splits"
                )
        decisions = [
            Decision.model_validate(step["decision"])
            for step in episode["steps"]
            if "decision" in step
        ]
        for index, decision in enumerate(decisions):
            label = progress_label(decisions, index, horizon)
            if label is None:
                continue
            probability = (
                decision.progress_probability
                if episode.get("horizon") == horizon
                else None
            )
            action = next(c for c in decision.candidates if c.id == decision.selected)
            distribution = [float(label), float(not label)]
            items.append(
                {
                    "uid": f"{episode['id']}:{index}",
                    "split": split,
                    "group": group,
                    "q": distribution,
                    "counts": distribution,
                    "probability": probability,
                    "horizon": horizon,
                    "completion_condition": PROGRESS[decision.goal],
                    "world": world_context(decision.before),
                    "subgoal": str(decision.goal),
                    "candidates": [
                        c.model_dump(mode="json") for c in decision.candidates
                    ],
                    "selected": decision.selected,
                    "action": action.action.model_dump(mode="json"),
                    "selector": decision.selector,
                    "harness": episode.get("module"),
                    "planner": episode.get("planner"),
                    "knowledge_version": decision.knowledge_version,
                    "evidence": list(decision.knowledge),
                    "candidate_probabilities": decision.candidate_probabilities,
                }
            )
    return {"schema_version": 1, "horizon": horizon, "items": items}


def binary_metrics(examples: list[tuple[float, float]]) -> dict:
    if not examples:
        return {"count": 0}
    n = len(examples)
    groups = [[] for _ in range(10)]
    for probability, outcome in examples:
        groups[min(int(probability * 10), 9)].append((probability, outcome))
    return {
        "count": n,
        "positives": int(sum(y for _, y in examples)),
        "brier": sum((p - y) ** 2 for p, y in examples) / n,
        "log_loss": -sum(log(max(1e-12, p if y else 1 - p)) for p, y in examples) / n,
        "ece_10": sum(abs(sum(p - y for p, y in bucket)) for bucket in groups) / n,
    }


def temperature_scale(p: float, temperature: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return 1 / (1 + exp(-log(p / (1 - p)) / temperature))


def calibrate(dataset: dict) -> dict:
    labeled = {"dev": [], "test": []}
    for row in dataset["items"]:
        if row["split"] in labeled and row["probability"] is not None:
            labeled[row["split"]].append((row["probability"], row["q"][0]))
    dev, test = labeled["dev"], labeled["test"]
    report = {"dev": binary_metrics(dev), "test": binary_metrics(test)}
    if not dev or not test or len({y for _, y in dev}) < 2:
        return {
            **report,
            "temperature": None,
            "status": "need both outcomes on dev and scored test examples",
        }
    temperature = min(
        (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0),
        key=lambda t: binary_metrics([(temperature_scale(p, t), y) for p, y in dev])[
            "log_loss"
        ],
    )
    base_rate = sum(y for _, y in dev) / len(dev)
    return {
        **report,
        "temperature": temperature,
        "test_scaled": binary_metrics(
            [(temperature_scale(p, temperature), y) for p, y in test]
        ),
        "test_dev_base_rate": binary_metrics([(base_rate, y) for _, y in test]),
    }
