"""Register an exported Factorio dataset with the independent OpenJev-RLCD trainer."""

import argparse
import json
from pathlib import Path
import sys

TASK = "factorio_iron_progress"
SPLITS = ("train", "dev", "test")


def load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported iron progress dataset schema")
    splits = {split: [] for split in SPLITS}
    for row in dataset["items"]:
        splits[row["split"]].append(row)
    return splits


def message(item: dict) -> str:
    # Hide recorded results, status and labels from the model's prompt.
    context = {
        key: item[key]
        for key in ("world", "subgoal", "action", "planner", "selector")
        if key in item
    }
    return (
        f"Factorio scenario and proposed action:\n{json.dumps(context, sort_keys=True)}\n"
        f"Completion condition: {item['completion_condition']}\n"
        f"Will the condition be observed within {item['horizon']} scored FLE steps "
        "including this action, with the same policy continuing? "
        "Think briefly, then finish with exactly one line: Answer: yes or Answer: no."
    )


def register(tasks: dict, splits: dict) -> None:
    tasks[TASK] = dict(
        labels=["yes", "no"], message=message, load=lambda: splits, k=lambda _: 2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    splits = load_dataset(args.dataset)
    if any(not splits[name] for name in SPLITS):
        parser.error(
            "OpenJev-RLCD requires nonempty, disjoint train, dev and test splits"
        )
    from openjev_rlcd.tasks import TASKS
    from openjev_rlcd.train import main as train

    register(TASKS, splits)
    flags = args.trainer_args
    sys.argv = [sys.argv[0], *flags, "--task", TASK]
    train()


if __name__ == "__main__":
    main()
