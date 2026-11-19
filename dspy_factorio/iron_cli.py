"""Run, compare, and learn from the typed iron-ore agent."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from uuid import uuid4

from dspy_factorio.agent import AgentConfig, build_lm, strip_code_fences
from dspy_factorio.env import (
    get_environment_info,
    load_project_env,
    make_env,
    obs_text,
    save_render,
    step_code,
)
from dspy_factorio.iron_domain import ActionKind, Decision, Goal, WorldView, candidates
from dspy_factorio.iron_eval import (
    SPLITS,
    calibrate,
    export_examples,
    load_runs,
    summarize,
)
from dspy_factorio.iron_fle import IronFLE, project
from dspy_factorio.iron_policy import (
    DSPyPlanner,
    JevSelector,
    Knowledge,
    fixed_planner,
    learn,
    scripted,
    world_context,
)
from dspy_factorio.iron_rollout import run_typed
from dspy_factorio.meetup import build_harness, episode_success
from dspy_factorio.recordings import save_episode

SCENARIO = "iron_ore_throughput"
OUTPUT = Path(".fle/renders/meetup")
HARNESSES = ("scripted", "dspy", "jev-empty", "jev-memory", "raw")
STARTS = {
    "cold": (),
    "unfueled": (ActionKind.SCOUT, ActionKind.MOVE, ActionKind.DRILL),
    "blocked": (ActionKind.SCOUT, ActionKind.MOVE, ActionKind.DRILL, ActionKind.FUEL),
}
RAW_API = (
    "Use nearest(Resource.IronOre), move_to(position), place_entity(Prototype.BurnerMiningDrill, "
    "position=..., direction=Direction.NORTH), insert_item(Prototype.Coal, drill, quantity=20), "
    "and place_entity(Prototype.WoodenChest, position=drill.drop_position). Print what you observe."
)


def fingerprint(view: WorldView, map_seed: int | None) -> str:
    canonical = json.dumps(
        [map_seed, world_context(view)], sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode()).hexdigest()[:16]


def prepare_start(game: IronFLE, view: WorldView, mode: str, layout: int) -> WorldView:
    for kind in STARTS[mode]:
        options = [
            c
            for c in candidates(view, Goal.ESTABLISH, milestone=True)
            if c.action.kind == kind
        ]
        index = layout if kind == ActionKind.DRILL else 0
        if index >= len(options):
            raise RuntimeError(
                f"Cannot prepare {mode}: {kind} index {index} is unavailable"
            )
        result = game.execute(view, options[index])
        if (
            result.status != "applied"
            or result.terminated
            or result.truncated
            or result.after is None
        ):
            raise RuntimeError(
                f"Cannot prepare {mode}: {kind} {result.status}: {result.reason}"
            )
        view = result.after
    return view


def run_raw(env, start: WorldView, goal: str, budget: int, on_step) -> str:
    agent = build_harness("ChainOfThought", "State-aware")
    observation = json.dumps(world_context(start), sort_keys=True)
    for index in range(1, budget + 1):
        prediction = agent(goal=goal, observation=observation, inventory_hint=RAW_API)
        code = strip_code_fences(prediction.program)
        obs, reward, terminated, truncated, info = step_code(
            env, code, capture_state=False
        )
        observation = obs_text(obs)
        after = project(obs, start.episode_id, start.version + index)
        on_step(
            {
                "index": index,
                "label": f"Raw Python step {index}",
                "program": code,
                "expected_result": prediction.expected_result,
                "reasoning": prediction.reasoning,
                "observation": observation,
                "reward": reward,
                "throughput": after.throughput,
                "task_success": after.task_success,
                "done": terminated or truncated,
                "fle_steps": 1,
                "outcome": "failed" if info.get("error_occurred") else "observed",
            }
        )
        if after.task_success:
            return "throughput verified"
        if terminated or truncated:
            return "environment stopped"
    return "FLE step budget reached"


def read_knowledge(path: Path | None) -> Knowledge:
    if path is None:
        return Knowledge()
    data = json.loads(path.read_text())
    snapshot = Knowledge(tuple(data["findings"]))
    if data["version"] != snapshot.version:
        raise ValueError("Knowledge snapshot content does not match its version")
    return snapshot


def run_episode(
    env,
    *,
    harness: str = "scripted",
    start_mode: str = "cold",
    layout: int = 0,
    budget: int = 16,
    milestone: bool = False,
    output: Path = OUTPUT,
    split: str = "train",
    start_group: str | None = None,
    knowledge: Knowledge = Knowledge(),
    planner_model: str = "openai/gpt-4o-mini",
    jev_model: str = "jev-latest",
    horizon: int = 3,
    renders: bool = False,
    first_action: int | None = None,
    planner_override: str | None = None,
) -> dict:
    import dspy

    if harness == "raw" and milestone:
        raise ValueError("The raw benchmark uses the full throughput task")
    if harness == "jev-memory" and not knowledge.findings:
        raise ValueError("jev-memory requires a nonempty training knowledge snapshot")
    if budget < 1 or horizon < 1 or layout < 0:
        raise ValueError(
            "Budget and horizon must be positive, and layout must be nonnegative"
        )
    if start_mode == "cold" and layout:
        raise ValueError("A cold start has no prebuilt drill layout")
    if first_action is not None and (first_action < 1 or harness == "raw"):
        raise ValueError(
            "A forced first candidate is a positive 1-based index for typed runs"
        )
    episode_id = uuid4().hex[:12]
    directory = output / episode_id
    metadata = get_environment_info(SCENARIO) or {}
    goal = metadata.get("description") or SCENARIO
    game = IronFLE(env)
    start = prepare_start(game, game.reset(episode_id), start_mode, layout)
    env.unwrapped.instance.pause()
    map_seed = game.map_seed()
    group = start_group or f"map-seed-{map_seed}"
    active_knowledge = knowledge if harness == "jev-memory" else Knowledge()
    planner_name = {"raw": "raw", "dspy": "dspy"}.get(
        harness, planner_override or "fixed"
    )
    use_dspy = planner_name in {"dspy", "raw"}
    use_jev = harness.startswith("jev")
    selector_name = jev_model if use_jev else "scripted"
    model = (
        " + ".join([planner_model, selector_name])
        if use_dspy and use_jev
        else (planner_model if use_dspy else selector_name)
    )
    episode = {
        "id": episode_id,
        "directory": str(directory),
        "scenario": SCENARIO,
        "goal": goal,
        "model": model,
        "module": harness,
        "signature": "Raw Python" if harness == "raw" else "Typed iron",
        "start": start_mode,
        "start_group": group,
        "start_fingerprint": fingerprint(start, map_seed),
        "map_seed": map_seed,
        "start_view": start.model_dump(mode="json"),
        "layout": layout,
        "split": split,
        "preparation_steps": len(STARTS[start_mode]),
        "budget": budget,
        "horizon": horizon,
        "planner": planner_name,
        "knowledge_version": active_knowledge.version,
        "milestone": milestone,
        "first_action": first_action,
        "steps": [],
        "elapsed": 0.0,
        "status": "running",
    }
    begun = monotonic()
    save_episode(episode)

    def record(step: dict) -> None:
        episode["steps"].append(step)
        episode["elapsed"] = monotonic() - begun
        if renders:
            step["image"] = str(
                save_render(env, directory / f"step_{step['index']:02d}.png")
            )
        save_episode(episode)

    def typed_step(decision: Decision, rationale: str) -> None:
        chosen = next(c for c in decision.candidates if c.id == decision.selected)
        result, index = decision.result, len(episode["steps"]) + 1
        record(
            {
                "index": index,
                "label": f"{index}. {chosen.action.kind.value}",
                "program": result.program,
                "observation": result.observation,
                "expected_result": chosen.expected,
                "reasoning": rationale,
                "reward": result.reward,
                "throughput": result.after.throughput if result.after else None,
                "task_success": result.after.task_success if result.after else None,
                "done": result.terminated or result.truncated,
                "fle_steps": result.fle_steps,
                "outcome": result.status,
                "decision": decision.model_dump(mode="json"),
            }
        )

    try:
        if not os.getenv("TYPESAFE_API_KEY") and harness.startswith("jev"):
            raise RuntimeError("Set TYPESAFE_API_KEY before running the Jev harness")
        with ExitStack() as stack:
            if use_dspy:
                lm = build_lm(AgentConfig(model=planner_model)).copy(cache=False)
                stack.enter_context(dspy.context(lm=lm))
            if harness == "raw":
                episode["status"] = run_raw(env, start, goal, budget, record)
            else:
                selector = scripted
                if harness.startswith("jev"):
                    from typesafe_sdk import TypeSafeClient

                    client = stack.enter_context(TypeSafeClient())
                    selector = JevSelector(client, jev_model, horizon)
                planner = DSPyPlanner() if planner_name == "dspy" else fixed_planner
                episode["status"] = run_typed(
                    game,
                    start,
                    budget,
                    milestone=milestone,
                    planner=planner,
                    selector=selector,
                    selector_name=selector_name,
                    knowledge=active_knowledge,
                    on_step=typed_step,
                    first_action=first_action,
                )
    except Exception as error:
        episode["status"] = "error"
        episode["error"] = f"{type(error).__name__}: {error}"
    finally:
        episode["elapsed"] = monotonic() - begun
        episode["success"] = episode_success(episode["steps"])
        save_episode(episode)
    return episode


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2)
        temporary.write("\n")
    Path(temporary.name).replace(path)


def run_live(args) -> int:
    load_project_env()
    if any(name.startswith("jev") for name in args.harnesses) and not os.getenv(
        "TYPESAFE_API_KEY"
    ):
        raise SystemExit("Set TYPESAFE_API_KEY before running the Jev harness")
    if args.update_knowledge is not None and args.split != "train":
        raise SystemExit(
            "Automatic knowledge updates are only available for training exploration"
        )
    if args.update_knowledge is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    knowledge = (
        learn(load_runs([args.output]))
        if args.update_knowledge is not None
        else read_knowledge(args.knowledge)
    )
    rows = []
    failed = False
    env = make_env(SCENARIO)
    try:
        for start in args.starts:
            for harness in args.harnesses:
                episode = run_episode(
                    env,
                    harness=harness,
                    start_mode=start,
                    layout=args.layout,
                    budget=args.steps,
                    milestone=args.milestone,
                    output=args.output,
                    split=args.split,
                    start_group=args.group,
                    knowledge=knowledge,
                    planner_model=args.planner_model,
                    jev_model=args.jev_model,
                    horizon=args.horizon,
                    renders=args.renders,
                    first_action=args.first_action,
                    planner_override=args.planner,
                )
                row = summarize(episode)
                failed |= (
                    episode["status"] == "error"
                    or episode["status"] == "game outcome unknown"
                )
                if episode.get("error"):
                    row["error"] = episode["error"]
                rows.append(row)
                if args.update_knowledge is not None:
                    knowledge = learn(load_runs([args.output]))
                    write_json(
                        args.update_knowledge,
                        {
                            "findings": knowledge.findings,
                            "version": knowledge.version,
                        },
                    )
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    for start in args.starts:
        matched = {row["start_fingerprint"] for row in rows if row["start"] == start}
        if len(matched) > 1:
            raise SystemExit(
                f"Unmatched initial conditions for {start}; see per-run replays"
            )
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser(
        "run", help="Run one or more harnesses at matched starts"
    )
    live.add_argument("--harnesses", nargs="+", choices=HARNESSES, default=["scripted"])
    live.add_argument("--starts", nargs="+", choices=STARTS, default=["cold"])
    live.add_argument(
        "--layout",
        type=int,
        default=0,
        help="Zero-based prepared drill placement choice",
    )
    live.add_argument("--steps", type=int, default=16)
    live.add_argument("--horizon", type=int, default=3)
    live.add_argument("--split", choices=SPLITS, default="train")
    live.add_argument(
        "--group",
        help="Group related maps into one split; defaults to the live map seed",
    )
    live.add_argument("--output", type=Path, default=OUTPUT)
    memory = live.add_mutually_exclusive_group()
    memory.add_argument(
        "--knowledge", type=Path, help="Load a frozen evaluation snapshot"
    )
    memory.add_argument(
        "--update-knowledge",
        type=Path,
        help="Rebuild a training snapshot from the output corpus after every run",
    )
    live.add_argument(
        "--planner",
        choices=("fixed", "dspy"),
        help="Planner for Jev or scripted-selector runs",
    )
    live.add_argument("--planner-model", default="openai/gpt-4o-mini")
    live.add_argument("--jev-model", default="jev-latest")
    live.add_argument("--milestone", action="store_true")
    live.add_argument("--renders", action="store_true")
    live.add_argument(
        "--first-action",
        type=int,
        help="Force a 1-based eligible candidate on the first decision",
    )
    live.set_defaults(handler=run_live)
    for command in ("learn", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("runs", nargs="+", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        if command == "export":
            sub.add_argument("--horizon", type=int, default=3)
    scoring = commands.add_parser("calibrate")
    scoring.add_argument("dataset", type=Path)
    scoring.add_argument("--harness", choices=HARNESSES)
    args = parser.parse_args()
    if args.command == "run":
        return args.handler(args)
    if args.command == "calibrate":
        dataset = json.loads(args.dataset.read_text())
        if args.harness:
            dataset["items"] = [
                row for row in dataset["items"] if row.get("harness") == args.harness
            ]
        print(json.dumps(calibrate(dataset), indent=2))
        return 0
    episodes = load_runs(args.runs)
    if args.command == "learn":
        snapshot = learn(episodes)
        result = {"findings": snapshot.findings, "version": snapshot.version}
    else:
        result = export_examples(episodes, args.horizon)
    write_json(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
