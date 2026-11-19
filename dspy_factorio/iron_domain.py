"""State and action rules for the iron-ore throughput scenario."""

from enum import StrEnum
from hashlib import sha256
from math import floor, hypot
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Point(Record):
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return hypot(self.x - other.x, self.y - other.y)


DRILL = "burner-mining-drill"
CHEST = "wooden-chest"
REACH = 9.0  # Conservative for the standard lab scenario; FLE checks actual reach.
DIRECTIONS = ("north", "east", "south", "west")
BLOCKED = {
    "full_output",
    "not_enough_space_in_output",
    "waiting_for_space_in_destination",
}
RECOVERABLE = {"rejected", "failed", "unverified"}


class Machine(Record):
    ref: str
    name: str
    position: Point
    width: float | None = None
    height: float | None = None
    status: str | None = None
    drop: Point | None = None
    fuel: int | None = None
    contents: int | None = None
    resources: tuple[str, ...] | None = None

    @property
    def fueled(self) -> bool | None:
        if self.status == "no_fuel":
            return False
        if self.status in {"working", *BLOCKED} or (self.fuel or 0) > 0:
            return True
        return None


class WorldView(Record):
    episode_id: str
    version: int
    position: Point | None = None
    inventory: dict[str, int] | None = None
    entities: tuple[Machine, ...] | None = None
    iron: Point | None = None
    throughput: float | None = None
    task_success: bool | None = None

    def count(self, item: str) -> int | None:
        return self.inventory.get(item, 0) if self.inventory is not None else None

    def machines(self, name: str) -> tuple[Machine, ...]:
        return tuple(entity for entity in self.entities or () if entity.name == name)

    def find(self, ref: str | None) -> Machine | None:
        return next(
            (entity for entity in self.entities or () if entity.ref == ref), None
        )


class Goal(StrEnum):
    ESTABLISH = "establish_extraction"
    OUTPUT = "restore_output"
    SCALE = "increase_throughput"


class ActionKind(StrEnum):
    SCOUT = "scout_iron"
    MOVE = "move"
    DRILL = "place_drill"
    FUEL = "fuel_drill"
    CHEST = "place_output_chest"
    DRAIN = "empty_output_chest"
    CHECK = "measure_throughput"


class Action(Record):
    kind: ActionKind
    target: Point | None = None
    entity: str | None = None
    direction: Literal["north", "east", "south", "west"] = "north"
    quantity: int = 0

    @model_validator(mode="after")
    def valid_arguments(self):
        positional = self.kind in {ActionKind.MOVE, ActionKind.DRILL, ActionKind.CHEST}
        inventory = self.kind in {ActionKind.FUEL, ActionKind.DRAIN}
        if (self.target is not None) != positional or (
            self.entity is not None
        ) != inventory:
            raise ValueError(f"Unexpected target for {self.kind}")
        if (inventory and self.quantity <= 0) or (not inventory and self.quantity != 0):
            raise ValueError(f"Unexpected quantity for {self.kind}")
        return self

    def key(self) -> str:
        return self.model_dump_json()

    def candidate_id(self, view: WorldView) -> str:
        digest = sha256(self.key().encode()).hexdigest()[:16]
        return f"{view.episode_id}:{view.version}:{digest}"


class Candidate(Record):
    id: str
    action: Action
    expected: str


class Result(Record):
    status: Literal["applied", "rejected", "failed", "unverified", "unknown"]
    after: WorldView | None
    reason: str | None = None
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    observation: str = ""
    program: str = ""
    fle_steps: int = 0


class Decision(Record):
    before: WorldView
    goal: Goal
    candidates: tuple[Candidate, ...]
    selected: str
    result: Result
    selector: str
    knowledge_version: str
    knowledge: tuple[dict, ...] = ()
    progress_probability: float | None = None
    candidate_probabilities: dict[str, float] = {}


def iron_drills(view: WorldView) -> tuple[Machine, ...]:
    return tuple(
        drill
        for drill in view.machines(DRILL)
        if drill.resources and "iron-ore" in drill.resources
    )


def output_chest(view: WorldView, drill: Machine) -> Machine | None:
    return next(
        (chest for chest in view.machines(CHEST) if chest.position == drill.drop), None
    )


def overlaps(view: WorldView, point: Point, width: float) -> bool:
    return any(
        abs(entity.position.x - point.x) < ((entity.width or 1) + width) / 2 - 0.01
        and abs(entity.position.y - point.y) < ((entity.height or 1) + width) / 2 - 0.01
        for entity in view.entities or ()
    )


def plan_fixed(view: WorldView, milestone: bool = False) -> Goal:
    drills = iron_drills(view)
    if not any(drill.fueled for drill in drills):
        return Goal.ESTABLISH
    if not milestone and any(
        not output_chest(view, drill) or drill.status in BLOCKED for drill in drills
    ):
        return Goal.OUTPUT
    return Goal.SCALE


def reached_goal(goal: Goal, before: WorldView, after: WorldView) -> bool | None:
    if after.task_success:
        return True
    if after.entities is None or before.entities is None:
        return None
    if goal == Goal.ESTABLISH:
        return sum(drill.fueled is True for drill in iron_drills(after)) > sum(
            drill.fueled is True for drill in iron_drills(before)
        )
    if goal == Goal.OUTPUT:

        def open_outputs(view: WorldView) -> int:
            return sum(
                output_chest(view, drill) is not None and drill.status not in BLOCKED
                for drill in iron_drills(view)
            )

        return open_outputs(after) > open_outputs(before)
    if before.throughput is None or after.throughput is None:
        return None
    return after.throughput > before.throughput


def candidates(
    view: WorldView,
    goal: Goal,
    excluded: frozenset[str] = frozenset(),
    *,
    milestone: bool = False,
) -> tuple[Candidate, ...]:
    actions: list[tuple[Action, str]] = []
    seen = set(excluded)

    def offer(action: Action, effect: str) -> None:
        key = action.key()
        if key not in seen:
            actions.append((action, effect))
            seen.add(key)

    def local(action: Action, location: Point, effect: str) -> None:
        if view.position is None:
            return
        if view.position.distance(location) > REACH:
            offer(
                Action(kind=ActionKind.MOVE, target=location),
                "Move within reach of target",
            )
        else:
            offer(action, effect)

    if view.entities is None or view.inventory is None or view.position is None:
        return ()
    drills = iron_drills(view)

    def fuel() -> None:
        for drill in drills:
            if drill.fueled is not True and view.count("coal"):
                quantity = min(view.count("coal"), 20)
                local(
                    Action(kind=ActionKind.FUEL, entity=drill.ref, quantity=quantity),
                    drill.position,
                    "Drill has fuel or is working",
                )

    def output() -> None:
        for drill in drills:
            if drill.drop is None:
                continue
            chest = output_chest(view, drill)
            if chest and drill.status in BLOCKED and (chest.contents or 0) > 0:
                local(
                    Action(
                        kind=ActionKind.DRAIN,
                        entity=chest.ref,
                        quantity=min(chest.contents, 50),
                    ),
                    chest.position,
                    "Remove ore from blocked output chest",
                )
            elif not chest and view.count(CHEST) and not overlaps(view, drill.drop, 1):
                local(
                    Action(kind=ActionKind.CHEST, target=drill.drop),
                    drill.drop,
                    "Wooden chest at drill output",
                )

    def build() -> None:
        if not view.count(DRILL):
            return
        if view.iron is None:
            offer(Action(kind=ActionKind.SCOUT), "Observe the nearest iron ore")
            return
        if view.position.distance(view.iron) > REACH:
            offer(
                Action(kind=ActionKind.MOVE, target=view.iron),
                "Approach known iron ore",
            )
            return
        # Nearby sites are proposals around a known ore tile. Exact placement and the
        # newly observed drill's resources determine whether each site really works.
        offsets = (
            (0, 0),
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
            (2, 2),
            (-2, 2),
            (2, -2),
            (-2, -2),
            (4, 0),
            (-4, 0),
            (0, 4),
            (0, -4),
        )
        for dx, dy in offsets:
            site = Point(
                x=floor(view.iron.x + 0.5) + dx, y=floor(view.iron.y + 0.5) + dy
            )
            if view.position.distance(site) > REACH or overlaps(view, site, 2):
                continue
            for direction, (ax, ay) in zip(
                DIRECTIONS,
                ((-0.5, -1.5), (1.5, -0.5), (-0.5, 1.5), (-1.5, -0.5)),
                strict=True,
            ):
                drop = Point(x=site.x + ax, y=site.y + ay)
                if not overlaps(view, drop, 1):
                    offer(
                        Action(kind=ActionKind.DRILL, target=site, direction=direction),
                        "Place a burner drill that covers iron",
                    )
            if len(actions) >= 6:
                break

    if goal in {Goal.OUTPUT, Goal.SCALE} and not milestone:
        output()
    fuel()
    if goal == Goal.ESTABLISH and not milestone:
        output()
    build()
    if not milestone and drills and len(actions) < 6:
        offer(
            Action(kind=ActionKind.CHECK), "Measure iron ore output in FLE verification"
        )
    return tuple(
        Candidate(id=action.candidate_id(view), action=action, expected=effect)
        for action, effect in actions[:8]
    )


def verify(action: Action, before: WorldView, after: WorldView) -> bool | None:
    target = action.target
    match action.kind:
        case ActionKind.SCOUT:
            return after.iron is not None
        case ActionKind.MOVE:
            if after.position is None or before.position is None:
                return None
            return after.position.distance(target) <= REACH and after.position.distance(
                target
            ) < before.position.distance(target)
        case ActionKind.DRILL:
            if after.entities is None:
                return None
            return any(
                drill.position == target and before.find(drill.ref) is None
                for drill in iron_drills(after)
            )
        case ActionKind.FUEL:
            drill = after.find(action.entity)
            return drill.fueled if drill else None
        case ActionKind.CHEST:
            if after.entities is None:
                return None
            return any(chest.position == target for chest in after.machines(CHEST))
        case ActionKind.DRAIN:
            initial, final = before.count("iron-ore"), after.count("iron-ore")
            return (
                final > initial if initial is not None and final is not None else None
            )
        case ActionKind.CHECK:
            return True if after.throughput is not None else None


def progress_label(decisions: list[Decision], index: int, horizon: int) -> bool | None:
    initial = decisions[index]
    elapsed = 0
    for decision in decisions[index:]:
        result = decision.result
        elapsed += result.fle_steps
        if result.status == "unknown" or result.after is None or elapsed > horizon:
            return None
        progress = reached_goal(initial.goal, initial.before, result.after)
        if progress is True:
            return True
        if result.terminated or elapsed == horizon:
            return progress  # Missing measurements remain unlabeled.
        if result.truncated:
            return None
    return None  # Budget ended before the outcome window closed.
