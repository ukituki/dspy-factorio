"""Translate the iron agent's typed actions to FLE and project its observations."""

from collections.abc import Mapping
from enum import Enum
from numbers import Integral
import re

from dspy_factorio.env import obs_text, reset_env, step_code
from dspy_factorio.iron_domain import (
    CHEST,
    Action,
    ActionKind,
    Candidate,
    Machine,
    Point,
    Result,
    WorldView,
    verify,
)
from dspy_factorio.meetup import task_success

IRON_MARKER = re.compile(r"IRON_PROBE\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def field(source, name: str, default=None):
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def point(source) -> Point | None:
    x, y = field(source, "x"), field(source, "y")
    if x is None or y is None:
        return None
    return Point(x=float(x), y=float(y))


def counts(source) -> dict[str, int] | None:
    if source is None:
        return None
    if isinstance(source, list):
        return {item["type"]: int(item["quantity"]) for item in source}
    if hasattr(source, "items"):
        return {
            item: int(count)
            for item, count in source.items()
            if isinstance(count, Integral)
        }
    return None


def project(
    obs: Mapping, episode_id: str, version: int, iron: Point | None = None
) -> WorldView:
    position = next(
        (
            point(entry)
            for entry in obs.get("character_positions", ())
            if entry.get("agent_idx") == 0
        ),
        None,
    )
    entities = None if obs.get("entities") is None else []
    for item in obs.get("entities") or ():
        location = point(field(item, "position"))
        name = field(item, "name")
        if not name or location is None:
            continue
        status = field(item, "status")
        if isinstance(status, Enum):
            status = status.value
        fuel = counts(field(item, "fuel"))
        inventory = counts(field(item, "inventory"))
        size = field(item, "tile_dimensions")
        ore = field(item, "resources")
        entities.append(
            Machine(
                ref=f"{name}@{location.x:g},{location.y:g}",
                name=name,
                position=location,
                width=field(size, "tile_width"),
                height=field(size, "tile_height"),
                status=status,
                drop=point(field(item, "drop_position")),
                fuel=sum(fuel.values()) if fuel is not None else None,
                contents=(
                    inventory.get("iron-ore", 0) if inventory is not None else None
                ),
                resources=(
                    tuple(field(entry, "name") for entry in ore)
                    if ore is not None
                    else None
                ),
            )
        )
    meta = (obs.get("task_verification") or {}).get("meta") or []
    meta = (
        meta
        if isinstance(meta, Mapping)
        else {row["key"]: row["value"] for row in meta}
    )
    rate = meta.get("reward_override")
    return WorldView(
        episode_id=episode_id,
        version=version,
        position=position,
        inventory=counts(obs.get("inventory")),
        entities=(
            tuple(sorted(entities, key=lambda entity: entity.ref))
            if entities is not None
            else None
        ),
        iron=iron,
        throughput=float(rate) if rate is not None else None,
        task_success=task_success(obs) if meta else None,
    )


def position_arg(location: Point) -> str:
    return f"Position(x={location.x!r}, y={location.y!r})"


def render_action(action: Action, view: WorldView) -> str:
    """Only compose fixed templates with validated numbers and enum names."""
    match action.kind:
        case ActionKind.SCOUT:
            return "iron = nearest(Resource.IronOre)\nprint(f'IRON_PROBE {iron.x} {iron.y}')"
        case ActionKind.MOVE:
            return f"print(move_to({position_arg(action.target)}))"
        case ActionKind.DRILL | ActionKind.CHEST:
            name = (
                "BurnerMiningDrill"
                if action.kind == ActionKind.DRILL
                else "WoodenChest"
            )
            return (
                f"print(place_entity(Prototype.{name}, position={position_arg(action.target)}, "
                f"direction=Direction.{action.direction.upper()}, exact=True))"
            )
        case ActionKind.FUEL | ActionKind.DRAIN:
            machine = view.find(action.entity)
            prototype = "WoodenChest" if machine.name == CHEST else "BurnerMiningDrill"
            operation, material = (
                ("insert_item", "Coal")
                if action.kind == ActionKind.FUEL
                else ("extract_item", "IronOre")
            )
            return (
                f"target = get_entity(Prototype.{prototype}, {position_arg(machine.position)})\n"
                f"print({operation}(Prototype.{material}, target, quantity={action.quantity}))"
            )
        case ActionKind.CHECK:
            return "print('Measure throughput')"


class IronFLE:
    def __init__(self, env):
        self.env = env
        self.current = None

    def reset(self, episode_id: str) -> WorldView:
        # FLE 0.4.3 Gym reset preserves the character's position. Reset it
        # explicitly so a previous arm cannot change where the next one scouts.
        self.env.unwrapped.instance.reset(reset_position=True)
        obs, _ = reset_env(self.env)
        self.current = project(obs, episode_id, 0)
        return self.current

    def map_seed(self) -> int:
        """Read the scenario seed for split grouping without advancing the clock."""
        client = self.env.unwrapped.instance.rcon_client
        value = client.send_command(
            "/sc rcon.print(game.surfaces[1].map_gen_settings.seed)"
        )
        return int(value.strip())

    def refresh(self, view: WorldView) -> WorldView:
        obs = self.env.unwrapped.get_observation(0).to_dict()
        return project(obs, view.episode_id, view.version + 1, view.iron)

    def execute(self, view: WorldView, candidate: Candidate) -> Result:
        if self.current != view or candidate.id != candidate.action.candidate_id(view):
            raise ValueError("Candidate is bound to a stale game observation")
        code = render_action(candidate.action, view)
        try:
            obs, reward, terminated, truncated, info = step_code(
                self.env, code, capture_state=False
            )
            text = obs_text(obs)
            iron = view.iron
            if candidate.action.kind == ActionKind.SCOUT:
                match = IRON_MARKER.search(text)
                if match:
                    iron = Point(x=float(match[1]), y=float(match[2]))
            after = project(obs, view.episode_id, view.version + 1, iron)
        except Exception as error:
            # A scored step may have changed the game before observation failed.
            # The run stops here, preserving uncertainty and its step accounting.
            try:
                after = self.refresh(view)
            except Exception:
                after = None
            self.current = after
            return Result(
                status="unknown",
                after=after,
                reason=f"{type(error).__name__}: {error}",
                program=code,
                fle_steps=1,
            )
        self.current = after
        effect = verify(candidate.action, view, after)
        if info.get("error_occurred"):
            status, reason = "failed", str(info.get("result") or text)
        elif effect is not True:
            status, reason = (
                "unverified",
                "Expected effect missing from game observation",
            )
        else:
            status, reason = "applied", None
        return Result(
            status=status,
            after=after,
            reason=reason,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            observation=text,
            program=code,
            fle_steps=1,
        )
