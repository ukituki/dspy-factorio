"""Shared helpers for creating and stepping FLE gym environments."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

# FLE still depends on OpenAI Gym. Silence the gym-notices deprecation banner
# before that import (Gymnasium is not a drop-in here: separate registries).
_notices_pkg = types.ModuleType("gym_notices")
_notices_mod = types.ModuleType("gym_notices.notices")
_notices_mod.notices = {}
sys.modules.setdefault("gym_notices", _notices_pkg)
sys.modules.setdefault("gym_notices.notices", _notices_mod)

import gym
from dotenv import load_dotenv
from fle.commons.models.game_state import GameState
from fle.env.gym_env.action import Action
from fle.env.gym_env.registry import (
    get_environment_info,
    list_available_environments,
)

# Importing the registry auto-registers all FLE tasks with gym.
import fle.env.gym_env.registry  # noqa: F401


def load_project_env() -> None:
    """Load `.env` from the project root if present."""
    load_dotenv(override=False)


def ensure_server_env(
    address: str = "127.0.0.1",
    port: int = 27000,
) -> None:
    """Prefer explicit RCON targeting over Docker discovery.

    FLE defaults to inspecting Docker for Factorio containers. On Apple Silicon
    that path is often slow or flaky — pin address/port instead.
    """
    os.environ.setdefault("FACTORIO_SERVER_ADDRESS", address)
    os.environ.setdefault("FACTORIO_SERVER_PORT", str(port))


def make_env(env_id: str = "iron_ore_throughput", *, run_idx: int = 0):
    """Create a registered FLE gym environment.

    Important: FLE's factory requires ``run_idx`` (container index). The
    upstream quickstart omits this, but ``gym.make(env_id)`` alone fails.
    """
    ensure_server_env()
    # FLE's observation_space is a nested Dict that trips the default checker.
    return gym.make(env_id, run_idx=run_idx, disable_env_checker=True)


def reset_env(env) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset and normalize the (obs, info) Gym API return."""
    result = env.reset(options={"game_state": None})
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, info
    return result, {}


def step_code(
    env,
    code: str,
    *,
    agent_idx: int = 0,
    game_state: GameState | None = None,
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    """Execute one Python program against the live Factorio instance."""
    inner = env.unwrapped
    if game_state is None:
        game_state = GameState.from_instance(inner.instance)
    action = Action(code=code, agent_idx=agent_idx, game_state=game_state)
    return env.step(action)


def obs_text(obs: Mapping[str, Any]) -> str:
    """Best-effort stdout/stderr text from an observation dict."""
    raw = obs.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw
    return str(obs.get("text") or "")


def namespace(env, *, agent_idx: int = 0):
    """Return the FLE namespace for an agent (tools like ``_render`` live here)."""
    return env.unwrapped.instance.namespaces[agent_idx]


def overview_center(env, *, agent_idx: int = 0):
    """Midpoint between the player and nearest iron ore (keeps both in frame).

    ``_render_simple`` caps radius around ~44 tiles even when zoomed out, so a
    player-centered view at spawn still misses iron near ``y≈70``. Centering on
    this midpoint with ``zoom≈0.25`` usually shows spawn + ore together.
    """
    from fle.env import Position
    from fle.env.game_types import Resource

    ns = namespace(env, agent_idx=agent_idx)
    player = ns.player_location
    iron = ns.nearest(Resource.IronOre)
    return Position(x=(player.x + iron.x) / 2, y=(player.y + iron.y) / 2)


def render_map(
    env,
    *,
    mode: Literal["simple", "sprites"] = "simple",
    agent_idx: int = 0,
    zoom: float | None = None,
    overview: bool = False,
    **kwargs: Any,
):
    """Render the current map around the player (or an overview center).

    - ``simple``: schematic grid (no sprite assets required)
    - ``sprites``: Factorio-like pixels (needs ``uv run fle sprites`` first)
    - ``zoom`` < 1 zooms out (more tiles). ``0.25`` is a good wide default.
    - ``overview=True``: center between player and nearest iron so both stay visible

    Returns an FLE ``RenderedImage`` (``.save(path)``, ``.show()``, ``.to_base64()``).
    """
    ns = namespace(env, agent_idx=agent_idx)
    if zoom is not None:
        kwargs.setdefault("zoom", zoom)
    if overview and "position" not in kwargs:
        kwargs["position"] = overview_center(env, agent_idx=agent_idx)
    if mode == "sprites":
        return ns._render(**kwargs)
    return ns._render_simple(**kwargs)


def save_render(
    env,
    path: str | Path,
    *,
    mode: Literal["simple", "sprites"] = "simple",
    agent_idx: int = 0,
    zoom: float | None = None,
    overview: bool = False,
    **kwargs: Any,
) -> Path:
    """Render and write a PNG; returns the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_map(
        env,
        mode=mode,
        agent_idx=agent_idx,
        zoom=zoom,
        overview=overview,
        **kwargs,
    ).save(str(out))
    return out


def describe_env(env_id: str) -> str:
    info = get_environment_info(env_id) or {}
    desc = info.get("description") or "(no description)"
    return f"{env_id}: {desc}"


def list_envs(*, throughput_only: bool = False) -> list[str]:
    ids = list_available_environments()
    if throughput_only:
        return [e for e in ids if "throughput" in e]
    return ids


__all__ = [
    "describe_env",
    "ensure_server_env",
    "get_environment_info",
    "list_envs",
    "load_project_env",
    "make_env",
    "namespace",
    "obs_text",
    "overview_center",
    "render_map",
    "reset_env",
    "save_render",
    "step_code",
]
