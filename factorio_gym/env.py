"""Shared helpers for creating and stepping FLE gym environments."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from typing import Any

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
    "obs_text",
    "reset_env",
    "step_code",
]
