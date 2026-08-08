"""Helpers to join the local Factorio desktop client to the FLE cluster."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import fle


def fle_cluster_dir() -> Path:
    """Directory containing the generated ``docker-compose.yml`` FLE runs."""
    return Path(fle.__file__).resolve().parent / "cluster"


def cluster_config_dir() -> Path:
    return fle_cluster_dir() / "config"


def compose_path() -> Path:
    return fle_cluster_dir() / "docker-compose.yml"


def ensure_join_config_files() -> list[Path]:
    """Create missing whitelist/banlist JSON files FLE's compose references."""
    config = cluster_config_dir()
    written: list[Path] = []
    for name in ("server-whitelist.json", "server-banlist.json"):
        path = config / name
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")
            written.append(path)
    return written


def disable_server_whitelist(compose_file: Path | None = None) -> bool:
    """Remove whitelist CLI flags so any local client can join.

    Factorio 2.x errors if ``--server-whitelist`` is set without
    ``--use-server-whitelist``. FLE ships both but an empty whitelist blocks
    joins — strip both flags instead.

    Returns True if the compose file was modified.
    """
    path = compose_file or compose_path()
    text = path.read_text(encoding="utf-8")
    new = text
    new = re.sub(r"\s*--use-server-whitelist\b", "", new)
    new = re.sub(
        r"\s*--server-whitelist\s+/opt/factorio/config/server-whitelist\.json\b",
        "",
        new,
    )
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def recreate_cluster(num: int = 1) -> None:
    """Force-recreate containers from the patched compose (Apple Silicon friendly)."""
    import os

    cluster = fle_cluster_dir()
    env = {**os.environ, "DOCKER_PLATFORM": "linux/arm64"}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "--pull", "never"],
        cwd=cluster,
        env=env,
        check=True,
    )


def prepare_live_client(*, recreate: bool = True) -> dict[str, object]:
    """Make the running FLE Factorio server joinable from the desktop client.

    - Ensures ``server-banlist.json`` exists (compose still references it)
    - Removes ``--server-whitelist`` / ``--use-server-whitelist`` from compose
    - Optionally recreates the container so the command change takes effect
    """
    created = ensure_join_config_files()
    patched = disable_server_whitelist()
    did_recreate = False
    if recreate and (patched or created):
        recreate_cluster()
        did_recreate = True
    elif recreate and not patched:
        # Compose already patched earlier but container may be crash-looping
        # from a half-applied patch — recreate if RCON is down.
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", 27000))
        except OSError:
            recreate_cluster()
            did_recreate = True
        finally:
            sock.close()
    return {
        "config_dir": str(cluster_config_dir()),
        "compose": str(compose_path()),
        "created_files": [str(p) for p in created],
        "whitelist_disabled": patched,
        "recreated": did_recreate,
        "connect": "127.0.0.1:34197",
        "server_version": "2.0.73",
    }


def connect_instructions() -> str:
    return """\
Connect with the Factorio desktop client
----------------------------------------
1. Install Factorio matching the server (Docker image factoriotools/factorio:2.0.73).
   Disable Space Age / elevated-rails / quality DLC mods if prompted to sync.
2. Multiplayer → Connect to address → 127.0.0.1:34197
3. Leave the client open; run an agent script in another terminal.
   You watch — the agent still acts via Python/RCON, not your keyboard.

If join fails after `fle cluster start`, re-run:
    uv run python examples/10_live_client_watch.py --prepare-only
(`fle cluster start` regenerates compose and re-enables the whitelist flag.)
"""


__all__ = [
    "cluster_config_dir",
    "compose_path",
    "connect_instructions",
    "disable_server_whitelist",
    "ensure_join_config_files",
    "fle_cluster_dir",
    "prepare_live_client",
]
