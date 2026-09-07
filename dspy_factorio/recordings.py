"""Local episode checkpoints and standalone, self-contained HTML replays."""

import base64
from html import escape
import json
from pathlib import Path
from typing import Any

from dspy_factorio.meetup import comparison_row


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _image(path: str, caption: str) -> str:
    image = Path(path)
    if not path or not image.is_file():
        return "<p>Image unavailable in this recording.</p>"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def replay_html(episode: dict[str, Any]) -> str:
    """Embed all screenshots and styles; opening the file needs no services."""
    row = comparison_row(episode, 1)
    metadata = "".join(
        f"<dt>{escape(key)}</dt><dd>{escape(str(value)) if value is not None else 'Unavailable'}</dd>"
        for key, value in row.items() if key != "Episode"
    )
    cards = []
    for step in episode["steps"]:
        blocks = []
        for key, title in (
            ("reasoning", "Model rationale"), ("expected_result", "Expected result"),
            ("program", "Program"), ("observation", "Game response"),
        ):
            if step.get(key):
                blocks.append(f"<h3>{title}</h3><pre>{escape(str(step[key]))}</pre>")
        cards.append(
            f"<section><h2>{escape(step['label'])}</h2><p>Reward: {escape(str(step.get('reward')))}"
            f" · Environment done: {escape(str(step.get('done')))}</p><div class='step'><div>"
            + "".join(blocks) + "</div>" + _image(step.get("image", ""), step["label"]) + "</div></section>"
        )
    errors = "".join(f"<pre>{escape(str(episode[key]))}</pre>" for key in ("error", "cleanup_error") if episode.get(key))
    reset = "<details><summary>Reset map</summary>" + _image(episode.get("reset_image", ""), "Before actions") + "</details>"
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Factorio episode replay</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17202a;background:#fafafa}
h1{font-size:32px}h2{font-size:23px}h3{font-size:16px}section{border-top:1px solid #ccc;margin-top:32px;padding-top:12px}
.step{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:14px;border-radius:8px;font-size:13px}
img{width:100%;height:auto}figure{margin:0}figcaption{color:#566573;font-size:13px}dl{display:grid;grid-template-columns:150px 1fr;gap:5px}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}
@media(max-width:750px){.step{grid-template-columns:1fr}}
</style><main><h1>Factorio · saved episode</h1><p>Offline replay · no live game or API calls</p>""" + (
        f"<p>{escape(episode['goal'])}</p><dl>{metadata}</dl>{errors}{reset}"
        + "".join(cards)
        + "<p>Max score is peak game reward. USD is an estimate where available. Missing usage is not zero.</p>"
        + "<details><summary>Usage and harness details</summary><pre>"
        + escape(json.dumps({key: episode.get(key) for key in ("usage", "signature", "signature_instructions")}, indent=2))
        + "</pre></details></main></html>"
    )


def save_episode(episode: dict[str, Any]) -> Path:
    directory = Path(episode["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(directory / "episode.json", json.dumps(episode, indent=2))
    replay = directory / "replay.html"
    _atomic_text(replay, replay_html(episode))
    return replay


def load_saved_episodes(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read only local recordings, tolerating incomplete/corrupt files.

    Resolve screenshots against their episode directory so a copied recording
    works on another machine rather than referencing the old absolute paths.
    """
    episodes, issues = [], []
    for path in sorted(root.glob("*/episode.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            issues.append(f"{path.parent.name}: {type(error).__name__}")
            continue
        required = ("id", "model", "module", "goal", "status")
        valid = isinstance(episode, dict) and all(isinstance(episode.get(key), str) for key in required)
        if not valid or not isinstance(episode.get("steps"), list) or not isinstance(episode.get("elapsed"), (int, float)):
            issues.append(f"{path.parent.name}: invalid episode metadata")
            continue
        valid_steps = all(
            isinstance(step, dict) and isinstance(step.get("index"), int)
            and isinstance(step.get("label"), str) and isinstance(step.get("observation"), str)
            for step in episode["steps"]
        )
        if not valid_steps:
            issues.append(f"{path.parent.name}: invalid step metadata")
            continue
        episode["directory"] = str(path.parent)
        if episode.get("reset_image"):
            reset_image = path.parent / Path(episode["reset_image"]).name
            episode["reset_image"] = str(reset_image) if reset_image.is_file() else ""
        for step in episode["steps"]:
            if step.get("image"):
                image = path.parent / Path(step["image"]).name
                step["image"] = str(image) if image.is_file() else ""
        if episode["status"] == "running":
            episode["status"] = "saved checkpoint"
        episodes.append(episode)
    return episodes, issues
