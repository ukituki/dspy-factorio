"""Interactive explore notebook for iron_ore_throughput.

Companion to docs/TASKS_GUIDE.md. Open with the *project* env (do not accept
marimo's isolated sandbox — this package is local, not on PyPI):

    uv run fle cluster start -n 1   # for live cells
    uv run marimo edit scenarios/1_iron_ore_throughput/01_explore_env.py

Requires a running Factorio cluster for the live cells.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # Repo root so `dspy_factorio` imports work when editing from scenarios/
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explore `iron_ore_throughput`

    Guided walkthrough of how FLE **tasks** work — based on
    [`docs/TASKS_GUIDE.md`](../../docs/TASKS_GUIDE.md).

    You do **not** need an LLM. Each action is a short **Python program** run
    inside Factorio; the reply is mostly **text** (`raw_text`) plus a reward.

    ```text
    goal + inventory + map
            │
            ▼
       reset() ──► write Python ──► step() ──► read stdout / reward
                        ▲                            │
                        └──────── next program ◄─────┘
    ```

    **Tip:** In the notebook footer, set *On Cell Change* → **lazy** so live
    Factorio cells only run when you ask.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Why this task?

    | Property | Value |
    |----------|--------|
    | Env / task id | `iron_ore_throughput` |
    | Goal | Automate **16 iron ore / 60 in-game seconds** |
    | Why easiest | Mining only — no smelting, assemblers, or oil |
    | Agents | 1 |
    | Typical horizon | 64 programs (lab-play default) |

    Harder tasks reuse the **same loop**; they mainly deepen the recipe tree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Task metadata (no Factorio needed)

    `get_environment_info` / `describe_env` read the gym registry. Safe to run
    even when the cluster is down.
    """)
    return


@app.cell
def _(mo):
    from dspy_factorio.env import (
        describe_env,
        get_environment_info,
        list_envs,
        load_project_env,
    )

    load_project_env()
    ENV_ID = "iron_ore_throughput"

    env_info = get_environment_info(ENV_ID) or {}
    ironish = [e for e in list_envs() if "iron" in e.lower()]

    mo.vstack(
        [
            mo.md(f"**describe_env:** `{describe_env(ENV_ID)}`"),
            mo.md("**Registry info keys / values:**"),
            mo.ui.table(
                [
                    {"key": k, "value": str(v)[:200]}
                    for k, v in sorted(env_info.items())
                ]
            ),
            mo.md(
                f"**Other `*iron*` envs ({len(ironish)}):** "
                + ", ".join(f"`{e}`" for e in ironish[:12])
                + (" …" if len(ironish) > 12 else "")
            ),
        ]
    )
    return ENV_ID, load_project_env


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What’s inside a task

    When you later `make_env("iron_ore_throughput")`, FLE builds:

    1. A live Factorio instance (Docker / RCON)
    2. A **task object** (goal, inventory, scoring rule)
    3. A gym wrapper that turns programs into `step()` calls

    Lab-play starts with a **populated inventory** and **all tech researched**.
    Early friction is “place and fuel correctly,” not grinding for a pickaxe.

    **Scoring (easy to miss):** after each program the task roughly sleeps
    ~60 in-game seconds, measures automatic iron-ore throughput, uses that as
    **reward**, and sets **done** when throughput ≥ 16.

    So a program that only prints `nearest(Resource.IronOre)` often scores
    `reward=0.0` — correct; nothing is mining yet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Connect to Factorio

    Prerequisites:

    ```bash
    uv run fle cluster start -n 1
    # wait until RCON is up — see docs/SETUP.md
    ```

    Click **Connect & reset** once. Re-clicking closes the previous session and
    opens a fresh one.
    """)
    return


@app.cell
def _(mo):
    get_session, set_session = mo.state(None)
    connect_btn = mo.ui.run_button(label="Connect & reset", kind="success")
    close_btn = mo.ui.run_button(label="Close env", kind="danger")
    mo.hstack([connect_btn, close_btn], justify="start", gap=1)
    return close_btn, connect_btn, get_session, set_session


@app.cell
def _(
    ENV_ID,
    close_btn,
    connect_btn,
    get_session,
    load_project_env,
    mo,
    set_session,
):
    from dspy_factorio.env import make_env, reset_env

    status = mo.md("_Not connected._")

    if close_btn.value:
        _sess_close = get_session()
        if _sess_close and _sess_close.get("env") is not None:
            try:
                _sess_close["env"].close()
            except Exception as exc:  # noqa: BLE001 — show in notebook
                status = mo.md(f"**Close error:** `{exc}`").callout(kind="warn")
            else:
                status = mo.md("Closed.").callout(kind="neutral")
        set_session(None)

    if connect_btn.value:
        _sess_old = get_session()
        if _sess_old and _sess_old.get("env") is not None:
            try:
                _sess_old["env"].close()
            except Exception:
                pass
        load_project_env()
        try:
            _env = make_env(ENV_ID, run_idx=0)
            reset_env(_env)
        except Exception as exc:  # noqa: BLE001
            set_session(None)
            status = mo.md(
                f"**Connect failed:** `{exc}`\n\n"
                "Is the cluster up? `uv run fle cluster start -n 1`"
            ).callout(kind="danger")
        else:
            set_session({"env": _env, "history": []})
            status = mo.md(
                f"Connected to `{ENV_ID}` and reset. Ready for a program."
            ).callout(kind="success")

    status
    return status


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Action = one Python program

    Each `step` sends an `Action` with:

    | Field | Meaning |
    |-------|---------|
    | `code` | Executable FLE Python (the whole “move”) |
    | `agent_idx` | `0` for this task |
    | `game_state` | Helpers usually pass the current state |

    **Good:**

    ```python
    nearest(Resource.IronOre)
    move_to(iron)
    place_entity(entity=Prototype.BurnerMiningDrill, position=iron, direction=Direction.NORTH)
    insert_item(Prototype.Coal, drill, quantity=5)
    ```

    **Bad** (crashes inside FLE):

    ```python
    nearest("iron-ore")
    move_to(pos=iron)
    place_entity("mining-drill", ...)
    ```

    Edit the program below, then **Run program**. Start with the explore
    snippet from the guide (find iron, print it).
    """)
    return


@app.cell
def _(mo):
    program = mo.ui.code_editor(
        value=(
            "iron = nearest(Resource.IronOre)\n"
            "print(iron)\n"
            "print(inspect_inventory())\n"
        ),
        language="python",
        label="FLE program",
    )
    run_btn = mo.ui.run_button(label="Run program", kind="success")
    mo.vstack([program, run_btn])
    return program, run_btn


@app.cell
def _(get_session, mo, program, run_btn, set_session):
    from dspy_factorio.env import obs_text, step_code

    result_view = mo.md(
        "Connect first, then click **Run program**."
    ).callout(kind="info")

    if run_btn.value:
        _sess_run = get_session()
        if _sess_run is None or _sess_run.get("env") is None:
            result_view = mo.md(
                "Not connected — use **Connect & reset** above."
            ).callout(kind="warn")
        else:
            _code = program.value.strip()
            try:
                _obs, _reward, _terminated, _truncated, _info = step_code(
                    _sess_run["env"], _code
                )
            except Exception as exc:  # noqa: BLE001
                result_view = mo.md(f"**step failed:** `{exc}`").callout(
                    kind="danger"
                )
            else:
                _text = obs_text(_obs) or "(empty raw_text)"
                _keys = sorted(_obs.keys()) if isinstance(_obs, dict) else []
                _entry = {
                    "code": _code,
                    "reward": _reward,
                    "terminated": _terminated,
                    "truncated": _truncated,
                    "error_occurred": bool(_info.get("error_occurred")),
                    "raw_text": _text,
                    "obs_keys": _keys,
                }
                _history = list(_sess_run.get("history") or [])
                _history.append(_entry)
                set_session({**_sess_run, "history": _history, "last": _entry})

                _preview = (
                    _text if len(_text) <= 2500 else _text[:2500] + "\n…(truncated)"
                )
                result_view = mo.vstack(
                    [
                        mo.md("### Step result").callout(kind="neutral"),
                        mo.md(
                            f"**reward** = `{_reward}` · "
                            f"**terminated** = `{_terminated}` · "
                            f"**truncated** = `{_truncated}` · "
                            f"**error_occurred** = `{_entry['error_occurred']}`"
                        ),
                        mo.md(
                            "Expected: exploring / unfueled setups often show "
                            "`reward=0.0` — nothing is mining yet."
                        ),
                        mo.accordion(
                            {
                                "raw_text (agent observation)": mo.ui.code_editor(
                                    value=_preview,
                                    language="text",
                                    disabled=True,
                                ),
                                "observation keys": mo.md(
                                    ", ".join(f"`{k}`" for k in _keys)
                                ),
                                "info highlights": mo.md(
                                    f"- `error_occurred`: `{_info.get('error_occurred')}`\n"
                                    f"- `ticks`: `{_info.get('ticks')}`\n"
                                    f"- has `output_game_state`: "
                                    f"`{'output_game_state' in _info}`"
                                ),
                            }
                        ),
                    ]
                )

    result_view
    return result_view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Observation fields that matter

    | Field | Friendly meaning |
    |-------|------------------|
    | `raw_text` | Stdout / stderr (+ throughput hint the task may append) |
    | `task_info` | Goal text, task key, trajectory length |
    | `inventory` | What you’re carrying |
    | `entities` | Machines on the map |
    | `flows` | Production rates |
    | `task_verification` | Quota hit yet? |

    Treat exceptions in `raw_text` as normal observations — fix them next step.
    """)
    return


@app.cell
def _(get_session, mo):
    _sess_hist = get_session()
    _rows = (_sess_hist or {}).get("history") or []
    if not _rows:
        history_view = mo.md("_Run a program to populate history._")
    else:
        history_view = mo.vstack(
            [
                mo.md(f"**Steps this session:** {len(_rows)}"),
                mo.ui.table(
                    [
                        {
                            "n": i + 1,
                            "reward": r["reward"],
                            "done": r["terminated"] or r["truncated"],
                            "error": r["error_occurred"],
                            "code_preview": r["code"].replace("\n", " ")[:80],
                        }
                        for i, r in enumerate(_rows)
                    ]
                ),
            ]
        )
    history_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Suggested next programs

    Paste these into the editor one at a time (same policy as
    `examples/03_scripted_miner.py`):

    **A — move + place drill**

    ```python
    iron = nearest(Resource.IronOre)
    move_to(iron)
    drill = place_entity(
        entity=Prototype.BurnerMiningDrill,
        position=iron,
        direction=Direction.NORTH,
    )
    print(drill)
    ```

    **B — fuel it**

    ```python
    entities = get_entities()
    print(entities)
    drill = next((e for e in entities if e.name == "burner-mining-drill"), None)
    if drill is not None:
        insert_item(Prototype.Coal, drill, quantity=5)
        print(drill)
    print(inspect_inventory())
    ```

    You’re “done” on this task when a **fueled** miner sits on ore, ore has
    somewhere to go (chest/belts), and measured rate ≥ **16 / 60s**
    (`terminated=True`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Friction cheat sheet

    | Symptom | Likely cause |
    |---------|----------------|
    | `missing run_idx` | Use `make_env(...)` (this notebook does) |
    | Hang on connect | Cluster not ready — [SETUP.md](../../docs/SETUP.md) |
    | Weird `AttributeError` | Strings instead of `Resource.*` / `Prototype.*` |
    | `unexpected keyword argument 'pos'` | Use `move_to(iron)`, not `move_to(pos=...)` |
    | Always `reward=0` | No automated mining, or no fuel / output buffer |
    | Program “failed” but process OK | Read `raw_text` — errors are observations |

    ### Where next

    - Scripted baseline: `uv run python examples/03_scripted_miner.py`
    - Maps: [VISUALIZATION.md](../../docs/VISUALIZATION.md)
    - LLM loop: `uv run python examples/04_dspy_agent_loop.py --steps 5`
    - Design loops: [SCENARIOS.md](../../docs/SCENARIOS.md)

    **Bottom line:** this env is the smallest complete FLE contract —
    **goal, kit, program-in / text-out, throughput score**.
    """)
    return


if __name__ == "__main__":
    app.run()
