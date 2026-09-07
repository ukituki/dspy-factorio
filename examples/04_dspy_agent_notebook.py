import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/04_dspy_agent_notebook.slides.json",
)


@app.cell(hide_code=True)
def imports():
    import html
    import json
    import sys
    import time
    from pathlib import Path
    from uuid import uuid4

    import dspy
    import marimo as mo

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dspy_factorio.agent import API_HINT, AgentConfig, FactorioProgrammer, build_lm, strip_code_fences
    from dspy_factorio.env import get_environment_info, load_project_env, make_env, obs_text, reset_env, save_render, step_code
    load_project_env()
    return (
        API_HINT,
        AgentConfig,
        FactorioProgrammer,
        Path,
        build_lm,
        dspy,
        get_environment_info,
        html,
        json,
        make_env,
        mo,
        obs_text,
        project_root,
        reset_env,
        save_render,
        step_code,
        strip_code_fences,
        time,
        uuid4,
    )


@app.cell(hide_code=True)
def introduction(mo):
    mo.md("""
    # An AI builds its first Factorio machine

    **A live DSPy agent lab · Predict → ChainOfThought**

    Give a language model a goal and a way to act. It writes a short Python
    program; Factorio executes it and reports what happened. The next action
    uses that feedback. This notebook follows **example 04**, with the same
    signature, API hints, bootstrap, and environment helpers.

    ### 1 · The task

    Our five-minute milestone: **find iron → move there → place a drill → add coal**.
    A burner mining drill needs an ore patch and fuel. The full environment asks
    for sustained iron-ore production; placing a fueled drill is only a milestone.

    The agent reads **text observations**. The map images below are for us to
    watch the world change; they are not sent to the model.
    """)
    return


@app.cell(hide_code=True)
def task(get_environment_info, mo):
    env_id = "iron_ore_throughput"
    environment_description = (get_environment_info(env_id) or {}).get("description") or env_id
    drill_goal = (
        "Place one BurnerMiningDrill on nearest iron ore, move_to first if needed, "
        "insert 5 coal, and print fresh entities/inventory as proof. "
        "Take one small verifiable step per program. If a fueled drill already exists, "
        "inspect it instead of placing another."
    )
    bootstrap = "print(inspect_inventory())\niron = nearest(Resource.IronOre)\nprint(f'iron={iron}')"
    mo.accordion({"Full environment goal": mo.plain_text(environment_description)})
    return bootstrap, drill_goal, env_id, environment_description


@app.cell(hide_code=True)
def explain_loop(mo):
    mo.md("""
    ### 2 · The loop

    **Goal + latest observation → DSPy → Python program → Factorio → new observation ↻**

    1. **Reset** the map and inspect inventory / nearest iron (scripted bootstrap).
    2. **Propose** one action with the selected model and module.
    3. **Execute** only the generated `program` in the game.
    4. **Observe** stdout, errors, reward, and a fresh map image; repeat.

    Like example 04, this agent receives the **latest observation**, not the entire
    transcript. The episode history below is for the audience. An episode ends
    at its step budget or when the environment reports done.

    ### 3 · Change one module

    `Predict` returns a program directly. `ChainOfThought` adds a `reasoning`
    output before the program. We display that model-written rationale separately
    from the actual game response. It is an explanation to inspect, not proof
    that the action worked.

    Run **Predict** first, then select **ChainOfThought** and run again with the
    same model, task, and step budget. Neither module learns between episodes.
    """)
    return


@app.cell
def module_factory(FactorioProgrammer, dspy, mo):
    def make_demo_agent(module_name: str) -> dspy.Module:
        if module_name == "ChainOfThought":
            return dspy.ChainOfThought(
                FactorioProgrammer,
                rationale_field=dspy.OutputField(
                    desc="Brief action rationale: what the latest observation implies, "
                         "what you will do next, and what game evidence to check."
                ),
            )
        return dspy.Predict(FactorioProgrammer)

    mo.md("""
    The change in DSPy is small; the game loop stays the same:

    ```python
    agent = dspy.Predict(FactorioProgrammer)
    # Switch to:
    agent = dspy.ChainOfThought(FactorioProgrammer)

    prediction = agent(goal=goal, observation=observation, inventory_hint=API_HINT)
    print(prediction.reasoning)  # ChainOfThought only
    program = prediction.program
    ```

    The constructor above also asks for a brief, action-focused rationale.
    Only `program` is executed in Factorio.
    """)
    return (make_demo_agent,)


@app.cell(hide_code=True)
def controls(mo):
    model_picker = mo.ui.dropdown(
        ["openai/gpt-4o-mini", "openai/gpt-5.1", "Custom model"],
        value="openai/gpt-4o-mini", label="Model", full_width=True,
    )
    custom_model = mo.ui.text(placeholder="provider/model", label="Custom model ID", full_width=True)
    module_picker = mo.ui.dropdown(["Predict", "ChainOfThought"], value="Predict", label="DSPy module")
    task_picker = mo.ui.dropdown(["One fueled drill", "Full throughput task"], value="One fueled drill", label="Task")
    step_budget = mo.ui.slider(1, 10, value=4, step=1, label="Agent steps", show_value=True)
    run_episode = mo.ui.run_button(label="Run new episode · reset game", kind="success", full_width=True)
    mo.vstack([
        mo.md("### 4 · Run the agent"),
        mo.hstack([model_picker, module_picker], widths="equal"),
        custom_model,
        mo.hstack([task_picker, step_budget], widths="equal"),
        run_episode,
        mo.md("Choose settings, then click **Run new episode**. Every click starts from a reset map and makes fresh model requests. Keep marimo's **On cell change → autorun** enabled for this button workflow. Stop a running episode with marimo's interrupt button."),
        mo.accordion({"Before presenting": mo.md("Run the Factorio cluster and configure provider credentials in the project's `.env`. Only one script should control the instance. Optional desktop spectator: `127.0.0.1:34197`. This notebook saves schematic PNGs without requiring sprite downloads.")}),
    ])
    return (
        custom_model,
        model_picker,
        module_picker,
        run_episode,
        step_budget,
        task_picker,
    )


@app.cell(hide_code=True)
def history_state(mo):
    get_episodes, set_episodes = mo.state([])
    return get_episodes, set_episodes


@app.cell(hide_code=True)
def render_helpers(Path, html, mo):
    def text_panel(text: str) -> mo.Html:
        return mo.Html(
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;'
            'overflow:auto;font-size:13px;padding:12px;border:1px solid var(--gray-5);'
            'border-radius:8px">' + html.escape(text) + '</pre>'
        )


    def step_card(record: dict) -> mo.Html:
        details = [mo.md(f"**{record['label']}**")]
        if record.get("reasoning"):
            details.extend([mo.md("**Model rationale · ChainOfThought**"), text_panel(record["reasoning"])])
        if record.get("program"):
            details.extend([mo.md("**Generated program**" if record["index"] > 0 else "**Scripted bootstrap**"), text_panel(record["program"])])
        details.extend([mo.md("**Game response**"), text_panel(record["observation"])])
        if record.get("reward") is not None:
            details.append(mo.md(f"Reward: **{record['reward']}** · Environment done: **{record['done']}**"))
        if record.get("image"):
            return mo.hstack([
                mo.vstack(details),
                mo.image(Path(record["image"]), alt=record["label"], width="100%", caption="Game state after this step"),
            ], widths="equal", align="start")
        return mo.vstack(details)

    return step_card, text_panel


@app.cell(hide_code=True)
def camera(Path, save_render):
    from dspy_factorio.env import namespace

    def save_episode_image(env, path: Path, *, follow_factory: bool = True) -> Path:
        """Keep context before building; zoom in when a drill appears."""
        has_drill = follow_factory and any(
            entity.name in {"burner-mining-drill", "electric-mining-drill"}
            for entity in namespace(env).get_entities()
        )
        return save_render(
            env, path, mode="simple", overview=True,
            zoom=2.0 if has_drill else 0.25,
        )

    return (save_episode_image,)


@app.cell
def episode_loop(
    API_HINT,
    AgentConfig,
    bootstrap,
    build_lm,
    custom_model,
    drill_goal,
    dspy,
    env_id,
    environment_description,
    json,
    make_demo_agent,
    make_env,
    mo,
    model_picker,
    module_picker,
    obs_text,
    project_root,
    reset_env,
    run_episode,
    save_episode_image,
    set_episodes,
    step_budget,
    step_card,
    step_code,
    strip_code_fences,
    task_picker,
    text_panel,
    time,
    uuid4,
):
    mo.stop(not run_episode.value, mo.md("Ready when you are. Select a model and click **Run new episode**."))
    _model = custom_model.value.strip() if model_picker.value == "Custom model" else model_picker.value
    mo.stop(not _model or "/" not in _model, mo.callout("Enter a model ID such as provider/model.", kind="warn"))
    _goal = drill_goal if task_picker.value == "One fueled drill" else environment_description
    _started = time.monotonic()
    _episode_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    _out = project_root / ".fle" / "renders" / "meetup" / _episode_id
    _out.mkdir(parents=True, exist_ok=True)
    _episode = {
        "id": _episode_id, "model": _model, "module": module_picker.value,
        "goal": _goal, "budget": step_budget.value, "steps": [], "status": "running",
        "elapsed": 0.0, "directory": str(_out),
    }
    _env = None
    mo.output.replace(mo.md(f"**Connecting to Factorio…** · {_model} · {module_picker.value}"))
    try:
        _agent = make_demo_agent(module_picker.value)
        _lm = build_lm(AgentConfig(model=_model)).copy(cache=False, num_retries=0, timeout=45)
        _env = make_env(env_id, run_idx=0)
        reset_env(_env)
        _reset_image = save_episode_image(_env, _out / "reset.png", follow_factory=False)
        mo.output.append(mo.image(_reset_image, width=500, caption="Reset map · before the scripted bootstrap"))
        _episode["reset_image"] = str(_reset_image)
        _obs, _reward, _terminated, _truncated, _ = step_code(_env, bootstrap)
        _observation = obs_text(_obs) or "(empty observation)"
        _image = save_episode_image(_env, _out / "step_00.png", follow_factory=False)
        _record = {"index": 0, "label": "Bootstrap · inventory and nearest iron", "program": bootstrap,
                   "reasoning": "", "observation": _observation, "image": str(_image),
                   "reward": float(_reward), "done": bool(_terminated or _truncated)}
        _episode["steps"].append(_record)
        mo.output.append(step_card(_record))

        with dspy.context(lm=_lm):
            for _step in range(1, int(step_budget.value) + 1):
                if _terminated or _truncated:
                    break
                mo.output.append(mo.md(f"**Step {_step}/{step_budget.value} · asking {_model}…**"))
                _prediction = _agent(goal=_goal, observation=_observation, inventory_hint=API_HINT)
                _program = strip_code_fences(_prediction.program)
                _reasoning = getattr(_prediction, "reasoning", "") or ""
                _record = {"index": _step, "label": f"Agent step {_step}", "program": _program,
                           "reasoning": _reasoning, "observation": "Execution pending", "image": "",
                           "reward": None, "done": False}
                _episode["steps"].append(_record)
                if _reasoning:
                    mo.output.append(mo.vstack([mo.md("**Model rationale · before execution**"), text_panel(_reasoning)]))
                mo.output.append(text_panel(_program))
                _obs, _reward, _terminated, _truncated, _ = step_code(_env, _program)
                _observation = obs_text(_obs) or "(empty observation)"
                _record.update(observation=_observation, reward=float(_reward), done=bool(_terminated or _truncated))
                _image = save_episode_image(_env, _out / f"step_{_step:02d}.png")
                _record["image"] = str(_image)
                mo.output.append(step_card(_record))
        _episode["status"] = "environment done" if _terminated or _truncated else "step budget reached"
    except KeyboardInterrupt:
        _episode["status"] = "interrupted"
    except Exception as _error:
        # Notebook boundary: keep completed steps visible when a provider or game call fails.
        _episode["status"] = "error"
        _episode["error"] = f"{type(_error).__name__}: {_error}"
    finally:
        if _env is not None:
            try:
                _env.close()
            except Exception as _close_error:
                _episode["cleanup_error"] = f"{type(_close_error).__name__}: {_close_error}"
        _episode["elapsed"] = round(time.monotonic() - _started, 1)
        (_out / "episode.json").write_text(json.dumps(_episode, indent=2), encoding="utf-8")
        set_episodes(lambda previous: [*previous, _episode])

    mo.output.replace(mo.callout(
        f"Episode saved · {_episode['status']} · {_episode['elapsed']} seconds. Review its steps below.",
        kind="warn" if _episode["status"] in ("error", "interrupted") else "info",
    ))
    return


@app.cell(hide_code=True)
def episode_selector(get_episodes, mo):
    _runs = get_episodes()
    _options = {
        f"{_i + 1} · {_r['module']} · {_r['model']} · {_r['elapsed']}s": _r["id"]
        for _i, _r in enumerate(_runs)
    } or {"No episodes yet": None}
    review_episode = mo.ui.dropdown(
        _options, value=list(_options)[-1], disabled=not _runs,
        label="Review episode", full_width=True,
    )
    _summary = [mo.md("### 5 · Episode review")]
    if _runs:
        _summary.append(mo.ui.table([
            {"Episode": _i + 1, "Module": _r["module"], "Model": _r["model"],
             "Agent steps": sum(_s["index"] > 0 for _s in _r["steps"]),
             "Seconds": _r["elapsed"], "Stop reason": _r["status"]}
            for _i, _r in enumerate(_runs)
        ], selection=None, show_column_summaries=False, show_data_types=False))
    else:
        _summary.append(mo.md("Your runs will appear here. Changing settings does not erase completed episodes."))
    _summary.append(review_episode)
    mo.vstack(_summary)
    return (review_episode,)


@app.cell(hide_code=True)
def episode_review(
    Path,
    get_episodes,
    json,
    mo,
    review_episode,
    step_card,
    text_panel,
):
    mo.stop(review_episode.value is None)
    _selected = next(_r for _r in get_episodes() if _r["id"] == review_episode.value)
    _review = [mo.md(f"**{_selected['module']} · {_selected['model']}**"), mo.plain_text(_selected["goal"])]
    if _selected.get("error"):
        _review.append(mo.callout(text_panel(_selected["error"]), kind="danger"))
    if _selected.get("cleanup_error"):
        _review.append(mo.callout(text_panel(_selected["cleanup_error"]), kind="warn"))
    if _selected.get("reset_image"):
        _review.append(mo.accordion({"Reset map": mo.image(Path(_selected["reset_image"]), width=600)}))
    _review.extend(step_card(_s) for _s in _selected["steps"])
    _review.extend([
        mo.md("**Read the evidence:** a finished step budget does not mean task success. Inspect fresh game output for drill state/fuel; the environment reward measures its own task. Compare the rationale with what actually happened."),
        mo.download(data=json.dumps(_selected, indent=2).encode(), filename=f"{_selected['id']}.json", label="Download episode transcript"),
        mo.plain_text("Saved images and transcript: " + _selected["directory"]),
    ])
    mo.vstack(_review, gap=2)
    return


if __name__ == "__main__":
    app.run()
