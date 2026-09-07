# Five-minute meetup demo: an AI builds its first Factorio machine

## Recommendation

Show a DSPy agent finding iron, moving to it, placing a burner mining drill,
and adding coal. Put the live Factorio client beside a readable view of the
agent's generated program and the game's response. The audience should see
the connection between a decision and a visible change in the world.

The presentation notebook is `examples/04_dspy_agent_notebook.py`, based on
the Predict loop from `examples/04_dspy_agent_loop.py`. It includes a one-drill
goal, the full throughput task, model selection, a Predict/ChainOfThought
switch, inline map images, and episode review. Launch it with:

```bash
uv run marimo edit examples/04_dspy_agent_notebook.py
```

Use presentation view and keep autorun enabled. The run button is the explicit
gate: changing settings does not run the game. Each episode resets the map and
saves a transcript and PNGs under `.fle/renders/meetup/`. The camera zooms in
once a drill appears. The original CLI script remains available unchanged.

Suggested title: **“Can an LLM build a factory? Giving AI a world to act in.”**

The takeaway: an agent generates an action, observes what actually happened,
and uses that feedback for its next action. DSPy defines the input/output
contract; FLE executes the generated Python and returns game observations.
This path uses text observations, not screen vision or keyboard control.

## Five-minute running order

| Time | On screen | What to explain |
|---|---|---|
| 0:00–0:30 | Factorio map and the one-drill goal | Factorio makes planning and resource constraints visible. |
| 0:30–1:00 | `goal + observation → program → game → observation` | Show the DSPy signature and one short generated action. |
| 1:00–2:45 | Live agent and game side by side | Find iron, move, place, fuel. Narrate the game feedback; explain an error if one naturally occurs. |
| 2:45–3:30 | Drill and fresh game-state evidence | Verify the drill exists and has fuel. Show actual production only if measured. |
| 3:30–4:20 | One captured action/response pair | Explain why feedback matters and how an agent can correct its next action. |
| 4:20–5:00 | Learning-path overview and repository link | Introduce optimization as the next experiment and leave room for one question. |

Aim for a live rollout below 105 seconds. This is a rehearsal target, not a
measured runtime. If it stalls for 20 seconds without useful visible progress,
switch to a clearly labelled recording of a real rehearsal.

## Why this path

- **Predict (`04`):** the clearest loop to explain and a small runtime surface;
  already prints code, observations, rewards, and saves session-specific PNGs.
- **RLM (`11`):** useful for a longer technical talk, but adds a second Python
  sandbox and extra latency. Its success flag is model-reported.
- **Flex (`12`, `13a`, `13b`):** a useful follow-up on learning program structure.
  The local saved training metadata reports six play demos and
  `module_src_changed: false`; it does not establish improved live performance.
- **GEPA (`07`, `08`):** train before a talk. The current metric checks generated
  code heuristically, so a high score does not establish better ore throughput.
- **Scripted miner (`03`, `10`):** good for checking the game connection and as
  an explicitly labelled scripted fallback, but alone does not demonstrate AI.
- **Marimo scenario notebook:** useful for preparation and exploring task state;
  the five-minute presentation should concentrate on the game and agent loop.

## Existing commands for preparation

Run from the repository root. These are existing entry points, not a finished
meetup runner. Preparation can reset the game or recreate the local server;
finish it before the audience sees the screen.

```bash
# Start only if the cluster is not already running.
uv run fle cluster start -n 1

# Prepare the server for a desktop spectator, before the talk.
uv run python examples/10_live_client_watch.py --prepare-only

# Existing baseline rehearsal; currently targets the broader environment goal.
uv run python examples/04_dspy_agent_loop.py --steps 4 --renders

# Scripted visual rehearsal/fallback, clearly identified as scripted.
uv run python examples/10_live_client_watch.py --skip-prepare --pause 8
```

Connect the desktop game to `127.0.0.1:34197`; match the actual server version
(the repository documents 2.0.73). Keep only one script controlling the game.
Use the installed dependencies and confirm model access before the event.
Choose the model using timed rehearsals; the existing script defaults to
`openai/gpt-4o-mini`, which has not been evaluated in this orientation pass.

## Remaining rehearsal work

1. Select the model and step budget using rehearsals. The notebook uses a
   45-second provider request timeout and no retries, but game operations and
   total episode time are not bounded by that timeout. Initialization currently
   runs after clicking the episode button, so include it in the talk timing.
2. Consider a fresh game-state check for the placed drill and fuel. Do not use the
   Flex text heuristic as proof: it can accept a program containing a coal
   insertion call without confirming that the call succeeded.
3. Check the notebook at projector size and record a successful run as backup.
   The in-session episode picker supports comparing runs; saved JSON and PNGs
   remain on disk after the notebook session ends.
4. Compare Predict and ChainOfThought with the same model and budget. The
   displayed rationale is a model-written explanation, not a success verdict.
   Extra explanation does not guarantee better actions.
5. If the drill sequence is reliably fast, extend it with a chest at the drill's
   output and show an increasing ore count. Keep this optional until rehearsed.

A fueled drill is a small milestone, not completion of the environment's
16-ore-per-60-seconds task. Report throughput only after measuring it.

### Initial notebook checks

Live checks with `openai/gpt-4o-mini`: Predict / 2 steps took 57.8 seconds and
the game reported a placed drill with coal. ChainOfThought / 2 steps took
49.7 seconds; ChainOfThought / 4 steps took 72.5 seconds. Both ChainOfThought
runs produced visible rationales but did not reach the fueled-drill milestone.
These are individual rehearsals, not a performance benchmark. The notebook
also passed marimo's strict check; control changes and invalid custom model
input were checked to leave the completed-episode history intact.

## Orientation notes — 2026-09-07

- Branch: `codex/ai-meetup-demo`, created from local `master` at `60adee1`.
  Local master contained four commits beyond `origin/master`; they are included.
- `dspy_factorio/env.py` owns the FLE connection, steps, observations, and renders.
- `dspy_factorio/agent.py` owns the DSPy signature and model configuration.
- Numbered examples separate baseline play, optimization, RLM, and Flex.
- `uv`, Deno, and the local Factorio Docker container were present; the container
  reported running with the game and RCON ports published.
- This pass inspected code, documentation, and local artifact metadata. It did
  not run an LLM rollout, verify RCON authentication, or measure demo latency.
