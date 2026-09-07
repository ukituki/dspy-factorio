# DSPy Factorio

Step-by-step practice for learning **[DSPy](https://dspy.ai)** by playing **Factorio** through the [Factorio Learning Environment (FLE)](https://jackhopkins.github.io/factorio-learning-environment/).

You write (or optimize) small Python programs that act in the game — `Predict`, `GEPA`, `RLM`, `Flex` — and see the factory respond. The game is the gym; DSPy is what you’re learning.

```text
FLE / Factorio  ←── programs / tools ──  DSPy modules
     obs text  ──────────────────────►  (Predict → GEPA → RLM → Flex)
```

## Learning path

| Step | What you practice | Entry |
|------|-------------------|-------|
| 1 | Connect + one FLE program | [HELLO_WORLD.md](docs/HELLO_WORLD.md) · `examples/01_…` |
| 1b | Task / obs / reward concepts | [TASKS_GUIDE.md](docs/TASKS_GUIDE.md) · `scenarios/1_iron_ore_throughput/01_explore_env.py` |
| 2 | Scripted multi-step play | `examples/03_scripted_miner.py` |
| 3 | DSPy `Predict` agent loop | `examples/04_dspy_agent_loop.py` |
| 4 | Optimize instructions (GEPA) | [GEPA_STARTER.md](docs/GEPA_STARTER.md) · `07`/`08` |
| 5 | REPL agent (`dspy.RLM`) | [RLM_STARTER.md](docs/RLM_STARTER.md) · `11` |
| 6 | Structure search (`dspy.Flex`) | [FLEX_STARTER.md](docs/FLEX_STARTER.md) · `12`/`13a`/`13b` |

Same early milestone across advanced paths: **place and fuel one burner mining drill**.

## Quick start

```bash
# 1) deps
uv sync

# 2) API keys
cp .env.example .env
# put OPENAI_API_KEY=... in .env

# 3) Factorio cluster (Docker required)
uv run fle cluster start -n 1
# wait ~30–90s on Apple Silicon for RCON to come up

# 4) Hello World
uv run python examples/01_hello_world.py
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/SETUP.md](docs/SETUP.md) | Install, cluster, env vars, Apple Silicon notes |
| [docs/HELLO_WORLD.md](docs/HELLO_WORLD.md) | First working program end-to-end |
| [docs/TASKS_GUIDE.md](docs/TASKS_GUIDE.md) | Task concepts via `iron_ore_throughput` (goal, obs, reward) |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | PNG map dumps (no client) + optional live Factorio client |
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | Building scripted + LLM scenarios |
| [docs/AI_OPTIMIZATION.md](docs/AI_OPTIMIZATION.md) | DSPy runtime vs optimization tracks |
| [docs/GEPA_STARTER.md](docs/GEPA_STARTER.md) | Minimal GEPA optimize → load flow |
| [docs/RLM_STARTER.md](docs/RLM_STARTER.md) | `dspy.RLM` REPL agent → place a fueled drill |
| [docs/FLEX_STARTER.md](docs/FLEX_STARTER.md) | `dspy.Flex` intro → train from play → run |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Docker / RCON / eval pitfalls we hit |

FLE reference (0.3.x docs — partly outdated vs installed 0.4.x):

- [Overview](https://jackhopkins.github.io/factorio-learning-environment/versions/0.3.0.html)
- [Sphinx / API](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/)
- [Quickstart](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/getting_started/quickstart.html)

## Examples

### Meetup notebook: Predict → ChainOfThought

```bash
uv run marimo edit examples/04_dspy_agent_notebook.py
```

Interactive version of example 04: choose a model, task, step budget, and
**Predict** or **ChainOfThought**, then click **Run new episode**. Each run
resets Factorio, displays generated programs and game-state images, and keeps
the model's written rationale beside the game response in ChainOfThought mode.
Use marimo's presentation view for the talk and keep **On cell change → autorun**
enabled. Changing controls alone does not start an episode.

Requires the running Factorio cluster and provider credentials in `.env`.
Fresh model requests bypass DSPy's cache. Images and JSON transcripts are saved
under `.fle/renders/meetup/<episode>/`; the review picker keeps runs from the
current notebook session. The displayed stop reason is not a success verdict.

| Script | Purpose |
|--------|---------|
| `examples/01_hello_world.py` | Connect + `nearest(Resource.IronOre)` |
| `examples/02_list_environments.py` | List FLE task IDs |
| `examples/03_scripted_miner.py` | Deterministic multi-step scenario |
| `examples/09_visualize_renders.py` | Save map PNGs after each action |
| `examples/10_live_client_watch.py` | Join Factorio client + slow watchable scenario |
| `examples/04_dspy_agent_loop.py` | Intro DSPy agent loop (`--renders` for map PNGs) |
| `examples/04_dspy_agent_notebook.py` | Meetup notebook: model selection, Predict/ChainOfThought, images and episode review |
| `examples/05_optimize_agent.py` | Offline BootstrapFewShot train |
| `examples/06_run_inspect_eval.py` | Thin wrapper for `fle inspect-eval` |
| `examples/07_gepa_train.py` | Offline GEPA train → save module |
| `examples/08_gepa_run.py` | Run a GEPA-compiled module in Factorio |
| `examples/11_dspy_rlm_miner.py` | `dspy.RLM` + `run_factorio` tool ([RLM_STARTER.md](docs/RLM_STARTER.md)) |
| `examples/12_dspy_flex_miner.py` | Flex intro (baseline), same drill goal ([FLEX_STARTER.md](docs/FLEX_STARTER.md)) |
| `examples/13a_dspy_flex_train.py` | Online play → demos → Flex+GEPA compile |
| `examples/13b_dspy_flex_run.py` | Load learned Flex → Factorio rollout |

## Important 0.4.x differences from upstream quickstart

- `fle eval` is **removed** → use `fle inspect-eval`
- Always pass `--model ...` (omitting it can crash)
- With **one** Factorio container use `--epochs 1` (default Pass@8 needs 8 instances)
- `gym.make(env_id)` needs `run_idx=0` — prefer `dspy_factorio.env.make_env`
- Pin `a2a-sdk>=0.3.26,<1` (1.x breaks imports)
- FLE still uses OpenAI `gym` (helpers silence deprecation noise; do not blindly switch to Gymnasium)
- LLM agents must emit `Resource.*` / `Prototype.*` enums and `move_to` before distant placements

```bash
uv run fle inspect-eval \
  --env-id iron_ore_throughput \
  --model openai/gpt-4o-mini \
  --limit 1 --epochs 1 \
  --trajectory-length 64 \
  --max-connections 1
```

## Project layout

```
dspy_factorio/          # env + agent + offline trainset
examples/              # numbered DSPy / FLE practice scripts
docs/                  # setup + step-by-step tutorials
.env                   # secrets (gitignored)
```

Repo: [github.com/ukituki/dspy-factorio](https://github.com/ukituki/dspy-factorio)
