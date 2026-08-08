# Factorio Gym

Local workspace for developing Factorio Learning Environment (FLE) scenarios and optimizing AI agents with DSPy.

Official docs (0.3.x — partly outdated vs installed 0.4.x):

- [FLE 0.3.0 release / overview](https://jackhopkins.github.io/factorio-learning-environment/versions/0.3.0.html)
- [Sphinx docs (API, tools, sprites, MCP)](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/)
- [Quickstart](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/getting_started/quickstart.html)

## Quick start

```bash
# 1) deps (already in pyproject.toml)
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
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | PNG map dumps (no client) + optional live Factorio client |
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | Building scripted + LLM scenarios |
| [docs/AI_OPTIMIZATION.md](docs/AI_OPTIMIZATION.md) | DSPy runtime vs optimization tracks |
| [docs/GEPA_STARTER.md](docs/GEPA_STARTER.md) | Minimal GEPA optimize → load flow |
| [docs/RLM_STARTER.md](docs/RLM_STARTER.md) | `dspy.RLM` REPL agent → place a fueled drill |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Docker / RCON / eval pitfalls we hit |

## Examples

| Script | Purpose |
|--------|---------|
| `examples/01_hello_world.py` | Connect + `nearest(Resource.IronOre)` |
| `examples/02_list_environments.py` | List FLE task IDs |
| `examples/03_scripted_miner.py` | Deterministic multi-step scenario |
| `examples/09_visualize_renders.py` | Save map PNGs after each action |
| `examples/10_live_client_watch.py` | Join Factorio client + slow watchable scenario |
| `examples/04_dspy_agent_loop.py` | Intro DSPy agent loop (`--renders` for map PNGs) |
| `examples/05_optimize_agent.py` | Offline BootstrapFewShot train |
| `examples/06_run_inspect_eval.py` | Thin wrapper for `fle inspect-eval` |
| `examples/07_gepa_train.py` | Offline GEPA train → save module |
| `examples/08_gepa_run.py` | Run a GEPA-compiled module in Factorio |
| `examples/11_dspy_rlm_miner.py` | `dspy.RLM` + `run_factorio` tool ([RLM_STARTER.md](docs/RLM_STARTER.md)) |

## Important 0.4.x differences from upstream quickstart

- `fle eval` is **removed** → use `fle inspect-eval`
- Always pass `--model ...` (omitting it can crash)
- With **one** Factorio container use `--epochs 1` (default Pass@8 needs 8 instances)
- `gym.make(env_id)` needs `run_idx=0` — prefer `factorio_gym.env.make_env`
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
factorio_gym/          # env + agent + offline trainset
examples/              # 04=Predict loop, 11=RLM, 07/08=GEPA
docs/                  # setup + tutorials
.env                   # secrets (gitignored)
```
