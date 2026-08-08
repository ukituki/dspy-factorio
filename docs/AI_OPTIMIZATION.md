# AI engine optimization

This repo uses **DSPy** as the AI layer on top of FLE's code-as-action interface.

Keep **intro**, **train**, and **run** concerns separate:

| Concern | Entry point | What it does |
|---------|-------------|--------------|
| **Intro runtime** | `examples/04_dspy_agent_loop.py` | Baseline `Predict` in Factorio (learning the loop) |
| **RLM runtime** | `examples/11_dspy_rlm_miner.py` | `dspy.RLM` REPL + `run_factorio` tool — [RLM_STARTER.md](RLM_STARTER.md) |
| **Bootstrap train** | `examples/05_optimize_agent.py` | Offline BootstrapFewShot → save JSON |
| **GEPA train** | `examples/07_gepa_train.py` | Offline GEPA → save JSON — [GEPA_STARTER.md](GEPA_STARTER.md) |
| **GEPA run** | `examples/08_gepa_run.py` | Load GEPA artifact → short Factorio rollout |

Train scripts never step Factorio. Run/intro scripts never call `compile()` / teleprompters.

## Architecture

```text
Intro (04)     build_agent() ──────────────────► Factorio

Train (05/07)  TRAIN/VAL demos + optimizer ──save──► .fle/*.json
                                                      │
Run (08)       load_agent(path) ◄─────────────────────┘ ──► Factorio
```

Core pieces:

- `factorio_gym/agent.py` — signature, `build_agent`, `load_agent`, `propose_program`
- `factorio_gym/trainset.py` — `TRAIN_DEMOS` / `VAL_DEMOS` (train only)
- `examples/04_…` — intro agent
- `examples/05_…` — BootstrapFewShot train
- `examples/07_…` / `08_…` — GEPA train / run — [GEPA_STARTER.md](GEPA_STARTER.md)

## Intro runtime (example 04)

```bash
uv run python examples/04_dspy_agent_loop.py --steps 5 --model openai/gpt-4o-mini

# Save a map PNG after each step (no Factorio desktop client needed)
uv run python examples/04_dspy_agent_loop.py --steps 3 --renders
open .fle/renders/dspy_agent/step_02.png
```

Flow: bootstrap inventory + iron → `propose_program` → `step_code` → optional
`save_render`. Uses signature + `API_HINT` only (no compiled demos). See
[VISUALIZATION.md](VISUALIZATION.md) § DSPy agent loop.

## Bootstrap train (example 05)

```bash
uv run python examples/05_optimize_agent.py \
  --model openai/gpt-4o-mini \
  --save .fle/optimized_factorio_agent.json
```

Saves a compiled module. To roll it out, mirror `08_gepa_run.py` (swap the `--program` path) rather than overloading example 04.

For reflective optimization with textual feedback, prefer GEPA: [GEPA_STARTER.md](GEPA_STARTER.md).

## What to optimize

1. **Prompt / demos** — expand `TRAIN_DEMOS` / BootstrapFewShot / GEPA  
2. **Model choice** — mini for iteration, stronger models for hard tasks  
3. **Trajectory policy** — step budget, early stop, observation trimming  
4. **Tool curriculum** — mining → smelting → belts → science  

## inspect-eval vs custom loop

| | Custom DSPy loop | `fle inspect-eval` |
|--|------------------|--------------------|
| Flexibility | High | Medium |
| Comparability | Low | High |
| Best for | Prompt/engine R&D | Benchmark numbers |

Recommended path: scripted `03` → intro `04` (or RLM `11`) → GEPA train `07` → GEPA run `08` → measure with `06`.

## Next upgrades

1. Execution metric that runs programs in FLE (still `score` + `feedback` for GEPA)  
2. `dspy.BetterTogether` (Bootstrap → GEPA) once the metric is solid  
3. Store successful `04` / `08` trajectories into `trainset`  
4. Multi-module pipeline: Planner → Coder → Critic  
