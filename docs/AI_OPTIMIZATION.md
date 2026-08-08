# AI engine optimization

This repo uses **DSPy** as the AI layer on top of FLE's code-as-action interface.

Keep **intro**, **train**, and **run** concerns separate:

| Concern | Entry point | What it does |
|---------|-------------|--------------|
| **Intro runtime** | `examples/04_dspy_agent_loop.py` | Baseline `Predict` in Factorio (learning the loop) |
| **RLM runtime** | `examples/11_dspy_rlm_miner.py` | `dspy.RLM` REPL + `run_factorio` — same drill goal — [RLM_STARTER.md](RLM_STARTER.md) |
| **Flex intro** | `examples/12_dspy_flex_miner.py` | Baseline `dspy.Flex` outer loop — [FLEX_STARTER.md](FLEX_STARTER.md) |
| **Flex train** | `examples/13a_dspy_flex_train.py` | Online play demos → Flex+GEPA → save |
| **Flex run** | `examples/13b_dspy_flex_run.py` | Load learned Flex → Factorio rollout |
| **Bootstrap train** | `examples/05_optimize_agent.py` | Offline BootstrapFewShot → save JSON |
| **GEPA train** | `examples/07_gepa_train.py` | Offline GEPA on `Predict` → save JSON — [GEPA_STARTER.md](GEPA_STARTER.md) |
| **GEPA run** | `examples/08_gepa_run.py` | Load GEPA artifact → short Factorio rollout |

Train scripts never step Factorio. Run/intro scripts never call `compile()` / teleprompters.

## Architecture

```text
Intro (04)     build_agent() ──────────────────► Factorio
RLM (11)       dspy.RLM + run_factorio ─────────► Factorio   ┐ same
Flex (12)      baseline Flex outer loop ───────► Factorio   ┘ drill goal
                 │
Train (13a)    play demos → Flex+GEPA ──save──► .fle/flex_from_play.json
                                                      │
Run (13b)      Flex(sig).load(...) ◄──────────────────┘ ──► Factorio

Train (05/07)  TRAIN/VAL + Predict/GEPA ──save──► .fle/gepa_*.json
                                                      │
Run (08)       load_agent (Predict) ◄─────────────────┘ ──► Factorio
```

Core pieces:

- `dspy_factorio/agent.py` — signature, `build_agent`, `load_agent`, `propose_program`
- `dspy_factorio/flex_drill.py` — shared drill goal / Flex load helpers for 12/13a/13b
- `dspy_factorio/trainset.py` — `TRAIN_DEMOS` / `VAL_DEMOS` (07; 13a uses play demos + `FLEX_VAL_DEMOS`)
- `examples/04_…` — intro agent
- `examples/05_…` — BootstrapFewShot train
- `examples/07_…` / `08_…` — GEPA on Predict — [GEPA_STARTER.md](GEPA_STARTER.md)
- `examples/11_…` / `12_…` / `13a_…` / `13b_…` — RLM + Flex path — [RLM_STARTER.md](RLM_STARTER.md) / [FLEX_STARTER.md](FLEX_STARTER.md)

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

Recommended path: scripted `03` → intro `04` → advanced miners `11` (RLM) / `12` (Flex intro) → Flex learn-from-play `13a`/`13b` → GEPA on Predict `07`/`08` → measure with `06`.

## Next upgrades

1. Execution metric that runs programs in FLE (still `score` + `feedback` for GEPA)  
2. `dspy.BetterTogether` (Bootstrap → GEPA) once the metric is solid  
3. Store successful `04` / `08` / `12` / `13b` trajectories into `trainset`  
4. Raise Flex `13a` `--auto` after play demos look sane — [FLEX_STARTER.md](FLEX_STARTER.md)  
