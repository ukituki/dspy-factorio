# Building scenarios

A **scenario** here means a repeatable loop: reset a task → run N programs → score / log.

## Choose a task

```bash
uv run python examples/02_list_environments.py --throughput
uv run python examples/02_list_environments.py --search science
```

Common starters:

| Env ID | Goal sketch |
|--------|-------------|
| `iron_ore_throughput` | Automate 16 iron ore / 60s |
| `iron_plate_throughput` | Automate 16 iron plate / 60s |
| `automation_science_pack_throughput` | Red science automation |
| `open_play` | Maximize production score |

## Pattern A — Scripted baseline

Use when validating the environment or collecting expert demos.

```python
from factorio_gym.env import make_env, reset_env, step_code, obs_text

STEPS = [
    "iron = nearest(Resource.IronOre); print(iron); move_to(iron)",
    "drill = place_entity(entity=Prototype.BurnerMiningDrill, position=iron, direction=Direction.NORTH); print(drill)",
]

env = make_env("iron_ore_throughput", run_idx=0)
reset_env(env)
for code in STEPS:
    obs, reward, done, trunc, info = step_code(env, code)
    print(obs_text(obs), reward)
env.close()
```

See `examples/03_scripted_miner.py`.

**Tips**

- Keep each step small and printable.
- Call `move_to(iron)` (or `move_to(position=iron)`) before `place_entity` when the patch is far from the player (max place distance is ~10). Never `move_to(pos=...)`.
- Use `Resource.*` / `Prototype.*` enums — string names raise AttributeError inside FLE.
- Persist `info["output_game_state"]` if you need restore/resume later.
- Failures in Lua/Python show up in `raw_text` — treat them as observations, not process crashes.

## Pattern B — Intro LLM / DSPy agent

```text
bootstrap inventory+iron  →  build_agent()  →  Python program  →  env.step
```

See `examples/04_dspy_agent_loop.py`. Baseline only — no teleprompting, no `--load`.

Design knobs: `--steps`, model, observation truncation, `API_HINT`.

For compiled GEPA agents use `07` (train) + `08` (run) — [GEPA_STARTER.md](GEPA_STARTER.md).

## Pattern B2 — `dspy.RLM` (REPL owns the loop)

```text
RLM sandbox  →  run_factorio(code) tool  →  Factorio  →  REPL  →  SUBMIT
```

One RLM call explores with Deno/Pyodide Python and a host tool that steps FLE. Best for error-fixing / probe-then-act. Tutorial: [RLM_STARTER.md](RLM_STARTER.md) (`examples/11_dspy_rlm_miner.py`). Needs Deno installed.

## Pattern C — Official inspect-eval harness

Use for comparable benchmarks and logging:

```bash
uv run python examples/06_run_inspect_eval.py --env-id iron_ore_throughput --steps 64
```

With one Factorio container always pass `--epochs 1`.

## Adding your own scenario script

1. Copy `examples/03_scripted_miner.py`
2. Change `STEPS` (or call `propose_program`)
3. Log `(step, code, raw_text, reward)` to JSONL under `.fle/runs/`
4. Feed good trajectories into `examples/05_optimize_agent.py` as DSPy examples

## Suggested development loop

```text
scripted baseline works?
        │
        ▼
collect few successful trajectories
        │
        ▼
DSPy BootstrapFewShot (offline)
        │
        ▼
agent loop on live env
        │
        ▼
inspect-eval for longer rollouts
```

## Scoring

Throughput tasks reward production toward a quota. Short Hello World runs often score `0.0` — that is normal until automation actually runs for the evaluation window (`sleep`, belts, fueled drills, etc.).
