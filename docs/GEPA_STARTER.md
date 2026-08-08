# GEPA starter (Factorio Gym)

Minimal DSPy [GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto) demo with a clear train / run split. Independent from the intro agent loop (`examples/04_dspy_agent_loop.py`).

This is a **starter**, not a production trainer. Prefer `auto="light"` until the metric is trustworthy.

## Full flow

```bash
# Train (offline — needs OPENAI_API_KEY; no Factorio)
uv run python examples/07_gepa_train.py --dry-run
uv run python examples/07_gepa_train.py --auto light

# Run (online — needs Factorio cluster)
uv run fle cluster start -n 1   # if needed
uv run python examples/08_gepa_run.py --steps 3
```

```text
07_gepa_train.py                         08_gepa_run.py
 TRAIN_DEMOS + VAL_DEMOS                   load_agent(program)
 rich metric (score+feedback)              short Factorio rollout
        │                                         ▲
        └──► .fle/gepa_factorio_agent.json ────────┘
```

| Concern | Script |
|---------|--------|
| **Train / compile** | `examples/07_gepa_train.py` |
| **Run compiled module** | `examples/08_gepa_run.py` |
| Intro baseline DSPy (no GEPA) | `examples/04_dspy_agent_loop.py` |

## Why GEPA here

| Piece | Role |
|-------|------|
| `FactorioProgrammer` | Student program (`dspy.Predict`) |
| `TRAIN_DEMOS` | Examples GEPA reflects on (maximize this later) |
| `VAL_DEMOS` | **Disjoint** set for Pareto selection |
| `gepa_metric` | Returns `dspy.Prediction(score=..., feedback=...)` |
| `reflection_lm` | Stronger LM @ `temperature=1.0` that rewrites instructions from feedback |

A float-only metric makes GEPA little better than random search. Feedback must say **what failed** and **what good looks like** (enums, `move_to(iron)` not `move_to(pos=...)`, no `nearest("iron-ore")`).

## Files

| Path | Role |
|------|------|
| `examples/07_gepa_train.py` | Optimize / compile / save |
| `examples/08_gepa_run.py` | Load saved program → Factorio |
| `factorio_gym/trainset.py` | `TRAIN_DEMOS` / `VAL_DEMOS` |
| `factorio_gym/agent.py` | Shared signature + `load_agent` |
| `.fle/gepa_logs/` | GEPA checkpoints (`log_dir`) |

## Metric contract (copy this pattern)

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = ...       # 0.0 .. 1.0
    feedback = ...    # specific natural-language critique
    return dspy.Prediction(score=score, feedback=feedback)  # not a dict
```

The demo metric is **offline heuristics** (enums, tools, `print`). Next upgrades:

1. AST-parse the program  
2. Execute one step in FLE and score `raw_text` / reward  
3. Per-predictor feedback once you add Planner → Coder modules  

## Budget knobs

Use **either** `auto=` **or** an explicit budget — not both.

| Mode | When |
|------|------|
| `--auto light` | First runs / metric debugging (default) |
| `--auto medium` | Everyday optimization |
| `--auto heavy` | Final pass before you trust the artifact |

Defaults in `07`:

- Task LM: `openai/gpt-4o-mini`  
- Reflection LM: `openai/gpt-4o` @ `temperature=1.0`  
- `log_dir=.fle/gepa_logs` (resume by reusing the same dir)  
- Save path: `.fle/gepa_factorio_agent.json`  

## Next steps

1. Grow `TRAIN_DEMOS`; keep `VAL_DEMOS` held out  
2. Replace heuristic `gepa_metric` with an execution metric (still return `score` + `feedback`)  
3. Try `auto="medium"` only after `light` looks sane  
4. Optional: chain Bootstrap → GEPA with `dspy.BetterTogether`  
5. Multi-module agents: GEPA’s per-predictor feedback shines once you split Planner/Coder  

Do **not** call `dspy.GEPA(...).compile()` inside `08_gepa_run.py`.
