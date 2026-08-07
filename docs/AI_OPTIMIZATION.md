# AI engine optimization

This repo uses **DSPy** as the AI layer on top of FLE's code-as-action interface.

## Architecture

```text
┌──────────────────┐     program (Python)     ┌────────────────────┐
│  DSPy module     │ ───────────────────────► │  FLE gym env       │
│  FactorioProgrammer │                         │  Factorio + tools  │
└────────▲─────────┘     raw_text / reward    └─────────┬──────────┘
         │                                              │
         └──────────────── observation ─────────────────┘
```

Core pieces:

- `factorio_gym/agent.py` — `FactorioProgrammer` signature, `SEED_DEMOS`, `API_HINT`, helpers
- `examples/04_dspy_agent_loop.py` — online REPL loop (bootstrap + few-shot agent)
- `examples/05_optimize_agent.py` — offline prompt optimization over the same demos

## Why the agent needs demos

Unprompted LLMs invent string APIs (`nearest("iron-ore")`) that crash in FLE. This repo fixes that by:

1. Hard API rules in the `FactorioProgrammer` docstring (`Resource.*` / `Prototype.*` / `move_to`)
2. `LabeledFewShot` over `SEED_DEMOS` in `build_agent()`
3. A bootstrap REPL step that prints inventory + nearest iron before the LLM acts
4. Passing `API_HINT` every step

Without those, `04` loops the same AttributeError forever.

## What to optimize

1. **Prompt / demos** (cheapest) — expand `SEED_DEMOS` / BootstrapFewShot / GEPA  
2. **Model choice** — mini for iteration, stronger models for hard throughput tasks  
3. **Trajectory policy** — step budget, early stop, observation trimming  
4. **Tool curriculum** — start with mining → smelting → belts → science  

## Offline optimization (no Factorio)

```bash
uv run python examples/05_optimize_agent.py \
  --model openai/gpt-4o-mini \
  --save .fle/optimized_factorio_agent.json
```

How it works:

1. Reuses `SEED_DEMOS` from `factorio_gym.agent` as the trainset  
2. `BootstrapFewShot` asks the LM to produce programs and keeps ones that pass `program_metric`  
3. Saves a compiled DSPy module you can `--load` in example 04  

**Metric.** The starter metric rewards `print` + FLE tools + `Resource.`/`Prototype.` enums, and penalizes stringly `nearest("...")`. Better metrics:

- Unit-test style: AST parse success
- Execution success rate on a frozen Factorio seed (online metric)
- Quota progress / production score after N steps

## Online loop

```bash
uv run python examples/04_dspy_agent_loop.py --steps 8 --model openai/gpt-4o-mini
```

Flow:

1. `reset` → bootstrap (`inspect_inventory` + `nearest(Resource.IronOre)`)
2. Each step: `propose_program(goal, observation, API_HINT)` → `step_code`
3. Observation text (including exceptions) feeds the next step

Optimization ideas while online:

| Idea | Why |
|------|-----|
| Truncate `raw_text` to last 2–4k chars | Avoid context blowups |
| Force a JSON plan then code | Separates reasoning from syntax |
| Retry once on parse/tool errors | Cheap robustness |
| Cache successful subroutines | Drill placement, fuel insert, belt lines |
| Expand `SEED_DEMOS` with chest/belt steps | Stops re-placing drills on occupied tiles |

## Loading an optimized module

```bash
uv run python examples/04_dspy_agent_loop.py \
  --load .fle/optimized_factorio_agent.json \
  --steps 8
```

Or in code:

```python
import dspy
from factorio_gym.agent import FactorioProgrammer, build_lm, AgentConfig

dspy.configure(lm=build_lm(AgentConfig()))
agent = dspy.Predict(FactorioProgrammer)
agent.load(".fle/optimized_factorio_agent.json")
```

## inspect-eval vs custom loop

| | Custom DSPy loop | `fle inspect-eval` |
|--|------------------|--------------------|
| Flexibility | High (your metrics, logging) | Medium |
| Comparability | Low | High (harness + scorers) |
| Container needs | 1 | 1 per epoch if Pass@N |
| Best for | Prompt/engine R&D | Benchmark numbers |

Recommended: iterate in examples 03→05→04, then measure with example 06.

## Cost / latency knobs

- `--trajectory-length` / `--steps` dominate cost  
- Prefer `gpt-4o-mini` while debugging tool use  
- Pin `--epochs 1` until you run a multi-instance cluster (`fle cluster start -n 8`)  
- Set `temperature=0.2` (default in `AgentConfig`) for more stable code  

## Next upgrades

1. Replace `program_metric` with an execution metric that runs programs in FLE  
2. Add GEPA (`dspy.GEPA`) once you have a reliable metric  
3. Store trajectories as DSPy examples automatically from `04_dspy_agent_loop.py`  
4. Multi-module pipeline: Planner → Coder → Critic  
