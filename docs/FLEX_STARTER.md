# `dspy.Flex` starter (DSPy Factorio)

Step-by-step tutorial: use DSPy's **Flex** module to place and fuel a burner mining drill on iron ore — the **same milestone** as [`RLM_STARTER.md`](RLM_STARTER.md) / `examples/03_scripted_miner.py`.

| Stage | Script | Role |
|-------|--------|------|
| **Intro** | [`examples/12_dspy_flex_miner.py`](../examples/12_dspy_flex_miner.py) | Baseline Flex, live play — learn the loop |
| **Train (13a)** | [`examples/13a_dspy_flex_train.py`](../examples/13a_dspy_flex_train.py) | Online play → demos → GEPA rewrites `module_src` |
| **Run (13b)** | [`examples/13b_dspy_flex_run.py`](../examples/13b_dspy_flex_run.py) | Load the learned Flex → same drill goal |

Sibling path (REPL owns the loop, no train/run split): [RLM_STARTER.md](RLM_STARTER.md) / example `11`.

Official docs: [Flex deep-dive](https://dspy.ai/diving-deeper/flex/) · [API](https://dspy.ai/api/modules/Flex/). Requires **DSPy ≥ 3.3.0** (`Flex` is `@experimental`).

## 0. Prerequisites

1. Same as [HELLO_WORLD.md](HELLO_WORLD.md): Docker healthy, `.env` with `OPENAI_API_KEY`, cluster up:

```bash
uv run fle cluster start -n 1
```

2. **Deno** (Flex + GEPA-authored code use the default Pyodide/WASM interpreter):

```bash
brew install deno   # or https://deno.land
which deno
```

3. Optional: stop other scripts that hold the Factorio instance (e.g. `10_live_client_watch.py`).

## 1. Mental model: intro vs learn-from-play

```text
12 intro:   baseline Flex ──program──► Factorio ──obs──► …   (no compile)

13a train:  baseline Flex ──play──► healthy demos ──GEPA──► module_src artifact
                 ▲                         │
                 └── Factorio online ───────┘

13b run:    load Flex(module_src) ──program──► Factorio ──obs──► …
```

| | `12` intro | `13a` → `13b` |
|--|------------|----------------|
| Goal | Place + fuel one burner drill | **Same** |
| Flex state | Baseline (`Predict` wrapper) | GEPA-rewritten `module_src` |
| Where demos come from | — | **Online Factorio play** |
| GEPA valset | — | Held-out `FLEX_VAL_DEMOS` |

Contrast with RLM (`11`): one REPL call + `run_factorio`. Flex keeps **your** outer step loop; training learns a better *proposer*, not a REPL policy.

Why not `tools=[run_factorio]` on Flex? With tools the baseline becomes an RLM — use example `11` for that path.

## 2. The simplest task

Full `iron_ore_throughput` wants 16 ore / 60s. This tutorial narrows the goal to:

1. Find iron (`nearest(Resource.IronOre)`)
2. `move_to` if far
3. Place `Prototype.BurnerMiningDrill`
4. `insert_item(Prototype.Coal, drill, quantity=5)`
5. Print proof (`get_entities` / inventory)

## 3. Intro (`12`): baseline Flex

```python
import dspy
from dspy_factorio.agent import FactorioProgrammer

flex = dspy.Flex(FactorioProgrammer)
print(flex.module_src)   # thin Module wrapping one Predict
```

```bash
uv run python examples/12_dspy_flex_miner.py
uv run python examples/12_dspy_flex_miner.py --verbose --steps 6
uv run python examples/12_dspy_flex_miner.py --renders
```

Uncompiled Flex ≈ `Predict` — expected for the intro. You’re learning the outer loop + `module_src` surface before compile.

## 4. Train (`13a`): demos from online play → tiny GEPA

Realistic Flex upside: demos are not only hand-written — they’re **steps that worked in Factorio**.

```text
for episode:
  reset + bootstrap
  for step:
    program = baseline_flex(goal, obs, hint)
    obs' = step_code(env, program)
    if healthy(obs'): keep Example(obs → program)
GEPA(Flex, max_metric_calls=24).compile(trainset=play_demos, valset=FLEX_VAL_DEMOS)
save .fle/flex_from_play.json
```

**Budget (important):** `auto=light` with Flex can mean **300+ metric calls** and still skip mutations when scores look perfect. The tutorial default is an explicit tiny cap:

| Flag | Role |
|------|------|
| *(default)* `--max-metric-calls 24` | Show the loop without burning the wallet |
| `--max-metric-calls 48` | Slightly more room for a code rewrite |
| `--auto light` | Full GEPA light budget — **expensive** for Flex; only when you mean it |

```bash
# Plan only
uv run python examples/13a_dspy_flex_train.py --dry-run

# Play + tiny GEPA (default 24 metric calls)
uv run python examples/13a_dspy_flex_train.py

# More play if you need more demos
uv run python examples/13a_dspy_flex_train.py --episodes 3 --steps 8 --max-metric-calls 24
```

What you should see as “learning” without a huge run:

1. **Play phase** — demos land in `.fle/flex_play_demos.json` (inspect these first)  
2. **Baseline vs after** — `13a` prints `module_src` before/after GEPA  
3. **Stricter drill valset** — `FLEX_VAL_DEMOS` (fuel-after-place / move_to-when-far) so the metric has headroom  

Artifacts:

| Path | What |
|------|------|
| `.fle/flex_play_demos.json` | Healthy (obs → program) rows from play |
| `.fle/flex_from_play.json` | Compiled Flex (`module_src` + state) |
| `.fle/flex_from_play.meta.json` | Budget, episodes, whether `module_src` changed |
| `.fle/flex_play_gepa_logs/` | GEPA checkpoints |

If `module_src` is unchanged after 24 calls, that’s OK for the tutorial — you still learned a **play dataset**. Raise `--max-metric-calls` only once demos look right.
## 5. Run (`13b`): load the learned program

```bash
uv run python examples/13b_dspy_flex_run.py --steps 6
uv run python examples/13b_dspy_flex_run.py --program .fle/flex_from_play.json --verbose
uv run python examples/13b_dspy_flex_run.py --renders
```

Load with `dspy.Flex(FactorioProgrammer).load(...)` — not `load_agent` from example `08` (that expects a bare `Predict`).

## 6. What good looks like

**Intro (`12`)** — a few steps, then:

```text
=== Flex intro result ===
success=True
Next: uv run python examples/13a_dspy_flex_train.py
```

**Train (`13a`)** — play blocks (through fuel, not place-only), then a short GEPA:

```text
Collected N play demos → .fle/flex_play_demos.json
GEPA compile … | max_metric_calls=24
=== baseline module_src (head) ===
…
=== module_src after GEPA (head) ===
…
Saved Flex → .fle/flex_from_play.json
```

Inspect demos / source:

```bash
python -c "import json; d=json.load(open('.fle/flex_play_demos.json')); print(len(d)); print(d[-1]['program'])"
python -c "import json; print(json.load(open('.fle/flex_from_play.json'))['module_src'][:800])"
```

**Run (`13b`)** — same drill success heuristic as intro (requires `insert_item` evidence), using the learned `module_src`.

## 7. Pitfalls

| Symptom | Fix |
|---------|-----|
| `deno: command not found` | Install Deno; restart the shell |
| GEPA plans 300+ metric calls | Don’t use `--auto light` for Flex tutorials — default is `--max-metric-calls 24` |
| Early stop after place-only (no coal) | Heuristic needs `insert_item` / entity status — not a “Placed Burner…” print |
| `13a`: fewer than `--min-demos` healthy steps | More `--episodes` / `--steps`, or stronger `--model` |
| Play keeps failing with `move_to(pos=...)` | Fix API via `API_HINT`; unhealthy steps are dropped on purpose |
| Loading Flex JSON with `load_agent` | Use `13b` / `build_flex_agent(..., path)` |
| Expecting full quota reward | Tutorial only places+fuels one drill |
| Competing with live client | Stop `10_live_client_watch.py` first |
| Putting `tools=[run_factorio]` on Flex | Baseline becomes RLM → use example `11` |
| API churn | `@experimental` since DSPy 3.3.0 — pin `dspy` if you depend on serialization |

Security note: Flex runs `module_src` in a sandbox by default. These scripts pass no host tools; only bridged predictor calls reach the host LM.

## 8. How this fits the learning path

```text
01 hello world  →  03 scripted miner  →  04 Predict loop
                              ↘
                         ┌──── 11 RLM              ← REPL + tools, same drill goal
                         │
                         └──── 12 Flex intro       ← baseline outer loop
                                → 13a train (play → GEPA)
                                → 13b run  (learned module_src)
07/08 GEPA on Predict  →  instruction search on hand-written demos
```

Next ideas:

1. A/B `12` vs `13b` vs `11` on cost and steps-to-success  
2. Raise `--auto` only after play demos look sane  
3. Add an execution metric (step Factorio inside GEPA) once the offline heuristic is trusted  
4. Widen the goal toward chest + belts + throughput  

Sibling tutorial: [RLM_STARTER.md](RLM_STARTER.md). Hand-written-demo GEPA: [GEPA_STARTER.md](GEPA_STARTER.md).
