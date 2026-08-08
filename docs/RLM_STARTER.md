# `dspy.RLM` starter (Factorio Gym)

Step-by-step tutorial: use DSPy's **Recursive Language Model** to place and fuel a burner mining drill on iron ore — the same simplest milestone as `examples/03_scripted_miner.py`, but driven by an LLM that explores via a Python REPL.

Sibling path (same goal, outer loop instead of REPL): [FLEX_STARTER.md](FLEX_STARTER.md) / [`examples/12_dspy_flex_miner.py`](../examples/12_dspy_flex_miner.py).

Runnable script: [`examples/11_dspy_rlm_miner.py`](../examples/11_dspy_rlm_miner.py).

## 0. Prerequisites

1. Same as [HELLO_WORLD.md](HELLO_WORLD.md): Docker healthy, `.env` with `OPENAI_API_KEY`, cluster up:

```bash
uv run fle cluster start -n 1
```

2. **Deno** (required by DSPy's default Pyodide/WASM interpreter):

```bash
brew install deno   # or https://deno.land
which deno
```

3. Optional: stop other scripts that hold the Factorio instance (e.g. `10_live_client_watch.py`) so this example can `reset` cleanly.

## 1. Mental model: RLM vs example 04 (and Flex)

| | `04_dspy_agent_loop` | `11_dspy_rlm_miner` | `12` / `13b` Flex |
|--|----------------------|---------------------|---------------------|
| Goal | Broader / quota-oriented | Place + fuel one drill | **Same as 11** |
| Loop owner | **Your** Python `for step` | **RLM** REPL iterations | **Your** Python `for step` |
| Each LLM turn | Emits one Factorio program | Sandbox Python that may call tools | One FLE program (via Flex) |
| Factorio access | You call `step_code` | Tool `run_factorio(code)` | You call `step_code` |
| When it stops | `--steps N` or env done | `SUBMIT(...)` or `max_iters` | Fueled-drill heuristic / `--steps` |
| Learning | — | — | `13a` learns `module_src` from online play |

```text
04 / 12 / 13b:  LM ──program──► Factorio ──obs──► LM ──program──► …

11:  LM ──sandbox Python──► Deno REPL
              │
              └── run_factorio(code) ──► Factorio ──text──► REPL ──► LM …
              │
              └── SUBMIT(summary=…, success=…)
```

RLM shines when the agent should **probe, fix errors, and decide the next probe in code**. Use `04` / Flex when you already want a fixed outer step budget. Flex intro is `12`; learn-from-play is `13a` → `13b` ([FLEX_STARTER.md](FLEX_STARTER.md)).

Official module docs: [dspy.RLM](https://dspy.ai/api/modules/RLM/).

## 2. The simplest task

Full `iron_ore_throughput` wants 16 ore / 60s. This tutorial narrows the goal to:

1. Find iron (`nearest(Resource.IronOre)`)
2. `move_to` if far
3. Place `Prototype.BurnerMiningDrill`
4. `insert_item(Prototype.Coal, drill, quantity=5)`
5. Print proof (`get_entities` / inventory)

That matches the scripted baseline in `03` — enough to see RLM work without chasing the quota.

## 3. Signature + tool

Signature = what RLM must eventually `SUBMIT`:

```python
class PlaceBurnerDrill(dspy.Signature):
    """…iterate with run_factorio, then SUBMIT…"""

    goal: str = dspy.InputField()
    api_docs: str = dspy.InputField()
    summary: str = dspy.OutputField()
    success: bool = dspy.OutputField()
```

Bridge to the game — a normal Python function; DSPy exposes it inside the sandbox:

```python
def run_factorio(code: str) -> str:
    """Execute one FLE Python program; return stdout/stderr."""
    obs, reward, terminated, truncated, _ = step_code(env, code)
    return obs_text(obs) or "(empty)"
```

Inside the RLM REPL the model writes things like:

```python
print(api_docs[:400])
out = run_factorio("iron = nearest(Resource.IronOre)\\nprint(iron); move_to(iron)")
print(out)
# … more steps …
SUBMIT(summary="Placed and fueled drill at …", success=True)
```

Built-ins also available: `llm_query(prompt)`, `llm_query_batched(prompts)`, `print`.

**Critical FLE detail:** each `run_factorio(code)` is a **fresh** Factorio program. Bindings like `drill = place_entity(...)` do **not** survive into the next tool call. Re-fetch with `nearest(...)` / `get_entities()`, or place+fuel in the **same** program after `move_to`.

**Two sandboxes:** the outer RLM REPL only has `run_factorio` / `print` / `SUBMIT`. FLE names (`nearest`, `Resource`, …) exist **only inside the string** passed to `run_factorio`. Strong models often call `nearest(...)` at top level → `NameError`, then burn `llm_query` inventing fake `game.` / `def run(game)` APIs. Trust `api_docs`; keep `--max-llm-calls` low.

## 4. Wire it up

```python
import dspy
from factorio_gym.env import make_env, reset_env, step_code, obs_text

env = make_env("iron_ore_throughput", run_idx=0)
reset_env(env)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", max_tokens=4000))

rlm = dspy.RLM(
    PlaceBurnerDrill,
    max_iters=12,
    max_llm_calls=5,
    verbose=True,          # watch the REPL
    tools=[run_factorio],
)

result = rlm(goal=goal, api_docs=API_DOCS)
print(result.success, result.summary)
env.close()
```

In DSPy **3.3.x** the constructor uses `max_iters` and `interpreter_factory` (not the older `max_iterations` / `interpreter` names in some docs).

## 5. Run the example

```bash
# Quiet success/failure summary
uv run python examples/11_dspy_rlm_miner.py

# See every REPL thought / code / stdout (best for learning)
uv run python examples/11_dspy_rlm_miner.py --verbose --max-iters 12

# Map PNGs after each Factorio tool call
uv run python examples/11_dspy_rlm_miner.py --renders
open .fle/renders/rlm_miner/step_01.png
```

Useful knobs:

| Flag | Role |
|------|------|
| `--verbose` | Print RLM trajectory |
| `--max-iters` | Cap REPL rounds (default 12) |
| `--max-llm-calls` | Cap `llm_query*` budget (default 5; raise only if needed) |
| `--model` | Outer orchestrator LM |
| `--sub-model` | Cheaper LM for `llm_query` (optional) |
| `--renders` | Schematic PNGs under `.fle/renders/rlm_miner/` |

## 6. What good looks like

Console should show several `=== run_factorio #N ===` blocks, then:

```text
=== RLM result ===
success=True
summary=… drill … coal …
factorio_steps=3   # often 3–6 for this tiny task
```

If `success=False`, re-run with `--verbose` and check whether the model:

- used `nearest("iron-ore")` instead of `Resource.IronOre`
- called `move_to(pos=…)` (invalid) instead of `move_to(iron)`
- tried to solve the whole factory in one giant program

## 7. Pitfalls

| Symptom | Fix |
|---------|-----|
| `NameError: nearest is not defined` in RLM (before Factorio) | Model ran FLE code in the outer REPL — wrap it in `run_factorio('''...''')` |
| Invents `from fle.env import *` / `game.nearest` / `def run(game)` | Ignore `llm_query` for API discovery; lower `--max-llm-calls`; trust `api_docs` |
| `deno: command not found` | Install Deno; restart the shell |
| `drill` is `None` / `NoneType` on insert | Variables do not persist across `run_factorio` — re-fetch or place+fuel in one program |
| Place fails: entity already exists | `get_entities()` + fuel existing drill; do not place again |
| RLM hangs / empty REPL | Raise `--max-iters`; use `--verbose` |
| `LLM call limit exceeded` | Raise `--max-llm-calls` or stop using `llm_query` for simple string checks |
| Tool returns huge text | Truncate in `run_factorio` (example already caps ~4k chars) |
| Competing with live client | Stop `10_live_client_watch.py` first |
| Expecting full quota reward | This tutorial only places+fuels one drill — short runs may still score `0.0`–few ore |

Security note: default interpreter is sandboxed WASM. Your `run_factorio` tool intentionally reaches the host Factorio process — treat tool surface as trusted code you wrote.

## 8. How this fits the learning path

```text
01 hello world  →  03 scripted miner  →  04 Predict loop
                              ↘
                         ┌──── 11 RLM (this doc)  ← REPL + tools
                         │         ↕ same drill goal
                         └──── 12 Flex intro → 13a play-train → 13b run
                              ↗
07/08 GEPA on Predict  →  instruction search (hand-written demos)
```

Next ideas once this works:

1. Widen the goal toward real throughput (chest + belts + `sleep`)
2. Add tools like `inspect_last_entities()` that return structured JSON
3. Compare cost/latency vs Flex (`12` / `13b`) on the same drill milestone
4. Wrap RLM in a `dspy.Module` and optimize with GEPA ([GEPA_STARTER.md](GEPA_STARTER.md))
