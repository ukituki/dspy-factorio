# Tasks guide (with `iron_ore_throughput`)

This doc introduces **how FLE tasks work** using the easiest registered environment. Read it after [HELLO_WORLD.md](HELLO_WORLD.md) if the gym / reward / “what am I optimizing?” loop still feels fuzzy.

You do **not** need an LLM here — concepts apply to scripted play and agents alike.

---

## 1. One-sentence picture

A **task** is a Factorio world plus a **goal**, a **starting kit**, and a **scoring rule**. Your agent does not press WASD; each turn it submits a short **Python program**. The game runs that program and replies with text (and structured state).

```text
goal + inventory + map
        │
        ▼
   reset() ──► write Python ──► step() ──► read stdout / reward
                    ▲                            │
                    └──────── next program ◄─────┘
```

---

## 2. Why start with `iron_ore_throughput`

| Property | Value |
|----------|--------|
| Env / task id | `iron_ore_throughput` |
| Goal | Automate **16 iron ore per 60 in-game seconds** |
| Why easiest | Only mining — no smelting, assemblers, or oil |
| Agents | 1 |
| Typical horizon | 64 programs (lab-play default) |

List everything available:

```bash
uv run python examples/02_list_environments.py
uv run python examples/02_list_environments.py --search iron
```

Harder tasks reuse the **same interaction pattern**; they mainly deepen the **recipe tree** (plates → circuits → science).

---

## 3. What’s inside the task (metadata that matters)

When you `make_env("iron_ore_throughput")`, FLE builds:

1. A live Factorio instance (Docker / RCON)
2. A **task object** that configures that instance
3. A gym wrapper (`FactorioGymEnv`) that turns programs into `step()` calls

### Goal text

You get a natural-language description, roughly:

> Create an automatic iron-ore factory that produces **16 iron-ore per 60 in-game seconds.**

Plus shared lab-play instructions: build an **automatic** factory (machines keep working without you hand-mining every tick), and after each step the env measures throughput.

In code:

```python
from dspy_factorio.env import get_environment_info, describe_env

print(describe_env("iron_ore_throughput"))
info = get_environment_info("iron_ore_throughput")
# info keys: description, task_key, num_agents, enable_vision, …
```

### Starting kit (you are not empty-handed)

Lab-play tasks start with a **populated inventory** (drills, belts, inserters, coal, poles, …) and **all technologies researched**. So early friction is “place and fuel correctly,” not “grind for a pickaxe.”

### Scoring rule (this is the part people miss)

After each program, the task roughly:

1. Lets the factory run for **~60 in-game seconds** (`sleep`-style holdout)
2. Measures how much **iron-ore** was produced automatically in that window
3. Uses that rate as **reward** (throughput override)
4. Marks the episode **done** when throughput **≥ 16**

So:

- A Hello World that only prints `nearest(Resource.IronOre)` often scores **`reward=0.0`** — correct, nothing is mining yet.
- Placing an **unfueled** drill still yields ~0.
- Fuel + (usually) a place for ore to go (chest / belt) is when reward starts moving.

---

## 4. Action = one Python program

Each `step` sends an `Action`:

| Field | Meaning |
|-------|---------|
| `code` | Executable FLE Python (the whole “move”) |
| `agent_idx` | Which character (always `0` for this task) |
| `game_state` | Optional restore point; helpers usually pass current state |

This repo wraps that for you:

```python
from dspy_factorio.env import make_env, reset_env, step_code, obs_text

env = make_env("iron_ore_throughput", run_idx=0)
reset_env(env)

obs, reward, terminated, truncated, info = step_code(
    env,
    """
iron = nearest(Resource.IronOre)
print(iron)
""".strip(),
)
print(obs_text(obs))
print(reward, terminated)
env.close()
```

**Policy that works:** one small, printable step per program (find ore → move → place drill → fuel → check entities). Giant one-shot factories are hard to debug when `raw_text` shows an exception.

### API footguns (same on every task)

```python
# good
nearest(Resource.IronOre)
move_to(iron)   # or move_to(position=iron)
place_entity(entity=Prototype.BurnerMiningDrill, position=iron, direction=Direction.NORTH)
insert_item(Prototype.Coal, drill, quantity=5)

# bad — crashes inside FLE
nearest("iron-ore")
move_to(pos=iron)          # unexpected keyword 'pos'
place_entity("mining-drill", ...)
```

If the ore patch is far, **`move_to` before `place_entity`** (place range is short).

---

## 5. Observation = what you read back

The gym returns a **dict**. For learning and LLM loops you mostly care about:

| Field | Friendly meaning |
|-------|------------------|
| `raw_text` | Stdout / stderr of your last program (+ throughput hint the task appends) |
| `task_info` | Goal text, task key, trajectory length |
| `inventory` | What you’re carrying |
| `entities` | Machines already on the map |
| `flows` | Production rates |
| `score` / reward | How the scorer valued this step |
| `task_verification` | Did we hit the quota yet? |

Helpers:

```python
obs_text(obs)   # prefers raw_text — what agents usually “see”
```

`info` also has useful extras: `error_occurred`, `output_game_state` (for resume), `achievements`, production scores.

**Mental model:** treat exceptions in `raw_text` as normal observations — fix that error next step; don’t restart the whole cluster unless RCON is dead.

---

## 6. Guided walkthrough (scripted, no LLM)

Run the packaged miner (same task):

```bash
uv run python examples/03_scripted_miner.py
```

What it does, step by step:

| Step | Program intent | What you should learn |
|------|----------------|------------------------|
| 1 | `nearest` + `inspect_inventory` | Observation channel is text; inventory is already stocked |
| 2 | `move_to` + `place_entity(...BurnerMiningDrill...)` | Placement needs enums + being in range |
| 3 | `get_entities` + `insert_item(...Coal...)` | Machines need fuel before throughput > 0 |

Then optionally watch or dump maps:

```bash
uv run python examples/09_visualize_renders.py   # PNG maps
uv run python examples/10_live_client_watch.py   # Factorio client
```

Docs: [VISUALIZATION.md](VISUALIZATION.md).

---

## 7. How an LLM agent uses the same task

Same env, different program source:

```text
goal string (from env description)
        +
last raw_text
        │
        ▼
  DSPy / your model  →  Python program  →  step_code()
```

Intro loop:

```bash
uv run python examples/04_dspy_agent_loop.py --env-id iron_ore_throughput --steps 5
```

The model should **not** invent a new API — it must emit FLE Python that could have been copy-pasted into `03_scripted_miner.py`.

---

## 8. “Am I done?” checklist for this task

You’re succeeding on `iron_ore_throughput` when:

1. At least one **fueled** miner sits on iron ore  
2. Ore has somewhere to go (chest at drop position and/or belts — otherwise drills jam)  
3. After the holdout window, measured rate is **≥ 16 ore / 60s**  
4. `terminated` becomes true (or inspect-eval reports success)

Until then, short runs with `reward=0` are expected — lengthen the trajectory and finish the automation loop.

---

## 9. How other tasks reuse this shape

| Same for every lab-play task | What changes |
|------------------------------|--------------|
| `make_env` / `reset` / `step(code)` | `throughput_entity` (plates, circuits, science, …) |
| Populated inventory + all tech | Recipe depth / fluid plumbing |
| ~60s throughput measurement | Quota target item |
| Text-first observations | Multi-agent adds `send_message()` + per-agent instructions |

So once `iron_ore_throughput` feels boring, bump difficulty by id only:

```bash
uv run python examples/01_hello_world.py --env-id iron_plate_throughput
```

---

## 10. Friction cheat sheet

| Symptom | Likely cause |
|---------|----------------|
| `missing run_idx` | Use `make_env(...)` or `gym.make(..., run_idx=0)` |
| Hang on connect | Cluster not ready / Docker discovery — see [SETUP.md](SETUP.md) |
| `AttributeError` / weird tool errors | String names instead of `Resource.*` / `Prototype.*` |
| `TypeError: unexpected keyword argument 'pos'` | Use `move_to(iron)` not `move_to(pos=...)` |
| Always `reward=0` | No automated mining yet, or no fuel / output buffer |
| Program “failed” but process OK | Read `raw_text` — Lua/Python errors are observations |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## 11. Where to go next

1. Design your own loop — [SCENARIOS.md](SCENARIOS.md)  
2. Optimize prompts / modules — [AI_OPTIMIZATION.md](AI_OPTIMIZATION.md), [GEPA_STARTER.md](GEPA_STARTER.md)  
3. REPL-style agents — [RLM_STARTER.md](RLM_STARTER.md), [FLEX_STARTER.md](FLEX_STARTER.md)

**Bottom line:** `iron_ore_throughput` is the smallest complete instance of the FLE contract — **goal, kit, program-in / text-out, throughput score**. Master that loop once; every harder env is the same loop with a deeper factory.

---

## Interactive notebook

Explore the same concepts live (metadata → connect → run programs):

```bash
uv run fle cluster start -n 1   # for live cells
uv run marimo edit scenarios/1_iron_ore_throughput/01_explore_env.py
```

Use the project env (`uv run …`). If marimo offers an isolated sandbox, choose **n** — this repo is not on PyPI.

Metadata cells work without Factorio; use **Connect & reset** before running programs.
