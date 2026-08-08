# Visualizing game state

You need either a **Factorio desktop client** (live view) or **PNG map dumps**
(no game install). Pick one:

| You have… | Use |
|-----------|-----|
| No Factorio desktop | **§1 PNG renders** (works now) |
| Factorio ≈ 2.0.73 | **§2 Live client** |

Official background:
[Sprites](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/advanced/sprites.html),
[Client-side setup](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/getting_started/installation.html)
(upstream still mentions client `1.1.110` — this workspace’s server is **2.0.73**).

---

## 1) PNG renders (no Factorio desktop)

FLE draws a schematic map from entity state after each step. No Steam/client
purchase required.

### Run

```bash
uv run fle cluster start -n 1   # if needed; wait for RCON auth = 1
uv run python examples/09_visualize_renders.py
open .fle/renders/simple/step_00_reset.png
open .fle/renders/simple/step_02.png   # drill should appear here
```

What you get under `.fle/renders/simple/`:

```text
step_00_reset.png
step_01.png          # found iron ore
step_02.png          # burner mining drill placed
step_03.png          # fuelled / inventory update
```

| Mode | Flag | Needs |
|------|------|--------|
| Schematic grid (default) | `--mode simple` | Nothing extra |
| Factorio-like pixels | `--mode sprites` | `uv run fle sprites` (+ `HF_TOKEN` if HuggingFace rate-limits) |

```bash
uv run python examples/09_visualize_renders.py --mode simple --out .fle/renders/run1
```

### From your own scripts

```python
from dspy_factorio.env import make_env, reset_env, step_code, save_render

env = make_env("iron_ore_throughput", run_idx=0)
reset_env(env)
step_code(env, "move_to(nearest(Resource.IronOre))")
save_render(env, ".fle/renders/after_move.png", mode="simple")
env.close()
```

Hook `save_render(env, path)` after any `step_code` / agent step to build a
frame sequence.

### DSPy agent loop

`examples/04_dspy_agent_loop.py` can dump a PNG after reset, bootstrap, and
each LLM step:

```bash
uv run fle cluster start -n 1          # if needed
# needs OPENAI_API_KEY in .env
uv run python examples/04_dspy_agent_loop.py --steps 3 --renders
open .fle/renders/dspy_agent/step_00_reset.png
open .fle/renders/dspy_agent/step_03.png
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--renders` | off | Enable map PNGs |
| `--render-dir` | `.fle/renders/dspy_agent` | Output folder |
| `--render-mode` | `simple` | `simple` or `sprites` |
| `--render-zoom` | `0.25` | Zoom out for `simple`. Still player-limited unless overview centering is on |
| *(auto)* | overview on | `simple` renders center between **player and nearest iron** so both stay in frame (~44-tile radius cap) |

The schematic renderer is **centered** (player, or overview midpoint) with a hard
radius cap (~44 tiles). Default zoom alone is not enough when iron is at
`y≈70` and the player is at spawn — hence overview centering.

`move_to` must receive a `Position` — never `pos=`:

```python
# correct
iron = nearest(Resource.IronOre)
move_to(iron)
# or: move_to(position=iron)

# wrong — TypeError: unexpected keyword argument 'pos'
move_to(pos=(15.5, 70.5))
```

`API_HINT` / the DSPy signature encode this; GEPA’s metric also penalizes `move_to(pos=`.

Typical layout after `--steps 3 --renders`:

```text
.fle/renders/dspy_agent/
  step_00_reset.png
  step_01_bootstrap.png   # inspect_inventory + nearest iron
  step_02.png             # after agent step 1
  step_03.png             # after agent step 2
  step_04.png             # after agent step 3
```

Same pattern in your own DSPy loop — call `save_render` **after** `step_code`:

```python
from pathlib import Path
from dspy_factorio.agent import API_HINT, AgentConfig, build_agent, propose_program
from dspy_factorio.env import make_env, reset_env, step_code, obs_text, save_render

out = Path(".fle/renders/dspy_agent")
agent = build_agent(AgentConfig(model="openai/gpt-4o-mini"))
env = make_env("iron_ore_throughput", run_idx=0)
reset_env(env)
save_render(env, out / "step_00_reset.png", mode="simple", zoom=0.25, overview=True)

obs, *_ = step_code(env, "print(inspect_inventory())")
observation = obs_text(obs)
save_render(env, out / "step_01_bootstrap.png", mode="simple", zoom=0.25, overview=True)

for i in range(1, 4):
    program = propose_program(
        agent,
        goal="Create an automatic iron-ore factory…",
        observation=observation,
        inventory_hint=API_HINT,
    )
    obs, reward, terminated, truncated, _ = step_code(env, program)
    observation = obs_text(obs)
    save_render(
        env, out / f"step_{i + 1:02d}.png", mode="simple", zoom=0.25, overview=True
    )
    if terminated or truncated:
        break
env.close()
```

GEPA online runs (`examples/08_gepa_run.py`) use the same `step_code` helper —
add `save_render` the same way after each step if you want frames there too.

### Optional: Factorio-like sprites later

```bash
# Needs network; unauthenticated HF downloads often hit 429 — set HF_TOKEN
uv run fle sprites
uv run python examples/09_visualize_renders.py --mode sprites
open .fle/renders/sprites/step_02.png
```

---

## 2) Live Factorio client (requires buying/installing the game)

Install Factorio ≈ **2.0.73** from [factorio.com](https://www.factorio.com/) or
Steam, then use this path. The Docker server is headless; the client is only a
spectator.

### Ports (instance 0)

| Traffic | Host |
|---------|------|
| Game (UDP) | **`34197`** → connect `127.0.0.1:34197` |
| RCON (TCP) | **`27000`** (agents; password `factorio`) |

### First-time flow

```bash
uv run fle cluster start -n 1
# wait until:
uv run python -c "from factorio_rcon import RCONClient; print(RCONClient('127.0.0.1',27000,'factorio').send_command('/sc rcon.print(1)'))"
# expect: 1

uv run python examples/10_live_client_watch.py --prepare-only
```

Then in Factorio: **Multiplayer → Connect to address → `127.0.0.1:34197`**
(blank password; sync/disable Space Age DLC if prompted).

Watchable demo (pauses between steps):

```bash
uv run python examples/10_live_client_watch.py --skip-prepare --pause 8
```

`--prepare-only` strips FLE’s whitelist flags so the client can join. Re-run it
after every `fle cluster start` / `restart` (compose regenerates).

### Troubleshooting (live client)

| Symptom | Fix |
|---------|-----|
| Timeout / refused | Cluster Up with `34197/udp`? |
| Kicked / whitelist | `--prepare-only` again |
| RCON refused / crash-loop | Log may say whitelist pair mismatch — `--prepare-only` strips **both** flags |
| Docker HTTP 500 | Restart Docker Desktop |
| Builds flash by | Use `--pause 8` on `10_live_client_watch.py` |

---

## 3) Vision observations (multimodal agents)

`fle inspect-eval --vision` attaches base64 map images (`obs["map_image"]`).
Needs sprites — see the
[Sprites docs](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/advanced/sprites.html).
