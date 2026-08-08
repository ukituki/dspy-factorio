# Hello World tutorial

Goal: connect to a live Factorio server, reset an FLE task, run one Python program, and print the REPL output.

## 0. Prerequisites

1. Docker Desktop healthy (`docker ps` is fast)
2. `.env` has `OPENAI_API_KEY` only if you later use LLM examples (not needed for Hello World)
3. Cluster up:

```bash
uv run fle cluster start -n 1
```

Wait until RCON authenticates (see [SETUP.md](SETUP.md)).

## 1. Mental model (REPL)

FLE agents do not press keys. Each **action** is a Python program executed inside the game sandbox:

1. **Observe** — stdout/stderr from the last program (`obs["raw_text"]`)
2. **Act** — emit new Python using tools like `nearest`, `place_entity`, `insert_item`
3. **Feedback** — environment returns text + structured fields + reward

```text
you  --Python-->  Factorio tools  --stdout-->  you
```

## 2. Minimal program

```python
from factorio_gym.env import make_env, reset_env, step_code, obs_text

env = make_env("iron_ore_throughput", run_idx=0)
obs, info = reset_env(env)

obs, reward, terminated, truncated, info = step_code(
    env,
    """
pos = nearest(Resource.IronOre)
print(pos)
print(inspect_inventory())
""".strip(),
)
print(obs_text(obs))
env.close()
```

Run the packaged version:

```bash
uv run python examples/01_hello_world.py
```

Expected: a Position for iron ore and an inventory dump. First connect injects many Lua tools — can take a minute on Apple Silicon.

## 3. Pitfalls the upstream quickstart misses

| Pitfall | Fix |
|---------|-----|
| `gym.make("iron_ore_throughput")` → `missing run_idx` | Pass `run_idx=0` |
| Docker discovery hangs | Set `FACTORIO_SERVER_ADDRESS=127.0.0.1` and `FACTORIO_SERVER_PORT=27000` (helpers do this) |
| `import a2a.types.TextPart` fails | Keep `a2a-sdk<1` |
| Deprecated `gym` import warnings | Use `factorio_gym.env` helpers (they silence gym-notices; FLE still needs `gym`) |

## 4. What to try next

1. `uv run python examples/03_scripted_miner.py` — multi-step scripted scenario  
2. `uv run python examples/10_live_client_watch.py` — watch in the Factorio client ([VISUALIZATION.md](VISUALIZATION.md))  
3. `uv run python examples/09_visualize_renders.py` — PNG map dumps if you lack the client  
4. `uv run python examples/04_dspy_agent_loop.py --steps 3` — LLM writes the programs  
5. Read [SCENARIOS.md](SCENARIOS.md) to design your own task loop  

## 5. Useful in-game APIs (starter set)

```python
nearest(Resource.IronOre)          # NEVER nearest("iron-ore")
move_to(iron)                      # Position arg — never move_to(pos=...); required before place_entity if far
place_entity(entity=Prototype.BurnerMiningDrill, position=pos, direction=Direction.NORTH)
place_entity_next_to(entity=Prototype.IronChest, reference_position=drill.drop_position, direction=Direction.SOUTH)
insert_item(Prototype.Coal, drill, quantity=5)
connect_entities(source, target, connection_type=Prototype.TransportBelt)
get_entities()
inspect_inventory()
sleep(10)
```

Print aggressively — the observation channel is text. Always use `Resource.*` / `Prototype.*` enums, not strings.
