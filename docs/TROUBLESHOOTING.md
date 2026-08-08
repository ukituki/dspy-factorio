# Troubleshooting

Lessons from bringing up FLE 0.4.3 on Apple Silicon.

## `fle eval` fails

```
Error: Eval is not supported anymore - Use `inspect-eval` instead
```

Use `fle inspect-eval` (see [SETUP.md](SETUP.md)).

## `inspect-eval --config ...` → `NoneType is not iterable`

Upstream bug: `"openrouter" in args.model` when `--model` is omitted. Always pass `--model`.

## `ImportError: cannot import name 'TextPart' from 'a2a.types'`

`a2a-sdk` 1.x is incompatible. Pin:

```bash
uv add 'a2a-sdk>=0.3.26,<1'
```

## `gym.make(...)` → `missing 1 required positional argument: 'run_idx'`

```python
gym.make("iron_ore_throughput", run_idx=0)
# or
from dspy_factorio.env import make_env
make_env("iron_ore_throughput", run_idx=0)
```

Prefer `make_env` — it also silences OpenAI Gym deprecation notices and disables the noisy default env checker.

## Gym deprecation banner / observation-space warnings

FLE still depends on unmaintained `gym` (not Gymnasium — separate registries). Importing via `dspy_factorio.env` stubs `gym-notices` and uses `disable_env_checker=True`. Do not swap in Gymnasium for `gym.make` unless FLE registers into that registry too.

## `AttributeError: 'str' object has no attribute 'value'` from `nearest(...)`

The program used a string resource name. FLE expects enums:

```python
nearest(Resource.IronOre)  # correct
nearest("iron-ore")        # crashes
```

Same rule for `place_entity(..., Prototype.BurnerMiningDrill, ...)`.

## `target position is too far away to place the entity`

Player must be within ~10 tiles. Call `move_to(iron)` / `move_to(position=iron)` before `place_entity` — never `move_to(pos=...)`.

## Docker Desktop hangs / `500` on docker.sock

Symptoms: `docker ps` stalls, containers become unkillable, RCON auth resets.

1. Quit Docker Desktop fully and reopen  
2. Wait until `docker ps` is snappy  
3. `uv run fle cluster start -n 1`  
4. Prefer `FACTORIO_SERVER_ADDRESS=127.0.0.1` to skip discovery  

Heavy stacks (Supabase, etc.) auto-starting alongside Factorio make this worse.

## RCON port open but auth fails

Factorio is still loading mods/map under box64. Wait for log line:

```text
Starting RCON interface
```

Then retry auth. First boot after recreate can take 1–2 minutes.

## Env create hangs while "Loading action ... place_entity"

Lua tool injection over RCON is slow on box64. Give it several minutes **once**. If CPU on the container drops to 0% and nothing progresses for >5 minutes, restart Docker + cluster.

Giving the container 4 CPUs / 2GB helps (see [SETUP.md](SETUP.md)).

## inspect-eval Pass@8 fails with "Container index N exceeds available containers"

Default `--pass-n` is 8. With one Factorio instance:

```bash
--epochs 1
```

Or start more instances: `fle cluster start -n 8`.

## `Unknown GenerateConfig field(s): transforms`

inspect-ai rejects top-level `transforms`. Provider options belong in `extra_body`. This repo's installed FLE solver was patched under:

`.venv/lib/python3.13/site-packages/fle/eval/inspect/integration/solver.py`

Re-installing FLE may revert the patch — re-apply or vendor a wrapper.

## `.env` `source` parse error

Avoid `AWS_ACCESS_KEY_ID=""`. Use empty unquoted values or omit the keys. Prefer `dotenv` / `load_dotenv` over `source .env`.

## `UnsupportedParamsError: gpt-5 … don't support temperature=0.2`

Most `gpt-5*` models (e.g. `gpt-5.5-luna`, `gpt-5-codex`) only accept `temperature=1`. `dspy_factorio.agent.build_lm` forces `1.0` for those IDs; `gpt-5.1` and non-gpt-5 models keep `AgentConfig.temperature` (default `0.2`).

## Score is always 0

Short trajectories rarely meet throughput quotas. Increase `--trajectory-length` / `--steps`, fuel machines, and `sleep` so production accrues.
