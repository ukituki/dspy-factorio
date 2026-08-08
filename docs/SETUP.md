# Setup

## Prerequisites

- Docker Desktop running and responsive (`docker ps` returns quickly)
- Python 3.13+ (this repo uses `uv`)
- Optional but recommended: Factorio desktop client ≈ **2.0.73** to watch agents live ([VISUALIZATION.md](VISUALIZATION.md))

## Install

```bash
uv sync
cp .env.example .env
# Edit .env — set at least OPENAI_API_KEY
```

Critical dependency pin (already in `pyproject.toml`):

```toml
"a2a-sdk>=0.3.26,<1"
"factorio-learning-environment[eval]>=0.4.3"
```

FLE 0.4.3 imports `TextPart` from `a2a`. `a2a-sdk` 1.x removed it — stay on 0.3.x.

## Start the Factorio cluster

```bash
uv run fle cluster start -n 1
```

On Apple Silicon FLE runs the `factoriotools/factorio:2.0.73` image via **box64**. First boot and first Lua tool inject are slow (often 1–3 minutes).

Check readiness:

```bash
# Port mapped? 
nc -z localhost 27000 && echo RCON_PORT_OPEN

# Auth works?
uv run python -c "from factorio_rcon import RCONClient; print(RCONClient('127.0.0.1',27000,'factorio').send_command('/sc rcon.print(1)'))"
```

Stop / restart:

```bash
uv run fle cluster stop
uv run fle cluster restart -n 1
```

### Recommended Apple Silicon compose tweaks

After `fle cluster start`, the generated file is:

`.venv/lib/python3.13/site-packages/fle/cluster/docker-compose.yml`

Bump resources (box64 is heavy) and pin platform:

```yaml
platform: linux/arm64
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 2048m
```

Then:

```bash
cd .venv/lib/python3.13/site-packages/fle/cluster
DOCKER_PLATFORM=linux/arm64 docker compose up -d --force-recreate --pull never
```

Note: `fle cluster start` regenerates this file — re-apply tweaks after regenerating.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM evals / DSPy agent |
| `FACTORIO_SERVER_ADDRESS` | Skip Docker discovery (use `127.0.0.1`) |
| `FACTORIO_SERVER_PORT` | RCON host port (default `27000`) |
| `FLE_DB_TYPE` | `sqlite` (default) or `postgres` |
| `FLE_TRAJECTORY_LENGTH` | Steps for inspect-eval trajectories |

Helpers in `factorio_gym.env.ensure_server_env()` set address/port defaults automatically.

## Verify installation

```bash
# List tasks (no Factorio needed)
uv run python examples/02_list_environments.py

# Live Hello World (Factorio required)
uv run python examples/01_hello_world.py
```

## Official eval CLI (0.4.x)

Upstream docs still show `fle eval --config ...`. That command raises:

> Eval is not supported anymore - Use `inspect-eval` instead

Use:

```bash
uv run fle inspect-eval \
  --env-id iron_ore_throughput \
  --model openai/gpt-4o-mini \
  --limit 1 \
  --epochs 1 \
  --trajectory-length 64 \
  --max-connections 1
```

Or:

```bash
uv run python examples/06_run_inspect_eval.py --steps 64
```

Logs land in `.fle/inspect_logs/`.
