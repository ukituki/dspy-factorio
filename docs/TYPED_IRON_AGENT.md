# Typed iron-ore agent

The typed harness uses FLE's `iron_ore_throughput` task and its verifier.
It learns about the game through scored observations, generates legally
plausible actions in Python, executes fixed FLE tool templates, and saves what
changed. The existing state-aware Python harness is available as the benchmark.

## Run a live episode

From the repository root with a local FLE server:

```bash
uv sync --locked
uv run fle cluster start -n 1
uv run python -m dspy_factorio.iron_cli run --milestone --steps 8 --renders
uv run python -m dspy_factorio.iron_cli run --starts cold unfueled blocked --steps 16 --renders
```

`--milestone` ends when observed fuel or operation confirms a fueled iron
drill. A full run ends on the FLE verifier, which requires at least 16 iron
ore per 60 game seconds. The scored budget includes the initial ore scout and
any explicit throughput measurements. Setup for prepared starts is reported
separately. JSON and standalone HTML are written under
`.fle/renders/meetup/<episode-id>/`. Open the existing meetup notebook to
review typed and raw runs.

An existing direct RCON server works as well as Docker. The FLE 0.4.3 lab
scenario with Factorio headless 2.0.73 needs `auto_pause: false` and
`autosave_interval: 0` in complete server settings. FLE injects Lua helpers the
scenario's normal autosave cannot serialize.

## Compare planners and selectors

```bash
# Configure the chosen DSPy provider credentials in .env.
uv run python -m dspy_factorio.iron_cli run \
  --harnesses scripted dspy raw --starts cold blocked --steps 16

# Configure TYPESAFE_API_KEY and use a pinned Jev model ID for published comparisons.
uv sync --locked --extra jev
uv run python -m dspy_factorio.iron_cli learn .fle/renders/meetup \
  --output .fle/iron-knowledge.json
uv run python -m dspy_factorio.iron_cli run --harnesses scripted jev-empty jev-memory \
  --knowledge .fle/iron-knowledge.json --starts unfueled blocked --steps 16

# Combine DSPy planning and Jev action selection.
uv run python -m dspy_factorio.iron_cli run --harnesses jev-memory --planner dspy \
  --knowledge .fle/iron-knowledge.json --starts blocked --steps 16
```

The runner checks that harnesses on each start have the same projected initial
state and seed. The typed harness gets structured fields during the episode,
while the raw harness sees the REPL response. Scout and measured game-reward
steps are counted in both budgets. Use the same planner when measuring the
selector or the memory effect. Compare verified completion, final and peak
throughput, scored steps, and failure outcomes. The default lab map is useful
for debugging and repeat tests. For transfer, run the task on genuinely
different Factorio maps and label each run `--split train|dev|test`. The map
seed is read from the connected server. Optional `--group` joins related maps
so they remain in one split.

Use `--first-action N` with a prepared start to observe an alternative first
choice, and `--layout N` to vary which initial drill placement the preparation
uses. Compare start fingerprints before interpreting alternatives. The
runner re-prepares those states through FLE. Capturing arbitrary exact game
checkpoints and resuming them has not been validated for this integration.

## Lifecycle

- `iron_domain.py` owns the observation projection's data model, checked
  action arguments, the candidate rules, and observable success predicates.
  Placement candidates use known entity footprints and proximity to an
  observed ore tile. Terrain, pathing, and live reach remain subject to FLE's
  execution checks. A rejected or unverified candidate is excluded for the
  rest of that rollout.
- `iron_fle.py` adapts structured observations to the domain and executes only
  templates for candidates tied to the current observation. The initial ore
  scout reads a tagged coordinate from its own fixed program. Raw FLE
  tooling state does not enter the policy interface.
- `iron_rollout.py` owns the budget and replan triggers. An ambiguous step
  stops with its failure recorded after attempting a fresh read. It avoids
  repeating a mutation whose outcome may already have changed the factory.
- `iron_policy.py` contains the fixed/DSPy planners, the deterministic and
  TypeSafe selectors, and the snapshot learner. Knowledge is collected from
  verified training transitions, grouped by subgoal, local condition, action
  kind, and evidence. Include `--update-knowledge .fle/iron-knowledge.json`
  on a training run to refresh the snapshot after each episode. Each evaluation
  episode holds the selected snapshot fixed and records its version.

## Export and score calibrated decisions

```bash
uv run python -m dspy_factorio.iron_cli export .fle/renders/meetup \
  --output .fle/iron-dataset.json --horizon 3
uv run python -m dspy_factorio.iron_cli calibrate .fle/iron-dataset.json --harness jev-memory
```

Each row asks whether the executed action led to a verified subgoal within
the specified number of scored FLE steps under the continuation policy.
Untried actions may have Jev forecasts but have no outcome labels. Rows with
an incomplete outcome window or ambiguous observation are omitted if no
success was observed. Export rejects map, family, or start overlap between
splits. Calibration fits temperature on dev and reports raw versus scaled
Brier score, log loss, and ECE on test, with the dev base rate as a comparator.
Use held-out scored runs for the intended continuation policy and report
support with the results. Jev's forecast for one horizon is excluded from
scoring at another.

The independent [OpenJev-RLCD](https://github.com/ZimmyGao/openjev-rlcd)
trainer can consume this dataset with the adapter in
`dspy_factorio/iron_rlcd.py`. From that checkout, with its own Python 3.12 and
CUDA environment, use:

```bash
PYTHONPATH="$PWD" .venv/bin/python /path/to/dspy-factorio/dspy_factorio/iron_rlcd.py \
  /path/to/iron-dataset.json --model Qwen/Qwen3-1.7B --objective rlcd \
  --no-score-function --output results/factorio-stage1.json
```

Follow the pinned upstream recipe's second stage after validating the dataset,
running a GPU smoke test, and checking the calibrated baseline on held-out
starts. The adapter passes dataset rows to the upstream task registry and does
not need the Python 3.13 Factorio environment.
