# SAP AI (Python/TensorFlow)

This repository is a dependency-free Super Auto Pets shop/battle simulator plus
an end-to-end TensorFlow training stack. It does not load SAP Calculator,
Angular, Node.js, or the old TensorFlow.js/MuZero project.

The Turtle rulebook was audited against SAP Calculator commit
`d165eb0a02f8aa0b54d72ed1d5490a44390d07f4`. Pet, food, perk, trigger ordering,
and shop/battle behavior live in `src/sapai/sim/rules/turtle.json`; the Python
engines execute generic selectors and effects. Shop events are real game events:
Turkey permanently buffs bought/summoned pets and Shark permanently scales when
a friend faints in the shop.

## What is implemented

- Exact, seeded shop transitions and native Turtle battle simulation.
- Structured battle frames and complete Arena shop/battle timelines.
- Sprite lookup through `assets/data/*NameId*` mappings, including token names.
- Self-contained battle and Arena HTML visualizations.
- SAP Library replay parsing and read-only Neon/Postgres ingestion.
- Stable board, W/D/L, and Arena-decision JSONL formats.
- Replay-ID-safe train/validation/test splitting and patch/catalog manifests.
- Checkpointed `BattleModel` and policy/value training with resume support.
- Empirical opponent populations and cached fixed-shape opponent tensors.
- Complete Arena rollouts using heuristic, random, model, or MCTS policies.
- Search distillation with 8 root candidates and 32 simulations by default.
- Batched TensorFlow policy/value evaluation through `evaluate_many`.
- A one-command training sequence and a Google Colab notebook.

The simulator currently targets Turtle. A training-label coverage gate raises an
error if a board contains a pet or perk absent from the pinned rulebook instead
of silently treating it as vanilla stats.

## Layout

```text
assets/                  sprites plus current pet/food/perk/toy mappings
notebooks/               Google Colab training notebook
src/sapai/
  sim/                   state, shop, battle, and data-driven rules
  data/                  replay parsing, stable serialization, dataset splits
  ml/                    encoders, models, checkpointed training pipelines
  search/                sampled-chance MCTS and TensorFlow evaluator
  training/              populations and complete Arena episodes
  visualization/         portable sprite-backed HTML renderer
tests/                    simulator and end-to-end workflow regression tests
```

## Local installation

Use Python 3.11 or 3.12. If this directory was moved after creating `.venv`,
recreate it because editable-install scripts contain absolute paths.

```bash
cd /Users/lgtyqz/Documents/sapai-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

The project now reads `assets/data` and `assets` by default. Override them with
global options before the subcommand or with environment variables:

```bash
export SAP_DATA_PATH=/path/to/assets/data
export SAP_ASSETS_PATH=/path/to/assets
python -m sapai.cli --data "$SAP_DATA_PATH" catalog-report --pack Turtle
```

## Visualizations

Generate a battle timeline (team position zero/front is listed first):

```bash
python -m sapai.cli visualize-battle \
  --player 'Ant,Cricket,Fish' \
  --opponent 'Mosquito,Horse,Otter' \
  --seed 7 \
  --output outputs/battle.html
```

Generate a complete Arena run containing every shop action and battle round:

```bash
python -m sapai.cli visualize-arena \
  --seed 3 \
  --output outputs/arena.html
```

Without `--boards`, Arena visualization uses a deterministic synthetic opponent
pool intended only for smoke tests. Use real boards and a trained policy with:

```bash
python -m sapai.cli visualize-arena \
  --boards data/boards.jsonl \
  --policy model \
  --policy-weights runs/policy-model \
  --output outputs/trained-arena.html
```

The HTML files embed only the sprites used by the timeline and work offline.
Use the slider, arrow buttons, or keyboard arrow keys to move between frames.

## Stable SAP Library data

Do not copy an existing `.env` into the repository. Set the Neon URL in the
process environment and export once:

```bash
export DATABASE_URL='postgresql://...?...sslmode=require'
python -m sapai.cli export-boards \
  --pack Turtle \
  --limit 10000 \
  --output data/boards.jsonl
```

If raw replay rows are already available as JSONL:

```bash
python -m sapai.cli parse-replays \
  --input data/raw-replays.jsonl \
  --output data/boards.jsonl
```

Keep `boards.jsonl` with the run artifacts. Training never queries a moving
database sample during an epoch.

## Recommended training sequence

The entire sequence is executable. For a first Colab validation run, use small
counts; increase them after all cells pass.

### 1. Generate native battle labels

Pairs are matched on `(turn, pack, version)`. Whole replay IDs are assigned to
one split before pairs are sampled, preventing the same run from leaking into
validation or test data.

```bash
python -m sapai.cli label-battles \
  --boards data/boards.jsonl \
  --output runs/battle-dataset \
  --examples 100000 \
  --simulations-per-pair 8 \
  --seed 2026
```

### 2. Train the W/D/L battle model

```bash
python -m sapai.cli train-battle \
  --dataset runs/battle-dataset \
  --output runs/battle-model \
  --epochs 20 \
  --batch-size 128 \
  --seed 2026
```

Every epoch writes a TensorFlow checkpoint and `history.json`. Rerunning the
same command resumes automatically; pass `--no-resume` for a fresh optimizer.

### 3. Cache the empirical population

```bash
python -m sapai.cli cache-population \
  --boards data/boards.jsonl \
  --output runs/population.npz
```

This stores stable encoded opponent inputs and metadata. It deliberately does
not cache model-dependent embeddings, so the file remains valid as weights
change.

### 4. Bootstrap a no-search policy/value model

```bash
python -m sapai.cli generate-arena \
  --boards data/boards.jsonl \
  --policy heuristic \
  --episodes 1000 \
  --output runs/arena-bootstrap.jsonl \
  --seed 2026

python -m sapai.cli train-policy \
  --dataset runs/arena-bootstrap.jsonl \
  --output runs/policy-model \
  --epochs 20 \
  --batch-size 128 \
  --seed 2026
```

Each decision includes the exact legal actions, selected/target policy,
next-battle W/D/L, terminal run value, and final trophy count.

### 5. Search and distill root visits

```bash
python -m sapai.cli generate-arena \
  --boards data/boards.jsonl \
  --policy search \
  --policy-weights runs/policy-model \
  --search-candidates 8 \
  --search-simulations 32 \
  --episodes 250 \
  --output runs/arena-search.jsonl \
  --seed 3026

python -m sapai.cli train-policy \
  --dataset runs/arena-search.jsonl \
  --output runs/policy-model \
  --epochs 25 \
  --batch-size 128 \
  --seed 2026
```

The second policy command resumes the 20-epoch checkpoint and trains through
epoch 25, so it performs five distillation epochs.

### One-command version

```bash
python -m sapai.cli train-sequence \
  --boards data/boards.jsonl \
  --workdir /content/drive/MyDrive/sapai-runs/run-001 \
  --battle-examples 100000 \
  --battle-epochs 20 \
  --bootstrap-episodes 1000 \
  --bootstrap-epochs 20 \
  --search-episodes 250 \
  --search-epochs 5 \
  --batch-size 128 \
  --seed 2026
```

Outputs include manifests, model configs, rolling checkpoints, final weights,
training histories, opponent tensors, both trajectory datasets, and a final
`summary.json`.

## Google Colab

Open `notebooks/sapai_colab_training.ipynb`, choose a T4 GPU runtime, provide
the Git repository URL and Drive path in the first configuration cell, then run
all cells. The notebook:

1. mounts Drive;
2. clones or updates this repository;
3. installs the editable `ml` package;
4. verifies GPU visibility and runs tests;
5. copies or locates stable `boards.jsonl`;
6. runs a small smoke sequence, then the configurable full sequence;
7. writes every checkpoint and dataset to Drive.

For T4 runs, start with batch size 64–128. `TensorFlowEvaluator.evaluate_many`
can evaluate 64–256 independent leaves per model call, although adaptive MCTS
tree traversal and Python simulation will usually be the throughput bottleneck.

## Model contract

The policy/value transformer encodes 13 entities:

```text
5 team pets + 5 shop pets + 2 foods + 1 global token
```

It scores simulator-generated legal actions represented by
`(kind, source, target, reorder permutation)` and produces legal-action logits,
run value, next-battle W/D/L, and expected final wins. Illegal actions never
enter the softmax. `BattleModel` compares two encoded teams and predicts W/D/L.

## Verification

```bash
python -m pytest -q
python -m ruff check src tests
python -m sapai.cli model-smoke
```

Before claiming competitive accuracy, continue expanding minimized regression
fixtures for toys, ailments, mana, swallowed/transformed-pet edge cases, and
future packs. Those mechanics should receive a pinned rulebook and coverage
gate before their boards are enabled for native training labels.
