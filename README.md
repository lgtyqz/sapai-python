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

The simulator currently targets Turtle. The rulebook automatically recognizes
its generated token pets, and catalog pets explicitly described as having no
ability use exact vanilla combat. Unknown replay pet IDs also use their recorded
stats as a tagged vanilla fallback so new catalog IDs do not stop ingestion. The
training-label coverage gate still raises for known ability pets absent from the
pinned rulebook. Unsupported or unknown perks are retained, tagged, reported,
and treated as no-effect fallbacks so cross-pack replay ailments do not stop a
run. Boards labeled Turtle but containing known pets exclusive to another pack
are excluded from Turtle training and opponent populations. Database pack
columns supply missing in-replay pack labels, and snapshots with no pets are
skipped during export.

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

Use Python 3.11 or 3.12. Virtual environments are not portable: activation and
installed command scripts contain the absolute directory in which the
environment was created.

For a fresh clone with no `.venv`:

```bash
cd /Users/lgtyqz/Documents/sapai-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
command -v python
python -c 'import sapai; print(sapai.__file__)'
```

The last two commands must report this repository's `.venv/bin/python` and
`src/sapai/__init__.py` respectively.

If the repository was moved after `.venv` was created, first move the stale
environment aside and create a genuinely new one. Running `python3 -m venv
.venv` over the existing directory is not a reliable recreation.

```bash
cd /Users/lgtyqz/Documents/sapai-python
deactivate 2>/dev/null || true
mv .venv .venv-before-move
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
command -v python
python -m sapai.cli --help
```

The backup can be removed after the new environment passes the smoke checks.
Always prefer `python -m pip`, `python -m pytest`, and `python -m sapai.cli`
after activation; this guarantees the module is resolved by the interpreter
shown by `command -v python`.

If an error names `/opt/homebrew/.../python3.12` instead of `.venv/bin/python`,
the virtual environment is not actually active. If it names a previous project
directory, the environment was moved and must be recreated as above.

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

Battle teams share one horizontal battlefield, face toward the center, and keep
their SAP front-to-back ordering. Attack, impact, damage, buff, perk, summon,
and faint transitions are animated; use Play/Pause and the speed selector for
automatic playback, or the slider and arrow keys for manual stepping.

Rendering uses the packaged `timeline.html` template. Each output directory gets
one shared `sapai.css`, one shared `sapai.js`, and a `sapai-assets/` directory
containing only the sprites used there. Battle and Arena HTML files contain only
their timeline JSON and reference those shared files, avoiding repeated runtime
code and base64 sprite data. The resulting directory works offline; copy the
HTML, CSS, JavaScript, and `sapai-assets/` directory together when moving it.

### Human Arena benchmark in Colab

The training notebook also contains an optional card-driven human benchmark.
Set `RUN_HUMAN_BENCHMARK=True`, choose a participant alias and a separate
`HUMAN_BENCHMARK_DIR`, then run the final benchmark cell after repository,
Drive, installation, and board validation setup. Set `REQUIRE_GPU=False` when
the notebook is being used only for human play.

The benchmark samples opponents through the same compatible board population
used by model Arena rollouts. It checkpoints every accepted move, restores the
current shop or battle review after a runtime reconnect, and writes immutable
completed episodes plus an aggregate `summary.json` to Drive. Rerun the final
cell after reconnecting because Colab callbacks belong to the current runtime.
Human benchmark artifacts are kept separate from policy-training datasets.

### Kaggle notebook

[`notebooks/sapai_kaggle_training.ipynb`](notebooks/sapai_kaggle_training.ipynb)
provides the same smoke run, resumable full-training sequence, portable replay,
and human Arena benchmark for Kaggle. It uses `/kaggle/working` for writable
outputs, `kaggle_secrets.UserSecretsClient` for `DATABASE_URL`, and a standard
Jupyter widget channel for the interactive card UI. The Kaggle UI uses only
the core `ipywidgets` models already bundled by Kaggle; it does not require a
separately registered AnyWidget frontend module. Dragging progressively
enhances the team cards, while **Reorder team** remains a complete fallback if
the notebook frontend blocks output JavaScript.

To continue from a saved Kaggle notebook version, attach that version as an
input and set `KAGGLE_PRIOR_RUN_DIR` or `KAGGLE_PRIOR_HUMAN_DIR` to the relevant
directory under `/kaggle/input`. The notebook copies it into writable storage
without replacing a newer in-session directory. Enable Internet for the Git
checkout, dependency installation, and database export; an attached
`boards.jsonl` avoids the database step.

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

Freeze choices are canonicalized once per shop roll: changing an offer's
frozen state removes its inverse action until the next roll. MCTS also skips
deterministic transitions back to an ancestor, so zero-cost action cycles do
not consume rollout depth or become policy targets.

### One-command version

```bash
python -m sapai.cli train-sequence \
  --boards data/boards.jsonl \
  --workdir /content/drive/MyDrive/sapai-runs/run-001 \
  --pack Turtle \
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
training histories, opponent tensors, resumable per-episode rollout files, both
combined trajectory datasets, and a final `summary.json`.

After a Colab disconnect, reconnect with the same `DRIVE_RUN_DIR`, board export,
seed, and data-generation counts. Rerunning `train-sequence` reuses completed
datasets and rollout episodes and restores model plus optimizer state from the
latest epoch checkpoint. Only an epoch interrupted before its checkpoint is
repeated.

Keras optimizer slots are built before checkpoint restoration so model tensors
tracked through Keras 3 optimizers are matched immediately. Each newly completed
epoch also writes an explicit `epoch-N.weights.h5` model snapshot beside the
TensorFlow optimizer/epoch checkpoint, and resume validates that checkpoint
number, completed epoch, and optimizer progress agree.

## Google Colab

Open `notebooks/sapai_colab_training.ipynb`, choose a T4 GPU runtime, add a
`DATABASE_URL` Colab secret, grant the notebook access to it, provide the Git
repository URL and Drive paths in the first configuration cell, then run all
cells. The notebook:

1. mounts Drive;
2. clones or updates this repository;
3. installs the editable package and exposes its `src` layout to the live Colab kernel;
4. verifies GPU visibility and runs tests;
5. creates `boards.jsonl` from the database when absent, then validates and copies it
   to the runtime for faster reads;
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
