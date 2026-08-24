# SAP AI (Python/TensorFlow)

This directory is a standalone Python implementation of the architecture from
the “Super Auto Pets Expertise” design discussion:

- exact, dependency-free shop state transitions;
- a native, data-driven battle simulator with explicit ability ordering and
  Turtle-pack coverage;
- SAP Library replay parsing and read-only Neon/Postgres ingestion;
- a small TensorFlow entity transformer with legal-action scoring, run value,
  next-battle, and expected-wins heads;
- low-budget policy-guided MCTS with sampled roll outcomes and progressive
  widening.

It does not import the old TensorFlow.js/MuZero code. The Python simulator is a
separate package so rules can be tested independently of learning code.

The rule data was audited against
[SAP Calculator](https://github.com/robertley/SAP-Calculator) commit
`d165eb0a02f8aa0b54d72ed1d5490a44390d07f4`. The simulator does not import,
execute, or package that Angular/TypeScript project. The source commit is stored
inside the rule file so datasets remain reproducible.

## Status

This is an end-to-end foundation, not a claim that every live SAP interaction is
already exact.

- Current Turtle pet, shop-food, and common-perk behavior is declared in one
  versioned JSON rulebook and executed by generic Python interpreters.
- Native battle implements base combat, Turtle pets, common perks, attack-order
  activation, faint/summon chains, Tiger repeats, and seeded random targeting.
- The shop runs applicable `Summoned`, `Friend summoned`, `Faint`,
  `Friend faints`, and `Hurt` rules too. Stat changes are permanent unless the
  ability explicitly says “until next turn”; shop faint damage can therefore
  produce further hurt/faint/summon chains.
- Battle training labels are generated directly by seeded native simulations.
- New packs should receive a complete rulebook and regression suite before they
  are enabled for training.
- The replay parser ports the important behavior from the existing
  `sap-board-query/parse-replays.js`, including coordinate reversal, permanent +
  temporary stats, pack IDs, perks, toys, and counters.
- Search ends at the battle boundary. A learned run value and battle evaluator
  supply leaf values, as proposed in the design.

The original `environment.ts` was treated as a behavioral sketch. This port
intentionally corrects several structural problems:

- `END_TURN` no longer performs hidden database I/O or battle simulation.
- RNG is injected and seedable.
- Roll is one stochastic action; possible shops are never enumerated.
- Reordering is an atomic free action rather than being multiplied into every
  other legal action.
- Buy destinations and food targets are explicit.
- Tier-up choices use `min(current_tier + 1, 6)` and are represented as grouped
  offers.
- State cloning, canonical hashing, and battle boundaries support MCTS
  transpositions and cycle detection.

## Layout

```text
src/sapai/
  sim/          state, actions, catalog, generic shop/battle interpreters
    rules/      versioned pet, food, perk, and ordering data
  data/         SAP Library replay parser and Neon client
  ml/           TensorFlow encoders, policy/value model, battle model
  search/       sampled-chance MCTS and TensorFlow evaluator adapter
tests/          simulator, ingestion, and search regression tests
```

## Installation

Simulator only:

```bash
cd sapai-python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

TensorFlow and development tools:

```bash
pip install -e '.[ml,dev]'
```

Complete local environment, including TensorFlow, Neon, and developer tools:

```bash
pip install -e '.[all]'
```

SAP Library/Neon access:

```bash
pip install -e '.[neon]'
```

Neon's Python documentation recommends a normal Postgres driver such as
Psycopg 3. The JavaScript-only `@neondatabase/serverless` package is therefore
replaced by `psycopg[binary]` in the Python package.

## Data paths

The package reads the existing current game-data directory instead of silently
copying a snapshot:

```bash
export SAP_DATA_PATH=/Users/lgtyqz/Documents/sap-ai/sap-data
```

Every CLI also accepts `--data PATH`. Pin/copy this directory into a dataset
artifact before a real training run so a balance update cannot change examples
mid-run. Store the game version with every replay/checkpoint.

## Smoke checks

From this directory:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests
.venv/bin/python -m sapai.cli --data ../sap-data catalog-report --pack Turtle
.venv/bin/python -m sapai.cli --data ../sap-data shop-demo --seed 7
.venv/bin/python -m sapai.cli --data ../sap-data model-smoke
```

Run a native battle (front pet first):

```bash
PYTHONPATH=src python -m sapai.cli --data ../sap-data battle \
  --player 'Ant,Fish' --opponent 'Mosquito,Otter' --seed 7
```

No Node.js, Angular injector, or SAP Calculator installation is needed. To add
or rebalance content, edit `src/sapai/sim/rules/turtle.json`; the Python engine
contains selectors and effect primitives, not pet-name dispatch logic.

## SAP Library / Neon

Do not copy the existing `.env`; it contains credentials. Provide the connection
string in the process environment:

```bash
export DATABASE_URL='postgresql://...?...sslmode=require'
PYTHONPATH=src python -m sapai.cli --data ../sap-data library-sample \
  --pack Turtle --turn 11 --limit 100
```

`SapLibraryClient` performs read-only, parameterized queries. For repeatable
training, export replay rows to JSONL once and use `read_replay_jsonl` rather
than querying `ORDER BY RANDOM()` during every epoch. Split by replay/player/date,
not individual boards from the same run.

## Model inputs and outputs

The policy/value model encodes 13 entities:

```text
5 team pets + 5 shop pets + 2 foods + 1 global token
```

Pet, perk, entity-type, numerical, positional, frozen, cost, turn, pack, and
version features pass through a four-layer, width-192 transformer. Legal actions
are generated by the simulator and scored as parameterized records:

```text
(kind, source, target, reorder permutation)
```

Illegal actions never enter the softmax. The model produces:

- legal-action logits;
- long-term run value;
- next-battle W/D/L auxiliary prediction;
- expected final wins auxiliary prediction.

`BattleModel` separately compares two encoded teams and produces W/D/L.

## Recommended training sequence

1. **Battle labels:** sample same-version, same-turn, same-pack board pairs from
   SAP Library and run multiple seeded `BattleSimulator` trials per pair.
2. **Battle model:** train W/D/L; hold out entire replay IDs/players/dates.
3. **Population evaluator:** cache opponent embeddings or cluster prototype
   boards for each `(pack, turn, version)`.
4. **No-search shop policy/value:** run complete Arena episodes against the
   empirical opponent distribution.
5. **Search improvement:** run 8 root candidates and 32 simulations per shop
   decision; distill root visit counts into the policy.
6. **Scale carefully:** batch 64–256 leaf evaluations on the T4. Simulator and
   orchestration throughput will likely be the bottleneck, not this network.

## Important next correctness work

Before training a competitive model:

- generate thousands of random Turtle boards and promote every discovered
  interaction error to a minimized regression fixture;
- add toys, ailments, mana, swallowed/transformed-pet edge cases, and remaining
  packs one group at a time;
- implement patch-pinned catalog snapshots and a coverage gate that refuses
  native simulation when an active pet/food/perk is unsupported;
- export a stable SAP Library dataset rather than training against a moving
  database sample.
