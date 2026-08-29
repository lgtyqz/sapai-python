The core paradigm is reasonable, but it is not yet a reliable iterative policy-improvement system. The biggest issues are the reward definition, bootstrap action coverage, and lack of held-out Arena evaluation—not model size or training duration.

## What the pipeline currently does

```text
Empirical boards
  └─ opponent population

1,000 heuristic Arena episodes
  └─ one-hot heuristic actions + actual run results
       └─ train PolicyValueModel for 20 epochs

Frozen bootstrap policy
  └─ generate 250 MCTS-guided episodes
       ├─ 32 tree traversals per decision
       └─ 8 exact battles per evaluated END_TURN leaf

Search visit distributions
  └─ continue training the same model for 5 epochs
       └─ final policy
```

This is behavior cloning followed by one search-distillation pass. It is not iterative self-play or repeated policy iteration.

The model trains four heads at [models.py:189](/Users/lgtyqz/Documents/sapai-python/src/sapai/ml/models.py:189):

- Legal-action policy.
- Final-run value.
- Next-battle W/D/L.
- Final trophy count.

The simulator leaf values influence MCTS visits and therefore the policy target. The actual completed Arena reward directly trains the value head.

## What is sound

Several assumptions are correct:

- The real simulator is authoritative for hypothetical and actual battles.
- Hidden opponents are sampled from the population rather than revealed to search.
- Hypothetical outcomes are applied to cloned states before continuation evaluation.
- Nonterminal continuation states are evaluated together.
- Actual Arena outcomes remain the source of final value targets.
- Search and actual Arena randomness use separate streams.
- End Turn is now guaranteed a candidate slot and its sampled value is cached per tree.

Those pieces form a valid one-step model-based bootstrap.

## High-priority correctness issues

### 1. The value objective is internally inconsistent

The policy value target is:

```text
10 trophies: 1.0
otherwise: trophies² / 200
```

Thus a 9-trophy loss is `0.405`, while a 10-trophy win is `1.0`. Meanwhile, the generic MCTS terminal fallback returns `-1` for a loss at [stochastic.py:263](/Users/lgtyqz/Documents/sapai-python/src/sapai/search/stochastic.py:263). The model uses a `tanh` output even though its training targets are normally nonnegative.

This makes “value” ambiguous:

- Probability of winning the Arena?
- Expected trophies?
- A shaped utility favoring completion?

Recommended resolution:

- Make the value head predict Arena completion probability: `1` for 10 trophies, `0` for terminal loss, using a sigmoid.
- Rename `expected_wins` to `expected_trophies`, normalize its target to `trophies / 10`, and use it as the dense shaping signal.
- If win/loss symmetry is preferred, use `+1/-1` consistently everywhere with `tanh`.
- Centralize all terminal-value calculations in one function.

The current squared reward should remain only if that exact utility is intentional.

### 2. Bootstrap training suppresses strategically important actions

The heuristic at [arena.py:39](/Users/lgtyqz/Documents/sapai-python/src/sapai/training/arena.py:39) never chooses:

- Freeze or unfreeze.
- Sell.
- Reorder.
- Usually board merge.
- Roll except in limited situations.

In a 50-episode synthetic diagnostic:

- 2,782 decisions were generated.
- Freeze, sell, reorder, and board merge were selected zero times.
- Roll was selected 295 times, about 10.6%.
- Reorder permutations contributed 275,947 individual legal-action instances.

The synthetic population is not a performance benchmark, but the action-support result follows directly from the heuristic.

Search then keeps only eight actions, one of which is reserved for End Turn. The remaining seven are selected using a network trained to assign near-zero probability to several entire action classes. Search therefore has little opportunity to discover good freeze, sell, or reorder strategies.

Recommended changes:

- Bootstrap from a mixture of heuristic and exploratory policies.
- Sample exploration uniformly by action kind, then within the selected kind. Uniform sampling over individual actions would be overwhelmed by reorder permutations.
- Reserve MCTS candidate slots across legal action kinds.
- Mix policy priors with a small action-kind-uniform prior.
- Include sensible rule-based examples of freezing before rolling, selling, and positioning.

### 3. There is no held-out Arena evaluation

`train-sequence` uses the complete compatible board population for:

- Bootstrap opponents.
- Search-leaf opponents.
- Actual search-rollout opponents.
- Any later visualization or informal evaluation.

It does not pass a validation dataset to policy training, and it does not run a final model benchmark. Training losses therefore cannot show whether Arena performance improved.

A proper split should be made by replay ID:

```text
80% training opponent pool
10% validation opponent pool
10% frozen test opponent pool
```

Preserve turn/version coverage in every split. Then evaluate fixed seeds against the test population for:

- Heuristic policy.
- No-search policy model.
- Search policy.
- Human benchmark, when comparison is desired.

Report trophy histogram, mean/median trophies, completion rate, battle W/D/L, turns, and confidence intervals.

### 4. Patch versions are effectively mixed

Generated `RunState` objects use version `"current"`. Empirical boards usually contain actual version strings. When no exact `"current"` group exists, `OpponentPopulation` falls back to any board from the same turn.

That means version filtering usually does not operate as it appears to. Boards from different patches can be mixed while the simulator uses one pinned rulebook.

Recommended options:

- Filter the export to one supported patch.
- Give generated states the pinned simulator version.
- Explicitly configure allowed versions.
- Log opponent counts per turn/version and fail rather than silently using distant fallbacks for serious training.

## Training-quality improvements

### Iterate search and training

There is currently only one search generation followed by five epochs of distillation. The improved model never generates better search targets.

A stronger loop is:

```text
Bootstrap policy
  → search iteration 1
  → train 1–3 epochs
  → held-out evaluation
  → search iteration 2 with updated model
  → train
  → evaluate
  → repeat
```

Maintain a replay buffer containing:

- Recent search trajectories.
- Some older search trajectories.
- Approximately 10–20% exploratory/bootstrap data.

This limits catastrophic forgetting and keeps rare action classes represented.

### Reduce battle-leaf sampling variance

Eight samples have a worst-case standard error of roughly `0.177` for a bounded W/D/L score. Because each leaf is sampled once and cached, additional MCTS visits never refine an unlucky estimate.

Better options:

- Use a common opponent-and-seed panel for all end-turn candidates in one search.
- Sample opponents without replacement when possible.
- Start with four samples, then increase to 8/16/32 for competitive leaves.
- Cache cumulative outcome statistics rather than one final value.
- Stratify opponents by replay, strength, or version instead of pure replacement sampling.

### Correct the `next_battle` semantics

Every decision made during a turn receives the battle outcome after all later shop actions are completed at [arena.py:224](/Users/lgtyqz/Documents/sapai-python/src/sapai/training/arena.py:224).

Therefore `next_battle` means:

> Outcome after following the behavior policy for the rest of this turn.

It does not mean:

> Outcome if the player ended the turn in this state.

That auxiliary target is valid, but it should be renamed or documented as policy-conditioned. If the intended target is current board strength, create examples only from post-`END_TURN` boards or simulate “end now” explicitly.

### Improve action scoring

The transformer returns entity representations and a pooled state, but the policy head discards individual entity representations at [models.py:119](/Users/lgtyqz/Documents/sapai-python/src/sapai/ml/models.py:119). An action is scored from:

- Global pooled state.
- Action kind.
- Integer source and target positions.
- Reorder indices.

The head never directly gathers the pet or food embedding at the selected source/target. It must reconstruct all position-to-entity relationships from one pooled vector.

A stronger scorer would concatenate:

```text
pooled state
+ source entity embedding
+ target entity embedding
+ action-kind embedding
```

Reordering should eventually be factorized into move/swap or pointer-style decisions rather than represented as up to 120 flat permutations.

## Performance findings

The most immediate training optimization is graph compilation.

On the local CPU with a batch of 128:

| Operation                |   Median |
| ------------------------ | -------: |
| Encoding the batch       |  20.8 ms |
| Current eager train step | 2,617 ms |
| `tf.function` train step |   586 ms |

That is approximately a 4.5× improvement after one-time tracing. Colab T4 numbers will differ, but repeated eager execution is clearly the wrong default.

Recommended optimization order:

1. Compile training and inference with stable input signatures.
2. Pad batches to fixed shapes to avoid retracing.
3. Add a training-only `record_trace=False` battle mode—search currently generates logs and animation frames it throws away.
4. Add `record_timeline=False` to `ArenaRunner`; training currently builds shop frames and full battle timelines that are discarded.
5. Batch MCTS leaf inference across concurrent Arena episodes.
6. Use mixed precision on T4 after compiled execution is stable.
7. Cache or shard encoded training tensors if encoding becomes the GPU bottleneck.

Tree reuse would also help: search currently clears its transposition table for every shop decision. Re-rooting at the chosen child would preserve useful work, including observed roll outcomes.

## Storage and data-loading inefficiencies

A diagnostic bootstrap file with 48 decisions occupied about 460 KiB, or roughly 9.6 KiB per decision. Full bootstrap training can produce tens of thousands of decisions.

Each episode is stored once in its checkpoint file and again in the combined JSONL, approximately doubling trajectory storage.

Improvements:

- Train directly from episode shards.
- Avoid writing the duplicate combined JSONL.
- Compress shards with gzip or zstd.
- Store compact encoded tensors rather than repeated full legal-action dictionaries.
- Stream or memory-map datasets instead of loading every decision into a Python list.

## Resume and reproducibility assumptions

Resume restores weights and optimizer state, but it is not bitwise equivalent to uninterrupted training:

- Python shuffling restarts from `seed + completed_epoch`, not the uninterrupted RNG state.
- TensorFlow/dropout RNG state is not checkpointed.
- `config.json` is overwritten before checkpoint compatibility is validated.
- Training checkpoints are not tied to a dataset hash.
- Episode manifests do not include simulator/rulebook/reward versions.
- `history.json` is not written atomically.

These do not always cause visible failures, but they weaken reproducibility. Immutable model, reward, simulator, dataset, and optimizer identities should be recorded and checked before restore.

## Recommended implementation order

1. Define the objective and unify reward/terminal-value semantics.
2. Add train/validation/test opponent populations and an automated Arena evaluation gate.
3. Fix bootstrap action coverage and action-kind-aware MCTS candidates.
4. Compile TensorFlow training/inference and disable unused simulator timelines.
5. Add common battle panels and adaptive leaf resampling.
6. Convert the one-shot search pass into iterative policy improvement with a replay mixture.
7. Improve the policy head’s direct source/target entity conditioning.
8. Harden manifests, checkpoint compatibility, and deterministic resume.

I would do the first four before investing in a longer full run. More epochs on the current targets would mostly make the existing biases more confident.
