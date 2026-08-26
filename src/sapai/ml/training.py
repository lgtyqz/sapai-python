from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sapai.data.replay import BoardSnapshot
from sapai.ml.encoding import encode_teams
from sapai.sim.battle import BattleResultKind, BattleSimulator
from sapai.sim.models import Team


@dataclass(frozen=True, slots=True)
class BattleExample:
    player: Team
    opponent: Team
    turn: int
    player_pack: str
    opponent_pack: str
    target: tuple[float, float, float]  # player win, draw, opponent win


def label_board_pairs(
    boards: Sequence[BoardSnapshot],
    simulator: BattleSimulator,
    *,
    examples: int,
    simulations_per_pair: int = 8,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[BattleExample]:
    """Sample compatible boards and label them with the native simulator."""

    rng = random.Random(seed)
    groups: dict[tuple[int, str, str], list[BoardSnapshot]] = {}
    for board in boards:
        simulator.assert_team_supported(board.team)
        groups.setdefault((board.turn, board.pack, board.version), []).append(board)
    eligible = [values for values in groups.values() if len(values) >= 2]
    if not eligible:
        raise ValueError("need at least two boards with matching turn, pack, and version")

    pairs: list[tuple[BoardSnapshot, BoardSnapshot]] = []
    for _ in range(examples):
        group = rng.choice(eligible)
        player, opponent = rng.sample(group, 2)
        pairs.append((player, opponent))

    labeled: list[BattleExample] = []
    progress_interval = max(1, examples // 20)
    for index, (player, opponent) in enumerate(pairs, start=1):
        counts = {kind: 0 for kind in BattleResultKind}
        for _ in range(simulations_per_pair):
            result = simulator.simulate(
                player.team,
                opponent.team,
                seed=rng.getrandbits(63),
            )
            counts[result.outcome] += 1
        total = max(1, simulations_per_pair)
        labeled.append(
            BattleExample(
                player=player.team.clone(),
                opponent=opponent.team.clone(),
                turn=player.turn,
                player_pack=player.pack,
                opponent_pack=opponent.pack,
                target=(
                    counts[BattleResultKind.PLAYER_WIN] / total,
                    counts[BattleResultKind.DRAW] / total,
                    counts[BattleResultKind.OPPONENT_WIN] / total,
                ),
            )
        )
        if progress is not None and (
            index == 1 or index == examples or index % progress_interval == 0
        ):
            progress(index, examples)
    return labeled


def encode_battle_examples(examples: Sequence[BattleExample]):
    """Create TensorFlow-ready model inputs and soft W/D/L targets."""

    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to encode battle examples") from error
    player = encode_teams(
        [example.player for example in examples],
        turns=[example.turn for example in examples],
        packs=[example.player_pack for example in examples],
    )
    opponent = encode_teams(
        [example.opponent for example in examples],
        turns=[example.turn for example in examples],
        packs=[example.opponent_pack for example in examples],
    )
    inputs = {
        **{f"player_{key}": value for key, value in player.items()},
        **{f"opponent_{key}": value for key, value in opponent.items()},
    }
    targets = np.asarray([example.target for example in examples], dtype=np.float32)
    return inputs, targets


try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover
    tf = None


if tf is not None:

    class BattleTrainer:
        def __init__(self, model, optimizer=None):
            self.model = model
            self.optimizer = optimizer or tf.keras.optimizers.AdamW(
                learning_rate=3e-4, weight_decay=1e-4, clipnorm=5.0
            )

        def train_step(self, inputs, targets):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        targets, outputs["logits"], from_logits=True
                    )
                )
                if self.model.losses:
                    loss += tf.add_n(self.model.losses)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(targets, axis=-1), tf.argmax(outputs["logits"], axis=-1)),
                    tf.float32,
                )
            )
            return {"loss": loss, "accuracy": accuracy}

else:

    class BattleTrainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow is not installed; install the 'ml' extra")
