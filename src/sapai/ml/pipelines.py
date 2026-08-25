"""Checkpointed TensorFlow training loops used locally and in Colab."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sapai.data.datasets import read_battle_examples
from sapai.ml.encoding import encode_states
from sapai.ml.models import (
    BattleModel,
    ModelConfig,
    PolicyValueModel,
    PolicyValueTrainer,
)
from sapai.ml.training import BattleTrainer, encode_battle_examples
from sapai.training.arena import ArenaDecision, read_arena_decisions


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    max_actions: int = 256


def _tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to train models") from error
    return tf


def _batches(values: list[Any], batch_size: int, rng: random.Random):
    indices = list(range(len(values)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [values[index] for index in indices[start : start + batch_size]]


def _tensor_dict(tf, values: dict[str, object]) -> dict[str, object]:
    return {key: tf.convert_to_tensor(value) for key, value in values.items()}


def _write_run_metadata(
    output: Path,
    *,
    kind: str,
    training: TrainingConfig,
    model: ModelConfig,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {"kind": kind, "training": asdict(training), "model": asdict(model)},
            indent=2,
        ),
        encoding="utf-8",
    )


def _epoch_weights_path(output: Path, epoch: int) -> Path:
    return output / "checkpoints" / f"epoch-{epoch}.weights.h5"


def _save_epoch_weights(model, output: Path, epoch: int, checkpoints: list[str]) -> None:
    destination = _epoch_weights_path(output, epoch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"epoch-{epoch}.tmp.weights.h5")
    model.save_weights(temporary)
    temporary.replace(destination)
    keep = {int(Path(path).name.rsplit("-", 1)[-1]) for path in checkpoints}
    keep.add(epoch)
    for path in destination.parent.glob("epoch-*.weights.h5"):
        name = path.name.removesuffix(".weights.h5")
        if name.endswith(".tmp"):
            continue
        saved_epoch = int(name.rsplit("-", 1)[-1])
        if saved_epoch not in keep:
            path.unlink()


def _restore_training_checkpoint(
    tf,
    *,
    model,
    optimizer,
    completed_epochs,
    manager,
    output: Path,
    resume: bool,
) -> tuple[int, str | None]:
    if not resume or not manager.latest_checkpoint:
        return 0, None

    # Keras 3 checkpoints expose model variables through the optimizer's tracked
    # trainable-variable list. Build slots before restore so those tensors match
    # immediately instead of remaining deferred while only the epoch counter loads.
    optimizer.build(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
    )
    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    restored_epoch = int(completed_epochs.numpy())
    checkpoint_epoch = int(Path(manager.latest_checkpoint).name.rsplit("-", 1)[-1])
    if checkpoint_epoch != restored_epoch:
        raise RuntimeError(
            f"checkpoint epoch mismatch: file={checkpoint_epoch}, state={restored_epoch}"
        )
    if restored_epoch > 0 and int(optimizer.iterations.numpy()) == 0:
        raise RuntimeError("checkpoint restored an epoch counter but no optimizer progress")
    explicit_weights = _epoch_weights_path(output, restored_epoch)
    if explicit_weights.exists():
        model.load_weights(explicit_weights)
    return restored_epoch, manager.latest_checkpoint


def train_battle_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    training_config: TrainingConfig | None = None,
    model_config: ModelConfig | None = None,
    resume: bool = True,
) -> dict[str, object]:
    tf = _tensorflow()
    training = training_config or TrainingConfig()
    model_settings = model_config or ModelConfig(num_layers=3)
    tf.keras.utils.set_random_seed(training.seed)
    train = read_battle_examples(Path(dataset_dir) / "train.jsonl")
    validation_path = Path(dataset_dir) / "validation.jsonl"
    validation = read_battle_examples(validation_path) if validation_path.exists() else []
    if not train:
        raise ValueError("battle training dataset is empty")

    output = Path(output_dir)
    _write_run_metadata(
        output,
        kind="battle",
        training=training,
        model=model_settings,
    )
    model = BattleModel(model_settings)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        clipnorm=5.0,
    )
    trainer = BattleTrainer(model, optimizer)
    sample_inputs, _ = encode_battle_examples(train[:1])
    model(_tensor_dict(tf, sample_inputs), training=False)
    optimizer.build(model.trainable_variables)
    completed_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=3)
    restored_epoch, restored_checkpoint = _restore_training_checkpoint(
        tf,
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
        manager=manager,
        output=output,
        resume=resume,
    )

    rng = random.Random(training.seed + int(completed_epochs.numpy()))
    history = _read_history(output)
    for epoch in range(int(completed_epochs.numpy()), training.epochs):
        metrics: list[dict[str, float]] = []
        for batch in _batches(train, training.batch_size, rng):
            inputs, targets = encode_battle_examples(batch)
            values = trainer.train_step(
                _tensor_dict(tf, inputs),
                tf.convert_to_tensor(targets),
            )
            metrics.append({key: float(value.numpy()) for key, value in values.items()})
        row = {"epoch": epoch + 1, **_mean_metrics(metrics)}
        if validation:
            row.update(_evaluate_battle(tf, model, validation, training.batch_size))
        history.append(row)
        completed_epochs.assign(epoch + 1)
        _save_epoch_weights(model, output, epoch + 1, manager.checkpoints)
        manager.save(checkpoint_number=epoch + 1)
        _save_history(output, history)
    model.save_weights(output / "battle.weights.h5")
    return {
        "epochs": int(completed_epochs.numpy()),
        "restored_from_epoch": restored_epoch,
        "restored_checkpoint": restored_checkpoint,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "checkpoint": manager.latest_checkpoint,
        "weights": str(output / "battle.weights.h5"),
        "last_metrics": history[-1] if history else {},
    }


def _evaluate_battle(tf, model, examples, batch_size: int) -> dict[str, float]:
    losses: list[float] = []
    accuracies: list[float] = []
    for start in range(0, len(examples), batch_size):
        inputs, targets = encode_battle_examples(examples[start : start + batch_size])
        outputs = model(_tensor_dict(tf, inputs), training=False)
        losses.append(
            float(
                tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        targets, outputs["logits"], from_logits=True
                    )
                ).numpy()
            )
        )
        accuracies.append(
            float(
                tf.reduce_mean(
                    tf.cast(
                        tf.equal(tf.argmax(targets, -1), tf.argmax(outputs["logits"], -1)),
                        tf.float32,
                    )
                ).numpy()
            )
        )
    return {
        "validation_loss": sum(losses) / len(losses),
        "validation_accuracy": sum(accuracies) / len(accuracies),
    }


def _policy_batch(tf, decisions: list[ArenaDecision], max_actions: int):
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to train policies") from error
    encoded = encode_states(
        [decision.state for decision in decisions],
        [decision.actions for decision in decisions],
        max_actions=max_actions,
    )
    policy = np.zeros((len(decisions), max_actions), dtype=np.float32)
    for index, decision in enumerate(decisions):
        if len(decision.search_policy) != len(decision.actions):
            raise ValueError("policy target length does not match legal actions")
        total = sum(decision.search_policy)
        if total <= 0:
            raise ValueError("policy target has no probability mass")
        policy[index, : len(decision.actions)] = np.asarray(decision.search_policy) / total
    targets = {
        "search_policy": policy,
        "run_value": np.asarray([decision.run_value for decision in decisions], np.float32),
        "next_battle": np.asarray([decision.next_battle for decision in decisions], np.float32),
        "expected_wins": np.asarray(
            [decision.expected_wins for decision in decisions], np.float32
        ),
    }
    return _tensor_dict(tf, encoded.as_dict()), _tensor_dict(tf, targets)


def train_policy_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    validation_path: str | Path | None = None,
    training_config: TrainingConfig | None = None,
    model_config: ModelConfig | None = None,
    resume: bool = True,
) -> dict[str, object]:
    tf = _tensorflow()
    training = training_config or TrainingConfig()
    model_settings = model_config or ModelConfig()
    tf.keras.utils.set_random_seed(training.seed)
    train = read_arena_decisions(dataset_path)
    validation = read_arena_decisions(validation_path) if validation_path else []
    if not train:
        raise ValueError("policy training dataset is empty")

    output = Path(output_dir)
    _write_run_metadata(
        output,
        kind="policy-value",
        training=training,
        model=model_settings,
    )
    model = PolicyValueModel(model_settings)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        clipnorm=5.0,
    )
    trainer = PolicyValueTrainer(model, optimizer)
    sample_inputs, _ = _policy_batch(tf, train[:1], training.max_actions)
    model(sample_inputs, training=False)
    optimizer.build(model.trainable_variables)
    completed_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=3)
    restored_epoch, restored_checkpoint = _restore_training_checkpoint(
        tf,
        model=model,
        optimizer=optimizer,
        completed_epochs=completed_epochs,
        manager=manager,
        output=output,
        resume=resume,
    )

    rng = random.Random(training.seed + int(completed_epochs.numpy()))
    history = _read_history(output)
    for epoch in range(int(completed_epochs.numpy()), training.epochs):
        metrics: list[dict[str, float]] = []
        for batch in _batches(train, training.batch_size, rng):
            inputs, targets = _policy_batch(tf, batch, training.max_actions)
            values = trainer.train_step(inputs, targets)
            metrics.append({key: float(value.numpy()) for key, value in values.items()})
        row = {"epoch": epoch + 1, **_mean_metrics(metrics)}
        if validation:
            row.update(
                _evaluate_policy(
                    tf,
                    model,
                    validation,
                    training.batch_size,
                    training.max_actions,
                )
            )
        history.append(row)
        completed_epochs.assign(epoch + 1)
        _save_epoch_weights(model, output, epoch + 1, manager.checkpoints)
        manager.save(checkpoint_number=epoch + 1)
        _save_history(output, history)
    model.save_weights(output / "policy.weights.h5")
    return {
        "epochs": int(completed_epochs.numpy()),
        "restored_from_epoch": restored_epoch,
        "restored_checkpoint": restored_checkpoint,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "checkpoint": manager.latest_checkpoint,
        "weights": str(output / "policy.weights.h5"),
        "last_metrics": history[-1] if history else {},
    }


def _evaluate_policy(tf, model, examples, batch_size: int, max_actions: int):
    values = []
    for start in range(0, len(examples), batch_size):
        inputs, targets = _policy_batch(tf, examples[start : start + batch_size], max_actions)
        outputs = model(inputs, training=False)
        policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=targets["search_policy"], logits=outputs["policy_logits"]
            )
        )
        value_loss = tf.reduce_mean(
            tf.keras.losses.huber(targets["run_value"], outputs["value"])
        )
        values.append(float((policy_loss + value_loss).numpy()))
    return {"validation_loss": sum(values) / len(values)}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def _read_history(output: Path) -> list[dict[str, float]]:
    path = output / "history.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save_history(output: Path, history: list[dict[str, float]]) -> None:
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
