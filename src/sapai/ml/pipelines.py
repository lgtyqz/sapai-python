"""Checkpointed TensorFlow training loops used locally and in Colab."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sapai.data.datasets import read_battle_examples
from sapai.ml.encoding import encode_states
from sapai.ml.models import (
    MODEL_SCHEMA_VERSION,
    BattleModel,
    ModelConfig,
    PolicyValueModel,
    PolicyValueTrainer,
)
from sapai.ml.training import BattleTrainer, encode_battle_examples
from sapai.rewards import POLICY_TARGET_SCHEMA, VALUE_OBJECTIVE
from sapai.training.arena import ArenaDecision, read_arena_decisions

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    max_actions: int = 256
    deterministic: bool = True
    warmup_epochs: int = 1
    cosine_decay_epochs: int = 10
    minimum_learning_rate: float = 3e-5


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


def _epoch_learning_rate(training: TrainingConfig, epoch: int) -> float:
    if training.warmup_epochs > 0 and epoch < training.warmup_epochs:
        return training.learning_rate * (epoch + 1) / training.warmup_epochs
    decay_epochs = max(1, training.cosine_decay_epochs)
    phase = (epoch - training.warmup_epochs) % decay_epochs
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase / decay_epochs))
    return training.minimum_learning_rate + (
        training.learning_rate - training.minimum_learning_rate
    ) * cosine


def _write_run_metadata(
    output: Path,
    *,
    kind: str,
    training: TrainingConfig,
    model: ModelConfig,
    dataset_paths: list[Path],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    training_values = asdict(training)
    immutable = {
        "format": "sapai-model-run-v2",
        "kind": kind,
        "model_schema": MODEL_SCHEMA_VERSION,
        "target_schema": POLICY_TARGET_SCHEMA if kind == "policy-value" else None,
        "objective": VALUE_OBJECTIVE if kind == "policy-value" else "battle-wdl",
        "model": asdict(model),
        "training_contract": {
            key: value
            for key, value in training_values.items()
            if key not in {"epochs", "seed"}
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    manifest_path = output / "run-manifest.json"
    legacy_training_seed = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_immutable = {key: existing.get(key) for key in immutable}
        existing_training_contract = existing_immutable.get("training_contract")
        if isinstance(existing_training_contract, dict):
            legacy_training_seed = existing_training_contract.get("seed")
            existing_immutable["training_contract"] = {
                key: value
                for key, value in existing_training_contract.items()
                if key != "seed"
            }
        if existing_immutable != immutable:
            raise ValueError(
                f"model/checkpoint contract changed in {output}; use a new output directory"
            )
        manifest = existing
        manifest["training_contract"] = immutable["training_contract"]
    else:
        if (output / "checkpoints").exists() or (output / "config.json").exists():
            raise ValueError(
                f"legacy unversioned checkpoints found in {output}; use a new v4 output directory"
            )
        manifest = {**immutable, "datasets": []}
    datasets = manifest.setdefault("datasets", [])
    training_seeds = manifest.setdefault("training_seeds", [])
    if legacy_training_seed is not None and legacy_training_seed not in training_seeds:
        training_seeds.append(legacy_training_seed)
    if legacy_training_seed is not None:
        for entry in datasets:
            entry.setdefault("seed", legacy_training_seed)
    if training.seed not in training_seeds:
        training_seeds.append(training.seed)
    for path in dataset_paths:
        entry = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "target_epochs": training.epochs,
            "seed": training.seed,
        }
        if entry not in datasets:
            datasets.append(entry)
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(
        output / "config.json",
        {"kind": kind, "training": training_values, "model": asdict(model)},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_final_weights(model, destination: Path) -> None:
    temporary = destination.with_name(destination.stem + ".tmp.weights.h5")
    model.save_weights(temporary)
    temporary.replace(destination)


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
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    tf = _tensorflow()
    training = training_config or TrainingConfig()
    model_settings = model_config or ModelConfig(num_layers=3)
    if training.deterministic:
        tf.config.experimental.enable_op_determinism()
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
        dataset_paths=[Path(dataset_dir) / "train.jsonl"],
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
    compiled_train_step = tf.function(trainer.train_step, reduce_retracing=True)
    if progress is not None:
        progress(
            "training_started",
            {
                "completed": restored_epoch,
                "total": max(training.epochs, restored_epoch),
                "train_examples": len(train),
                "validation_examples": len(validation),
                "batches_per_epoch": (
                    len(train) + training.batch_size - 1
                )
                // training.batch_size,
                "restored_checkpoint": restored_checkpoint,
            },
        )

    history = _read_history(output)
    for epoch in range(int(completed_epochs.numpy()), training.epochs):
        epoch_seed = training.seed + epoch * 1_000_003
        tf.keras.utils.set_random_seed(epoch_seed)
        rng = random.Random(epoch_seed)
        learning_rate = _epoch_learning_rate(training, epoch)
        optimizer.learning_rate.assign(learning_rate)
        metrics: list[dict[str, float]] = []
        for batch in _batches(train, training.batch_size, rng):
            inputs, targets = encode_battle_examples(batch)
            values = compiled_train_step(
                _tensor_dict(tf, inputs),
                tf.convert_to_tensor(targets),
            )
            metrics.append({key: float(value.numpy()) for key, value in values.items()})
        row = {"epoch": epoch + 1, "learning_rate": learning_rate, **_mean_metrics(metrics)}
        if validation:
            row.update(_evaluate_battle(tf, model, validation, training.batch_size))
        history.append(row)
        completed_epochs.assign(epoch + 1)
        _save_epoch_weights(model, output, epoch + 1, manager.checkpoints)
        manager.save(checkpoint_number=epoch + 1)
        _save_history(output, history)
        if progress is not None:
            progress(
                "epoch_completed",
                {
                    "completed": epoch + 1,
                    "total": training.epochs,
                    "metrics": row,
                    "checkpoint": manager.latest_checkpoint,
                },
            )
    _save_final_weights(model, output / "battle.weights.h5")
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


def _policy_batch(
    tf,
    decisions: list[ArenaDecision],
    max_actions: int,
    shop_pet_capacity: int | None = None,
):
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to train policies") from error
    encoded = encode_states(
        [decision.state for decision in decisions],
        [decision.actions for decision in decisions],
        max_actions=max_actions,
        shop_pet_capacity=shop_pet_capacity,
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
        "next_battle_after_policy": np.asarray(
            [decision.next_battle_after_policy for decision in decisions], np.float32
        ),
        "expected_trophies": np.asarray(
            [decision.expected_trophies for decision in decisions], np.float32
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
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    tf = _tensorflow()
    training = training_config or TrainingConfig()
    model_settings = model_config or ModelConfig()
    if training.deterministic:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(training.seed)
    train = read_arena_decisions(dataset_path)
    validation = read_arena_decisions(validation_path) if validation_path else []
    if not train:
        raise ValueError("policy training dataset is empty")
    shop_pet_capacity = max(
        5,
        max(len(decision.state.shop.pets) for decision in train + validation),
    )

    output = Path(output_dir)
    _write_run_metadata(
        output,
        kind="policy-value",
        training=training,
        model=model_settings,
        dataset_paths=[Path(dataset_path)]
        + ([Path(validation_path)] if validation_path else []),
    )
    model = PolicyValueModel(model_settings)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        clipnorm=5.0,
    )
    trainer = PolicyValueTrainer(model, optimizer)
    sample_inputs, _ = _policy_batch(
        tf,
        train[:1],
        training.max_actions,
        shop_pet_capacity,
    )
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
    compiled_train_step = tf.function(trainer.train_step, reduce_retracing=True)
    if progress is not None:
        progress(
            "training_started",
            {
                "completed": restored_epoch,
                "total": max(training.epochs, restored_epoch),
                "train_examples": len(train),
                "validation_examples": len(validation),
                "batches_per_epoch": (
                    len(train) + training.batch_size - 1
                )
                // training.batch_size,
                "restored_checkpoint": restored_checkpoint,
            },
        )

    history = _read_history(output)
    for epoch in range(int(completed_epochs.numpy()), training.epochs):
        epoch_seed = training.seed + epoch * 1_000_003
        tf.keras.utils.set_random_seed(epoch_seed)
        rng = random.Random(epoch_seed)
        learning_rate = _epoch_learning_rate(training, epoch)
        optimizer.learning_rate.assign(learning_rate)
        metrics: list[dict[str, float]] = []
        for batch in _batches(train, training.batch_size, rng):
            inputs, targets = _policy_batch(
                tf,
                batch,
                training.max_actions,
                shop_pet_capacity,
            )
            values = compiled_train_step(inputs, targets)
            metrics.append({key: float(value.numpy()) for key, value in values.items()})
        row = {"epoch": epoch + 1, "learning_rate": learning_rate, **_mean_metrics(metrics)}
        if validation:
            row.update(
                _evaluate_policy(
                    tf,
                    model,
                    validation,
                    training.batch_size,
                    training.max_actions,
                    shop_pet_capacity,
                )
            )
        history.append(row)
        completed_epochs.assign(epoch + 1)
        _save_epoch_weights(model, output, epoch + 1, manager.checkpoints)
        manager.save(checkpoint_number=epoch + 1)
        _save_history(output, history)
        if progress is not None:
            progress(
                "epoch_completed",
                {
                    "completed": epoch + 1,
                    "total": training.epochs,
                    "metrics": row,
                    "checkpoint": manager.latest_checkpoint,
                },
            )
    _save_final_weights(model, output / "policy.weights.h5")
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


def _evaluate_policy(
    tf,
    model,
    examples,
    batch_size: int,
    max_actions: int,
    shop_pet_capacity: int,
):
    rows: list[dict[str, float]] = []
    for start in range(0, len(examples), batch_size):
        inputs, targets = _policy_batch(
            tf,
            examples[start : start + batch_size],
            max_actions,
            shop_pet_capacity,
        )
        outputs = model(inputs, training=False)
        policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=targets["search_policy"], logits=outputs["policy_logits"]
            )
        )
        action_kind_membership = tf.one_hot(
            inputs["action_kinds"],
            depth=model.config_object.action_kinds,
            dtype=tf.float32,
        )
        target_kind_policy = tf.einsum(
            "ba,bak->bk",
            targets["search_policy"],
            action_kind_membership,
        )
        predicted_kind_policy = tf.einsum(
            "ba,bak->bk",
            tf.nn.softmax(outputs["policy_logits"]),
            action_kind_membership,
        )
        value_loss = tf.reduce_mean(
            tf.keras.losses.huber(targets["run_value"], outputs["value"])
        )
        battle_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                targets["next_battle_after_policy"],
                outputs["next_battle_after_policy"],
            )
        )
        trophy_loss = tf.reduce_mean(
            tf.keras.losses.huber(
                targets["expected_trophies"], outputs["expected_trophies"]
            )
        )
        rows.append(
            {
                "validation_policy_loss": float(policy_loss.numpy()),
                "validation_policy_accuracy": float(
                    tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.argmax(target_kind_policy, axis=-1),
                                tf.argmax(predicted_kind_policy, axis=-1),
                            ),
                            tf.float32,
                        )
                    ).numpy()
                ),
                "validation_concrete_action_accuracy": float(
                    tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.argmax(targets["search_policy"], axis=-1),
                                tf.argmax(outputs["policy_logits"], axis=-1),
                            ),
                            tf.float32,
                        )
                    ).numpy()
                ),
                "validation_value_loss": float(value_loss.numpy()),
                "validation_value_brier": float(
                    tf.reduce_mean(
                        tf.square(targets["run_value"] - outputs["value"])
                    ).numpy()
                ),
                "validation_battle_loss": float(battle_loss.numpy()),
                "validation_battle_accuracy": float(
                    tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.argmax(targets["next_battle_after_policy"], axis=-1),
                                tf.argmax(outputs["next_battle_after_policy"], axis=-1),
                            ),
                            tf.float32,
                        )
                    ).numpy()
                ),
                "validation_trophy_loss": float(trophy_loss.numpy()),
                "validation_trophy_mae": float(
                    tf.reduce_mean(
                        tf.abs(
                            targets["expected_trophies"] - outputs["expected_trophies"]
                        )
                    ).numpy()
                ),
            }
        )
    return _mean_metrics(rows)


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
    _atomic_write_json(output / "history.json", history)
