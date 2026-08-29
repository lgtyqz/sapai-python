from __future__ import annotations

from dataclasses import asdict, dataclass

try:  # TensorFlow is optional for simulator-only installations.
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - exercised without ML extra
    tf = None

MODEL_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id_buckets: int = 2048
    perk_buckets: int = 256
    entity_types: int = 8
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 6
    ff_dim: int = 640
    dropout: float = 0.1
    action_kinds: int = 16
    action_positions: int = 8


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="sapai")
    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, config: ModelConfig, **kwargs):
            super().__init__(**kwargs)
            self.config = config
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=config.num_heads,
                key_dim=config.d_model // config.num_heads,
                dropout=config.dropout,
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(config.ff_dim, activation="gelu"),
                    tf.keras.layers.Dropout(config.dropout),
                    tf.keras.layers.Dense(config.d_model),
                ]
            )
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.dropout = tf.keras.layers.Dropout(config.dropout)
            self.supports_masking = True

        def call(self, inputs, mask=None, training=False):
            attention_mask = None
            if mask is not None:
                attention_mask = mask[:, tf.newaxis, :]
            attended = self.attention(
                inputs,
                inputs,
                attention_mask=attention_mask,
                training=training,
            )
            values = self.norm1(inputs + self.dropout(attended, training=training))
            return self.norm2(values + self.ffn(values, training=training))

        def get_config(self):
            return {**super().get_config(), "config": asdict(self.config)}

    @tf.keras.utils.register_keras_serializable(package="sapai")
    class EntityEncoder(tf.keras.layers.Layer):
        def __init__(self, config: ModelConfig, **kwargs):
            super().__init__(**kwargs)
            self.config = config
            self.id_embedding = tf.keras.layers.Embedding(config.id_buckets, config.d_model)
            self.type_embedding = tf.keras.layers.Embedding(config.entity_types, config.d_model)
            self.perk_embedding = tf.keras.layers.Embedding(config.perk_buckets, config.d_model)
            self.numeric = tf.keras.layers.Dense(config.d_model)
            self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        def call(self, inputs, training=False):
            mask = tf.cast(inputs["entity_mask"], tf.bool)
            values = (
                self.id_embedding(inputs["entity_ids"])
                + self.type_embedding(inputs["entity_types"])
                + self.perk_embedding(inputs["perk_ids"])
                + self.numeric(inputs["entity_features"])
            )
            for block in self.blocks:
                values = block(values, mask=mask, training=training)
            values = self.norm(values)
            float_mask = tf.cast(mask[..., tf.newaxis], values.dtype)
            pooled = tf.reduce_sum(values * float_mask, axis=1) / tf.maximum(
                tf.reduce_sum(float_mask, axis=1), 1.0
            )
            return values, pooled

    @tf.keras.utils.register_keras_serializable(package="sapai")
    class PolicyValueModel(tf.keras.Model):
        """Entity transformer with a legal-action scorer and scalar value head."""

        def __init__(self, config: ModelConfig | None = None, **kwargs):
            super().__init__(**kwargs)
            self.config_object = config or ModelConfig()
            config = self.config_object
            self.encoder = EntityEncoder(config)
            self.kind_embedding = tf.keras.layers.Embedding(config.action_kinds, 48)
            self.source_embedding = tf.keras.layers.Embedding(config.action_positions, 24)
            self.target_embedding = tf.keras.layers.Embedding(config.action_positions, 24)
            self.order_embedding = tf.keras.layers.Embedding(config.action_positions, 16)
            self.action_projection = tf.keras.layers.Dense(config.d_model, activation="gelu")
            self.policy_hidden = tf.keras.layers.Dense(config.d_model, activation="gelu")
            self.policy_output = tf.keras.layers.Dense(1)
            self.value = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(config.d_model, activation="gelu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            self.next_battle_after_policy = tf.keras.layers.Dense(3, activation="softmax")
            self.expected_trophies = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, inputs, training=False):
            entities, state = self.encoder(inputs, training=training)
            source_entities = tf.gather(
                entities,
                inputs["action_source_entities"],
                batch_dims=1,
            )
            target_entities = tf.gather(
                entities,
                inputs["action_target_entities"],
                batch_dims=1,
            )
            order_entities = tf.gather(
                entities,
                inputs["action_order_entities"],
                batch_dims=1,
            )
            rank_weights = tf.cast(
                tf.reshape(tf.range(1, 6), [1, 1, 5, 1]),
                order_entities.dtype,
            )
            order_context = tf.reduce_sum(order_entities * rank_weights, axis=2) / 15.0
            action = tf.concat(
                [
                    self.kind_embedding(inputs["action_kinds"]),
                    self.source_embedding(inputs["action_sources"]),
                    self.target_embedding(inputs["action_targets"]),
                    tf.reshape(
                        self.order_embedding(inputs["action_orders"]),
                        [tf.shape(inputs["action_orders"])[0], -1, 5 * 16],
                    ),
                    source_entities,
                    target_entities,
                    order_context,
                ],
                axis=-1,
            )
            action = self.action_projection(action)
            repeated_state = tf.repeat(state[:, tf.newaxis, :], tf.shape(action)[1], axis=1)
            logits = self.policy_output(
                self.policy_hidden(tf.concat([repeated_state, action], axis=-1))
            )[..., 0]
            mask = tf.cast(inputs["action_mask"], tf.bool)
            logits = tf.where(mask, logits, tf.cast(-1e9, logits.dtype))
            return {
                "policy_logits": logits,
                "value": self.value(state)[..., 0],
                "next_battle_after_policy": self.next_battle_after_policy(state),
                "expected_trophies": self.expected_trophies(state)[..., 0],
            }

    @tf.keras.utils.register_keras_serializable(package="sapai")
    class BattleModel(tf.keras.Model):
        """Antisymmetry-aware W/D/L evaluator for a pair of encoded teams."""

        def __init__(self, config: ModelConfig | None = None, **kwargs):
            super().__init__(**kwargs)
            self.config_object = config or ModelConfig(num_layers=3)
            self.encoder = EntityEncoder(self.config_object)
            self.head = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(self.config_object.d_model, activation="gelu"),
                    tf.keras.layers.Dropout(self.config_object.dropout),
                    tf.keras.layers.Dense(3),
                ]
            )

        def call(self, inputs, training=False):
            player_inputs = {
                key.removeprefix("player_"): value
                for key, value in inputs.items()
                if key.startswith("player_")
            }
            opponent_inputs = {
                key.removeprefix("opponent_"): value
                for key, value in inputs.items()
                if key.startswith("opponent_")
            }
            _, player = self.encoder(player_inputs, training=training)
            _, opponent = self.encoder(opponent_inputs, training=training)
            comparison = tf.concat(
                [player, opponent, player - opponent, player * opponent], axis=-1
            )
            logits = self.head(comparison, training=training)
            return {"logits": logits, "probabilities": tf.nn.softmax(logits)}

    class PolicyValueTrainer:
        def __init__(self, model: PolicyValueModel, optimizer=None):
            self.model = model
            self.optimizer = optimizer or tf.keras.optimizers.AdamW(
                learning_rate=3e-4, weight_decay=1e-4, clipnorm=5.0
            )

        def train_step(self, inputs, targets):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                policy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=targets["search_policy"], logits=outputs["policy_logits"]
                    )
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
                trophies_loss = tf.reduce_mean(
                    tf.keras.losses.huber(
                        targets["expected_trophies"], outputs["expected_trophies"]
                    )
                )
                total = policy_loss + value_loss + 0.25 * battle_loss + 0.1 * trophies_loss
                if self.model.losses:
                    total += tf.add_n(self.model.losses)
            gradients = tape.gradient(total, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return {
                "loss": total,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "battle_loss": battle_loss,
                "expected_trophies_loss": trophies_loss,
            }

else:

    class _TensorFlowRequired:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow is not installed; install the 'ml' extra")

    TransformerBlock = _TensorFlowRequired
    EntityEncoder = _TensorFlowRequired
    PolicyValueModel = _TensorFlowRequired
    BattleModel = _TensorFlowRequired
    PolicyValueTrainer = _TensorFlowRequired
