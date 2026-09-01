import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError:
    np = None
    tf = None

from sapai.ml.encoding import encode_states, encode_teams
from sapai.ml.models import BattleModel, ModelConfig, PolicyValueModel, PolicyValueTrainer
from sapai.ml.pipelines import _restore_training_checkpoint
from sapai.ml.training import BattleTrainer
from sapai.sim.actions import ActionKind
from sapai.sim.models import Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
from tests.helpers import catalog


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class TensorFlowModelTest(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(d_model=48, num_layers=1, num_heads=3, ff_dim=64)
        self.environment = ShopEnvironment(catalog())
        self.state = self.environment.reset(seed=2)
        self.actions = self.environment.legal_actions(self.state)

    def test_policy_value_forward_and_train_step(self):
        encoded = encode_states([self.state], [self.actions])
        inputs = {key: tf.convert_to_tensor(value) for key, value in encoded.as_dict().items()}
        model = PolicyValueModel(self.config)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, 256))
        self.assertEqual(tuple(outputs["value"].shape), (1,))
        self.assertTrue(0.0 <= float(outputs["value"][0].numpy()) <= 1.0)
        self.assertIn("next_battle_after_policy", outputs)
        self.assertIn("expected_trophies", outputs)
        policy = np.zeros((1, 256), dtype=np.float32)
        policy[0, 0] = 1.0
        targets = {
            "search_policy": policy,
            "run_value": np.array([0.5], dtype=np.float32),
            "next_battle_after_policy": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "expected_trophies": np.array([0.4], dtype=np.float32),
        }
        losses = PolicyValueTrainer(model).train_step(inputs, targets)
        self.assertTrue(float(losses["loss"].numpy()) > 0)

    def test_actions_gather_their_actual_source_target_and_order_entities(self):
        self.state.team = Team.from_pets(
            [catalog().pet_by_name("Ant").create(), catalog().pet_by_name("Fish").create()]
        )
        actions = self.environment.legal_actions(self.state)
        encoded = encode_states([self.state], [actions])
        buy = next(action for action in actions if action.kind is ActionKind.BUY_PET)
        food = next(
            action
            for action in actions
            if action.kind is ActionKind.BUY_FOOD and action.target >= 0
        )
        sell = next(action for action in actions if action.kind is ActionKind.SELL_PET)
        reorder = next(action for action in actions if action.kind is ActionKind.REORDER)
        indexes = {action: actions.index(action) for action in (buy, food, sell, reorder)}

        self.assertEqual(
            encoded.action_source_entities[0, indexes[buy]], 5 + buy.source
        )
        self.assertEqual(encoded.action_target_entities[0, indexes[buy]], buy.target)
        self.assertEqual(
            encoded.action_source_entities[0, indexes[food]], 10 + food.source
        )
        self.assertEqual(encoded.action_target_entities[0, indexes[food]], food.target)
        self.assertEqual(encoded.action_source_entities[0, indexes[sell]], sell.source)
        self.assertEqual(
            tuple(encoded.action_order_entities[0, indexes[reorder], : len(reorder.order)]),
            reorder.order,
        )

    def test_level_reward_shop_overflow_is_encoded_without_position_overflow(self):
        ant = catalog().pet_by_name("Ant")
        self.state.shop = Shop(
            pets=[ShopPet(ant.create(instance_id=index)) for index in range(7)],
            foods=[],
        )
        actions = self.environment.legal_actions(self.state)
        freeze = next(
            action
            for action in actions
            if action.kind is ActionKind.FREEZE_PET and action.source == 6
        )

        encoded = encode_states([self.state], [actions])
        freeze_index = actions.index(freeze)
        inputs = {
            key: tf.convert_to_tensor(value) for key, value in encoded.as_dict().items()
        }
        outputs = PolicyValueModel(self.config)(inputs)

        self.assertEqual(encoded.entity_ids.shape[1], 15)
        self.assertEqual(encoded.action_source_entities[0, freeze_index], 11)
        self.assertEqual(encoded.action_sources[0, freeze_index], 8)
        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, 256))

    def test_battle_model_forward(self):
        ant = catalog().pet_by_name("Ant").create()
        fish = catalog().pet_by_name("Fish").create()
        player = encode_teams([Team.from_pets([ant])])
        opponent = encode_teams([Team.from_pets([fish])])
        inputs = {
            **{f"player_{key}": tf.convert_to_tensor(value) for key, value in player.items()},
            **{f"opponent_{key}": tf.convert_to_tensor(value) for key, value in opponent.items()},
        }
        outputs = BattleModel(self.config)(inputs)
        self.assertEqual(tuple(outputs["probabilities"].shape), (1, 3))
        self.assertAlmostEqual(
            float(tf.reduce_sum(outputs["probabilities"]).numpy()), 1.0, places=5
        )

    def test_legacy_checkpoint_restores_model_and_optimizer_before_next_epoch(self):
        ant = catalog().pet_by_name("Ant").create()
        fish = catalog().pet_by_name("Fish").create()
        player = encode_teams([Team.from_pets([ant])])
        opponent = encode_teams([Team.from_pets([fish])])
        inputs = {
            **{f"player_{key}": tf.convert_to_tensor(value) for key, value in player.items()},
            **{
                f"opponent_{key}": tf.convert_to_tensor(value)
                for key, value in opponent.items()
            },
        }
        targets = tf.convert_to_tensor([[1.0, 0.0, 0.0]])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source_model = BattleModel(self.config)
            source_model(inputs, training=False)
            source_optimizer = tf.keras.optimizers.AdamW(3e-4)
            BattleTrainer(source_model, source_optimizer).train_step(inputs, targets)
            source_epoch = tf.Variable(1, dtype=tf.int64, trainable=False)
            source_checkpoint = tf.train.Checkpoint(
                model=source_model,
                optimizer=source_optimizer,
                completed_epochs=source_epoch,
            )
            source_manager = tf.train.CheckpointManager(
                source_checkpoint,
                str(output / "checkpoints"),
                max_to_keep=3,
            )
            source_manager.save(checkpoint_number=1)
            expected = [value.numpy().copy() for value in source_model.trainable_variables]

            target_model = BattleModel(self.config)
            target_model(inputs, training=False)
            target_optimizer = tf.keras.optimizers.AdamW(3e-4)
            target_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
            target_checkpoint = tf.train.Checkpoint(
                model=target_model,
                optimizer=target_optimizer,
                completed_epochs=target_epoch,
            )
            target_manager = tf.train.CheckpointManager(
                target_checkpoint,
                str(output / "checkpoints"),
                max_to_keep=3,
            )
            restored_epoch, restored_path = _restore_training_checkpoint(
                tf,
                model=target_model,
                optimizer=target_optimizer,
                completed_epochs=target_epoch,
                manager=target_manager,
                output=output,
                resume=True,
            )

        self.assertEqual(restored_epoch, 1)
        self.assertIsNotNone(restored_path)
        self.assertEqual(int(target_optimizer.iterations.numpy()), 1)
        for expected_value, restored_value in zip(
            expected, target_model.trainable_variables, strict=True
        ):
            np.testing.assert_allclose(restored_value.numpy(), expected_value)


if __name__ == "__main__":
    unittest.main()
