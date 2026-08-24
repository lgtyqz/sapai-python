import unittest

try:
    import numpy as np
    import tensorflow as tf
except ModuleNotFoundError:
    np = None
    tf = None

from sapai.ml.encoding import encode_states, encode_teams
from sapai.ml.models import BattleModel, ModelConfig, PolicyValueModel, PolicyValueTrainer
from sapai.sim.models import Team
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
        policy = np.zeros((1, 256), dtype=np.float32)
        policy[0, 0] = 1.0
        targets = {
            "search_policy": policy,
            "run_value": np.array([0.5], dtype=np.float32),
            "next_battle": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "expected_wins": np.array([4.0], dtype=np.float32),
        }
        losses = PolicyValueTrainer(model).train_step(inputs, targets)
        self.assertTrue(float(losses["loss"].numpy()) > 0)

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


if __name__ == "__main__":
    unittest.main()
