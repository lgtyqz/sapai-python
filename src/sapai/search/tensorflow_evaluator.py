from __future__ import annotations

from sapai.ml.encoding import encode_states
from sapai.sim.actions import Action
from sapai.sim.models import RunState


class TensorFlowEvaluator:
    """Adapter from ``PolicyValueModel`` to the search ``Evaluator`` protocol."""

    def __init__(self, model, *, max_actions: int = 256):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("install the 'ml' extra to evaluate TensorFlow models") from error
        self.model = model
        self.max_actions = max_actions
        self._predict = tf.function(
            lambda inputs: self.model(inputs, training=False),
            reduce_retracing=True,
        )

    def evaluate(self, state: RunState, actions: list[Action]) -> tuple[list[float], float]:
        results = self.evaluate_many([state], [actions])
        return results[0]

    def evaluate_many(
        self,
        states: list[RunState],
        actions: list[list[Action]],
    ) -> list[tuple[list[float], float]]:
        """Evaluate independent leaves in one TensorFlow call."""

        try:
            import tensorflow as tf
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("install the 'ml' extra to evaluate TensorFlow models") from error
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal length")
        encoded = encode_states(states, actions, max_actions=self.max_actions)
        inputs = {key: tf.convert_to_tensor(value) for key, value in encoded.as_dict().items()}
        outputs = self._predict(inputs)
        results = []
        for index, legal in enumerate(actions):
            probabilities = tf.nn.softmax(
                outputs["policy_logits"][index, : len(legal)]
            ).numpy()
            results.append((probabilities.tolist(), float(outputs["value"][index].numpy())))
        return results
