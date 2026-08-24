from __future__ import annotations

from sapai.ml.encoding import encode_states
from sapai.ml.models import PolicyValueModel
from sapai.sim.catalog import Catalog
from sapai.sim.shop import ShopEnvironment


def run_model_smoke(catalog: Catalog) -> dict[str, tuple[int, ...]]:
    import tensorflow as tf

    environment = ShopEnvironment(catalog)
    state = environment.reset(seed=7)
    actions = environment.legal_actions(state)
    encoded = encode_states([state], [actions])
    inputs = {key: tf.convert_to_tensor(value) for key, value in encoded.as_dict().items()}
    model = PolicyValueModel()
    outputs = model(inputs, training=False)
    return {key: tuple(value.shape) for key, value in outputs.items()}
