import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from deepcave.evaluators.sampling_density import joint, marginal


def test_marginal_histogram_and_uniform_baseline():
    hyperparameter = UniformFloatHyperparameter("x", lower=0.0, upper=1.0)

    result = marginal([0.05, 0.06, 0.07, 0.8], hyperparameter, bins=20)

    assert result["mode"] == "single"
    assert len(result["grid"]) == 20
    assert len(result["pdf"]) == len(result["uniform"])
    assert result["pdf"][1] > result["pdf"][-1]
    assert result["uniform"][0] == 1.0


def test_categorical_marginal_uses_frequency_mass():
    hyperparameter = CategoricalHyperparameter("choice", choices=["a", "b", "c"])

    result = marginal([0.0, 0.0, 0.5, 0.5, 0.5], hyperparameter)

    assert result["is_categorical"] is True
    assert result["categories"] == ["a", "b", "c"]
    assert np.allclose(result["pdf"], [0.4, 0.6, 0.0])
    assert np.allclose(result["uniform"], [1 / 3] * 3)


def test_joint_histogram_has_expected_shape_and_count():
    hp1 = UniformFloatHyperparameter("x", lower=0.0, upper=1.0)
    hp2 = UniformFloatHyperparameter("y", lower=0.0, upper=1.0)

    result = joint([0.1, 0.2, 0.9], [0.1, 0.2, 0.8], hp1, hp2, bins=4)

    assert len(result["x_edges"]) == 5
    assert len(result["y_edges"]) == 5
    assert np.asarray(result["counts"]).shape == (4, 4)
    assert int(np.asarray(result["counts"]).sum()) == 3
    assert len(result["points"]) == 3


def test_plugin_is_registered_and_importable():
    from deepcave.config import Config
    from deepcave.plugins.hyperparameter.sampling_density import SamplingDensity

    plugin = SamplingDensity()
    assert plugin.id == "sampling_density"
    assert any(
        isinstance(candidate, SamplingDensity)
        for candidate in Config().PLUGINS["Hyperparameter Analysis"]
    )