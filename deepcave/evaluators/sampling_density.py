"""Sampling density calculations for encoded configurations."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)

from deepcave.runs import AbstractRun, Objective, Status


def is_discrete(hyperparameter: Hyperparameter) -> bool:
    """Return whether a hyperparameter should be represented by discrete bins."""
    return isinstance(
        hyperparameter,
        (CategoricalHyperparameter, IntegerHyperparameter, OrdinalHyperparameter),
    )


def _encoded_data(
    run: AbstractRun,
    objective: Objective,
    budget: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load successful configurations and fold repeated seed evaluations."""
    data = run.get_encoded_data(
        objectives=objective,
        budget=budget,
        seed=seed,
        statuses=Status.SUCCESS,
        include_config_ids=True,
    )
    if data.empty:
        raise ValueError("No successful trials are available for the selected budget.")

    config_id = "config_id"
    if config_id not in data.columns:
        raise ValueError("Encoded run data does not contain configuration ids.")
    return data.groupby(config_id, as_index=False).mean(numeric_only=True)


def _values_for_hp(
    data: pd.DataFrame, hyperparameter: Hyperparameter
) -> np.ndarray:
    """Get finite encoded values for one hyperparameter."""
    values = pd.to_numeric(data[hyperparameter.name], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Hyperparameter '{hyperparameter.name}' has no finite observations.")
    return values


def _uniform_bounds(hyperparameter: Hyperparameter) -> Tuple[float, float]:
    """Return the encoded support used by DeepCAVE for one hyperparameter."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        n_choices = len(hyperparameter.choices)
        return 0.0, float(max(n_choices - 1, 0)) / float(max(n_choices - 1, 1))
    if isinstance(hyperparameter, OrdinalHyperparameter):
        n_choices = len(hyperparameter.sequence)
        return 0.0, float(max(n_choices - 1, 0)) / float(max(n_choices - 1, 1))
    if isinstance(hyperparameter, Constant):
        return 1.0, 1.0
    return float(hyperparameter.lower), float(hyperparameter.upper)


def _categories(hyperparameter: Hyperparameter) -> Tuple[List[float], List[str]]:
    """Return encoded category positions and labels."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        labels = [str(choice) for choice in hyperparameter.choices]
    elif isinstance(hyperparameter, OrdinalHyperparameter):
        labels = [str(choice) for choice in hyperparameter.sequence]
    else:
        raise TypeError("The hyperparameter is not categorical.")
    denominator = max(len(labels) - 1, 1)
    return [float(index) / denominator for index in range(len(labels))], labels


def marginal(
    values: Sequence[float],
    hyperparameter: Hyperparameter,
    bins: int = 200,
) -> Dict[str, Any]:
    """Calculate a one-dimensional encoded marginal density."""
    observations = np.asarray(values, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError(f"Hyperparameter '{hyperparameter.name}' has no finite observations.")

    if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter)):
        positions, labels = _categories(hyperparameter)
        counts = [int(np.count_nonzero(np.isclose(observations, position))) for position in positions]
        total = max(int(observations.size), 1)
        return {
            "mode": "single",
            "hp": hyperparameter.name,
            "grid": positions,
            "pdf": [float(count / total) for count in counts],
            "uniform": [float(1.0 / len(positions))] * len(positions),
            "rug": observations.tolist(),
            "categories": labels,
            "is_categorical": True,
        }

    lower, upper = _uniform_bounds(hyperparameter)
    if upper <= lower:
        return {
            "mode": "single",
            "hp": hyperparameter.name,
            "grid": [float(lower)],
            "pdf": [1.0],
            "uniform": [1.0],
            "rug": observations.tolist(),
            "categories": [],
            "is_categorical": False,
        }

    if isinstance(hyperparameter, IntegerHyperparameter):
        integer_values = np.arange(int(hyperparameter.lower), int(hyperparameter.upper) + 1)
        encoded_values = integer_values.astype(float)
        counts = [int(np.count_nonzero(np.isclose(observations, value))) for value in encoded_values]
        total = max(int(observations.size), 1)
        return {
            "mode": "single",
            "hp": hyperparameter.name,
            "grid": encoded_values.tolist(),
            "pdf": [float(count / total) for count in counts],
            "uniform": [float(1.0 / len(encoded_values))] * len(encoded_values),
            "rug": observations.tolist(),
            "categories": [str(value) for value in integer_values],
            "is_categorical": True,
        }

    n_grid = max(2, min(int(bins), 200))
    edges = np.linspace(lower, upper, n_grid + 1)
    counts, _ = np.histogram(observations, bins=edges, density=False)
    widths = np.diff(edges)
    density = counts / max(float(observations.size), 1.0) / widths
    return {
        "mode": "single",
        "hp": hyperparameter.name,
        "grid": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "pdf": density.astype(float).tolist(),
        "uniform": [float(1.0 / (upper - lower))] * n_grid,
        "rug": observations.tolist(),
        "categories": [],
        "is_categorical": False,
    }


def joint(
    x_values: Sequence[float],
    y_values: Sequence[float],
    hp1: Hyperparameter,
    hp2: Hyperparameter,
    bins: int = 80,
) -> Dict[str, Any]:
    """Calculate a two-dimensional histogram in encoded space."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        raise ValueError("The selected hyperparameter pair has no finite observations.")

    def edges(values: np.ndarray, hyperparameter: Hyperparameter) -> np.ndarray:
        lower, upper = _uniform_bounds(hyperparameter)
        if upper <= lower:
            return np.asarray([lower - 0.5, upper + 0.5])
        if is_discrete(hyperparameter):
            unique = np.unique(values)
            if unique.size > 1:
                gaps = np.diff(unique)
                inner = (unique[:-1] + unique[1:]) / 2
                return np.r_[unique[0] - gaps[0] / 2, inner, unique[-1] + gaps[-1] / 2]
        return np.linspace(lower, upper, max(2, min(int(bins), 80)) + 1)

    x_edges = edges(x, hp1)
    y_edges = edges(y, hp2)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
    return {
        "mode": "pair",
        "hp1": hp1.name,
        "hp2": hp2.name,
        "x_edges": x_edges.astype(float).tolist(),
        "y_edges": y_edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
        "points": np.column_stack((x, y)).astype(float).tolist(),
    }


def calculate(
    run: AbstractRun,
    objective: Objective,
    budget: float,
    mode: str,
    hp1_name: str,
    hp2_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate the JSON-safe density payload used by the plugin."""
    data = _encoded_data(run, objective, budget, seed=seed)
    hp1 = run.configspace[hp1_name]
    if isinstance(hp1, Constant):
        raise ValueError(f"Hyperparameter '{hp1_name}' is constant and cannot be plotted.")

    if mode == "single":
        result = marginal(_values_for_hp(data, hp1), hp1)
    elif mode == "pair":
        if hp2_name is None or hp2_name == hp1_name:
            raise ValueError("Pair mode requires two different hyperparameters.")
        hp2 = run.configspace[hp2_name]
        if isinstance(hp2, Constant):
            raise ValueError(f"Hyperparameter '{hp2_name}' is constant and cannot be plotted.")
        result = joint(
            _values_for_hp(data, hp1),
            _values_for_hp(data, hp2),
            hp1,
            hp2,
        )
    else:
        raise ValueError("Mode must be 'single' or 'pair'.")

    try:
        incumbent, _ = run.get_incumbent(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, seed=seed
        )
        encoded = run.encode_config(incumbent)
        result["incumbent"] = float(encoded[list(run.configspace.keys()).index(hp1_name)])
        if mode == "pair":
            result["incumbent"] = [
                float(encoded[list(run.configspace.keys()).index(hp1_name)]),
                float(encoded[list(run.configspace.keys()).index(hp2_name)]),
            ]
    except (RuntimeError, ValueError):
        result["incumbent"] = None
    return result