"""Sampling density calculations in the original hyperparameter space."""

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


def _raw_data(
    run: AbstractRun,
    objective: Objective,
    budget: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load successful configurations as their original hyperparameter values."""
    configs = run.get_configs(budget=budget, seed=seed, statuses=Status.SUCCESS)
    if not configs:
        raise ValueError("No successful trials are available for the selected budget.")
    return pd.DataFrame(
        [{name: config[name] for name in run.configspace.keys()} for config in configs.values()]
    )


def _values_for_hp(data: pd.DataFrame, hyperparameter: Hyperparameter) -> np.ndarray:
    """Get observed original values for one hyperparameter."""
    values = data[hyperparameter.name].to_numpy()
    if values.size == 0:
        raise ValueError(f"Hyperparameter '{hyperparameter.name}' has no finite observations.")
    return values


def _uniform_bounds(hyperparameter: Hyperparameter) -> Tuple[float, float]:
    """Return the original numeric support of a hyperparameter."""
    if isinstance(hyperparameter, Constant):
        return float(hyperparameter.value), float(hyperparameter.value)
    if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter)):
        values = (
            hyperparameter.choices
            if isinstance(hyperparameter, CategoricalHyperparameter)
            else hyperparameter.sequence
        )
        return 0.0, float(max(len(values) - 1, 0))
    return float(hyperparameter.lower), float(hyperparameter.upper)


def _categories(hyperparameter: Hyperparameter) -> Tuple[List[str], List[str]]:
    """Return original category values and labels."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        labels = [str(choice) for choice in hyperparameter.choices]
    elif isinstance(hyperparameter, OrdinalHyperparameter):
        labels = [str(choice) for choice in hyperparameter.sequence]
    else:
        raise TypeError("The hyperparameter is not categorical.")
    return labels, labels


def marginal(
    values: Sequence[float],
    hyperparameter: Hyperparameter,
    bins: int = 200,
) -> Dict[str, Any]:
    """Calculate a one-dimensional marginal density in original value space."""
    observations = np.asarray(values)
    if observations.size == 0:
        raise ValueError(f"Hyperparameter '{hyperparameter.name}' has no finite observations.")

    if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter)):
        positions, labels = _categories(hyperparameter)
        counts = [
            int(np.count_nonzero(np.asarray([str(value) for value in observations]) == position))
            for position in positions
        ]
        total = max(int(observations.size), 1)
        return {
            "hp": hyperparameter.name,
            "grid": positions,
            "pdf": [float(count / total) for count in counts],
            "uniform": [float(1.0 / len(positions))] * len(positions),
            "rug": observations.tolist(),
            "categories": labels,
            "is_categorical": True,
        }

    lower, upper = _uniform_bounds(hyperparameter)
    observations = pd.to_numeric(observations, errors="coerce")
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError(f"Hyperparameter '{hyperparameter.name}' has no finite observations.")
    if upper <= lower:
        return {
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
        counts = [int(np.count_nonzero(observations == value)) for value in integer_values]
        total = max(int(observations.size), 1)
        return {
            "hp": hyperparameter.name,
            "grid": integer_values.tolist(),
            "pdf": [float(count / total) for count in counts],
            "uniform": [float(1.0 / len(integer_values))] * len(integer_values),
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
        "hp": hyperparameter.name,
        "grid": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "pdf": density.astype(float).tolist(),
        "uniform": [float(1.0 / (upper - lower))] * n_grid,
        "rug": observations.tolist(),
        "categories": [],
        "is_categorical": False,
    }


def _distribution_summary(
    values: Sequence[float], hyperparameter: Hyperparameter
) -> Dict[str, Any]:
    """Describe concentration, edge effects, coverage, and category preference."""
    observations = np.asarray(values)
    if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter)):
        observations = observations[pd.notna(observations)]
    else:
        observations = pd.to_numeric(observations, errors="coerce")
        observations = observations[np.isfinite(observations)]
    lower, upper = _uniform_bounds(hyperparameter)
    scale = max(upper - lower, 1.0)
    numeric_observations = (
        np.asarray([_categories(hyperparameter)[0].index(str(value)) for value in observations])
        if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter))
        else observations.astype(float)
    )
    summary: Dict[str, Any] = {
        "sample_count": int(observations.size),
        "edge_fraction": float(
            np.mean(
                (numeric_observations <= lower + 0.1 * scale)
                | (numeric_observations >= upper - 0.1 * scale)
            )
        ),
    }
    if isinstance(hyperparameter, (CategoricalHyperparameter, OrdinalHyperparameter)):
        positions, labels = _categories(hyperparameter)
        counts = np.asarray(
            [
                np.count_nonzero(np.asarray([str(value) for value in observations]) == position)
                for position in positions
            ]
        )
        shares = counts / max(observations.size, 1)
        summary.update(
            category_labels=labels,
            category_shares=shares.astype(float).tolist(),
            preferred_category=labels[int(np.argmax(shares))],
            preferred_share=float(np.max(shares)),
            missing_categories=[labels[index] for index, count in enumerate(counts) if count == 0],
        )
        return summary

    histogram, _ = np.histogram(numeric_observations, bins=10, range=(lower, upper))
    occupied = int(np.count_nonzero(histogram))
    summary.update(
        max_bin_share=float(np.max(histogram) / max(observations.size, 1)),
        occupied_bin_fraction=float(occupied / 10.0),
    )
    return summary


def calculate(
    run: AbstractRun,
    objective: Objective,
    budget: float,
    hp1_name: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate the JSON-safe density payload used by the plugin."""
    data = _raw_data(run, objective, budget, seed=seed)
    hp1 = run.configspace[hp1_name]
    if isinstance(hp1, Constant):
        raise ValueError(f"Hyperparameter '{hp1_name}' is constant and cannot be plotted.")

    values = _values_for_hp(data, hp1)
    result = marginal(values, hp1)
    result["summary"] = _distribution_summary(values, hp1)

    try:
        incumbent, _ = run.get_incumbent(
            objectives=objective, budget=budget, statuses=Status.SUCCESS, seed=seed
        )
        result["incumbent"] = incumbent[hp1_name]
    except (RuntimeError, ValueError):
        result["incumbent"] = None
    return result