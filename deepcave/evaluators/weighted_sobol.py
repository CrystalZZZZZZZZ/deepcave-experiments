"""Weighted Sobol sensitivity estimates for DeepCAVE runs."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from ConfigSpace import (
	CategoricalHyperparameter,
	Constant,
	UniformIntegerHyperparameter,
)
from sklearn.ensemble import RandomForestRegressor

from deepcave.runs import AbstractRun, Status
from deepcave.runs.objective import Objective


def _json_float(value: float) -> float:
	"""Convert numpy values to JSON-compatible finite floats."""
	value = float(value)
	return value if np.isfinite(value) else 0.0


def _sample_empirical(
	x: np.ndarray,
	rng: np.random.Generator,
	size: int,
) -> np.ndarray:
	"""Sample the independent product of empirical marginal distributions."""
	samples = np.empty((size, x.shape[1]), dtype=float)
	for column in range(x.shape[1]):
		values = x[:, column]
		indices = rng.integers(0, len(values), size=size)
		samples[:, column] = values[indices]
	return samples


def _sample_uniform(
	x: np.ndarray,
	hyperparameters: list[Any],
	rng: np.random.Generator,
	size: int,
) -> np.ndarray:
	"""Sample the ConfigSpace reference distribution in encoded coordinates."""
	samples = np.empty((size, x.shape[1]), dtype=float)
	for column, hyperparameter in enumerate(hyperparameters):
		if isinstance(hyperparameter, CategoricalHyperparameter):
			values = [float(hyperparameter.to_vector(choice)) for choice in hyperparameter.choices]
			samples[:, column] = rng.choice(values, size=size, replace=True)
		elif isinstance(hyperparameter, UniformIntegerHyperparameter):
			values = [
				float(hyperparameter.to_vector(value))
				for value in range(hyperparameter.lower, hyperparameter.upper + 1)
			]
			samples[:, column] = rng.choice(values, size=size, replace=True)
		else:
			samples[:, column] = rng.uniform(0.0, 1.0, size=size)
	return samples


def _estimate_indices(
	model: RandomForestRegressor,
	sampler,
	n_samples: int,
	n_features: int,
	rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate indices with the paper's Monte Carlo estimators (5.3) and (5.4)."""
	matrix_a = sampler(n_samples, rng)
	matrix_b = sampler(n_samples, rng)
	prediction_a = model.predict(matrix_a)
	mean_a = float(np.mean(prediction_a))
	variance = float(np.mean(prediction_a**2) - mean_a**2)
	if not np.isfinite(variance) or variance <= 1e-12:
		raise ValueError("The surrogate model has near-zero output variance.")

	first_order = np.empty(n_features, dtype=float)
	total_effect = np.empty(n_features, dtype=float)
	for column in range(n_features):
		# Equation (5.3): keep y=x_i from A and replace z with B.
		complement = matrix_a.copy()
		complement[:, np.arange(n_features) != column] = matrix_b[
			:, np.arange(n_features) != column
		]
		prediction_complement = model.predict(complement)
		first_order[column] = (np.mean(prediction_a * prediction_complement) - mean_a**2) / variance

		# Equation (5.4): keep z from A and replace y=x_i with B.
		target = matrix_a.copy()
		target[:, column] = matrix_b[:, column]
		prediction_target = model.predict(target)
		total_effect[column] = np.mean((prediction_a - prediction_target) ** 2) / (2.0 * variance)
	return first_order, total_effect


def calculate(
	run: AbstractRun,
	objective: Objective,
	budget: Optional[float] = None,
	n_samples: int = 2048,
	n_trees: int = 64,
	seed: int = 0,
	show_uniform_control: bool = True,
) -> Dict[str, Any]:
	"""Calculate weighted and optional uniform Sobol indices for one run/budget."""
	if budget is None:
		budget = run.get_highest_budget()
	if n_samples < 2 or n_trees < 1:
		raise ValueError("n_samples must be at least 2 and n_trees must be positive.")

	dataframe = run.get_encoded_data(
		objective,
		budget,
		statuses=[Status.SUCCESS],
		include_config_ids=True,
	)
	if dataframe.empty:
		raise ValueError("No successful trials are available for this objective and budget.")

	hp_names = list(run.configspace.keys())
	dataframe = dataframe.groupby("config_id", as_index=False).mean(numeric_only=True)
	dataframe = dataframe.dropna(subset=[objective.name])
	if len(dataframe) < 3:
		raise ValueError("At least three successful configurations are required.")

	hyperparameters = list(run.configspace.values())
	constant_names = {hp.name for hp in hyperparameters if isinstance(hp, Constant)}
	active_names = [name for name in hp_names if name not in constant_names]
	if not active_names:
		raise ValueError("Weighted Sobol requires at least one non-constant hyperparameter.")

	x = dataframe[active_names].to_numpy(dtype=float)
	for column in range(x.shape[1]):
		values = x[:, column]
		if not np.isfinite(values).all():
			finite = values[np.isfinite(values)]
			if len(finite) == 0:
				raise ValueError(f"Hyperparameter '{active_names[column]}' has no finite values.")
			x[~np.isfinite(values), column] = float(np.median(finite))

	y = dataframe[objective.name].to_numpy(dtype=float)
	if objective.optimize == "upper":
		y = -y

	active_hyperparameters = [run.configspace[name] for name in active_names]
	model = RandomForestRegressor(
		n_estimators=n_trees,
		min_samples_leaf=3,
		random_state=seed,
		n_jobs=-1,
	)
	model.fit(x, y)

	rng = np.random.default_rng(seed)
	weighted_sampler = lambda size, generator: _sample_empirical(  # noqa: E731
		x, generator, size
	)
	weighted_s1, weighted_st = _estimate_indices(
		model, weighted_sampler, n_samples, len(active_names), rng
	)

	result: Dict[str, Any] = {
		"hp_names": active_names,
		"s1_weighted": [_json_float(value) for value in weighted_s1],
		"st_weighted": [_json_float(value) for value in weighted_st],
		"n_configs": int(len(dataframe)),
		"budget": _json_float(budget),
		"seed": int(seed),
		"notes": [f"excluded constant hp: {name}" for name in sorted(constant_names)],
	}

	if show_uniform_control:
		uniform_sampler = lambda size, generator: _sample_uniform(  # noqa: E731
			x, active_hyperparameters, generator, size
		)
		uniform_s1, uniform_st = _estimate_indices(
			model, uniform_sampler, n_samples, len(active_names), rng
		)
		result["s1_uniform"] = [_json_float(value) for value in uniform_s1]
		result["st_uniform"] = [_json_float(value) for value in uniform_st]
	return result
