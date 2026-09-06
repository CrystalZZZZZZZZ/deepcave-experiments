"""Hyperparameter importance using DeepCAVE run data and Random Forests."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


def random_forest_importance(
    run: AbstractRun,
    objective: Union[int, str, Objective, None] = None,
    budget: Optional[float] = None,
    method: str = "gini",
    n_estimators: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """Train a Random Forest on encoded (config -> objective) data.

    The encoded representation and the objective data come from DeepCAVE's own
    ``Run.get_encoded_data`` API, so the search space handling stays consistent
    with the rest of the tool.

    Parameters
    ----------
    method:
        ``gini`` uses sklearn ``feature_importances_``;
        ``permutation`` uses permutation importance (slower, more reliable).
    """
    if objective is None:
        objective_obj = run.get_objectives()[0]
    elif isinstance(objective, Objective):
        objective_obj = objective
    else:
        objective_obj = run.get_objective(objective)
        if objective_obj is None:
            raise ValueError(f"objective not found: {objective}")

    if budget is None:
        budget = run.get_highest_budget()

    encoded = run.get_encoded_data(objectives=objective_obj, budget=budget)
    hyperparameter_names: List[str] = list(run.configspace.keys())
    missing_hp = [hp for hp in hyperparameter_names if hp not in encoded.columns]
    if missing_hp:
        raise RuntimeError(f"missing encoded hyperparameters: {missing_hp}")

    X = encoded[hyperparameter_names].astype(float)
    y = pd.to_numeric(encoded[objective_obj.name], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if len(y) < 3:
        raise RuntimeError("too few evaluated configurations for importance analysis")

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    forest.fit(X.to_numpy(), y.to_numpy())

    if method == "gini":
        scores = forest.feature_importances_
    elif method == "permutation":
        result = permutation_importance(
            forest,
            X.to_numpy(),
            y.to_numpy(),
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
        scores = result.importances_mean
    else:
        raise ValueError(f"unknown method: {method}")

    table = pd.DataFrame(
        {
            "hyperparameter": hyperparameter_names,
            "importance": scores,
            "n_configs": len(y),
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def deepcave_plugin_importance(
    run: AbstractRun,
    objective_id: Optional[int] = None,
    method: str = "global",
    n_trees: int = 30,
):
    """Optional bridge to DeepCAVE's native Importances plugin.

    This wraps the official API example flow:
    ``generate_inputs -> generate_outputs -> load_outputs``.

    Note
    ----
    The underlying fANOVA/LPI evaluator is officially supported on Linux. On
    macOS the evaluator may fail; use :func:`random_forest_importance` as the
    robust fallback there.
    """
    from deepcave.plugins.hyperparameter.importances import Importances

    if objective_id is None:
        objective_id = run.get_objective_ids()[0]

    plugin = Importances()
    inputs = plugin.generate_inputs(
        hyperparameter_names=list(run.configspace.keys()),
        objective_id1=objective_id,
        objective_id2=None,
        budget_ids=run.get_budget_ids(),
        method=method,
        n_hps=len(run.configspace.keys()),
        n_trees=n_trees,
    )
    outputs = plugin.generate_outputs(run, inputs)
    return plugin.load_outputs(run, inputs, outputs)
