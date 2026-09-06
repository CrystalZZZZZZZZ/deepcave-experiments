"""Extract per-trial incumbent trajectories from a DeepCAVE run."""

from typing import Optional, Union

import pandas as pd

from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


def _resolve_objective(run: AbstractRun, objective: Optional[Union[int, str, Objective]]):
    if objective is None:
        return run.get_objectives()[0]
    if isinstance(objective, Objective):
        return objective
    return run.get_objective(objective)


def extract_trajectory(
    run: AbstractRun,
    objective: Optional[Union[int, str, Objective]] = None,
    budget: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return the incumbent trajectory after every evaluated trial.

    Parameters
    ----------
    run:
        DeepCAVE run (or group) object.
    objective:
        Objective id, name, or ``Objective``. Default: first objective.
    budget:
        Actual budget value to consider. Default: run's highest budget.
    seed:
        Only consider this seed. Default: all seeds.

    Returns a DataFrame with columns:

    - ``step``: evaluation ordinal (1-based) among trials of the chosen budget;
    - ``trial_id``: index into ``run.history``;
    - ``time``: end time of the trial;
    - ``incumbent_value``: actual objective value of the current best config;
    - ``incumbent_cost``: same value transformed to a *lower-is-better* cost;
    - ``improved``: whether the incumbent changed at this step.

    The trajectory is computed with DeepCAVE's own ``get_incumbent``, so the
    returned values match what DeepCAVE plugins show.
    """
    obj = _resolve_objective(run, objective)
    if budget is None:
        budget = run.get_highest_budget()

    ordered = sorted(
        enumerate(run.history), key=lambda item: (item[1].end_time, item[0])
    )

    maximize = obj.optimize == "upper"
    rows = []
    prefix_ids = []
    previous_cost = None

    for history_id, _trial in ordered:
        prefix_ids.append(history_id)
        try:
            _config, value = run.get_incumbent(
                objectives=obj,
                budget=budget,
                seed=seed,
                selected_ids=prefix_ids,
            )
        except RuntimeError:
            # No eligible trial on this budget/seed yet.
            continue

        cost = -float(value) if maximize else float(value)
        if previous_cost is None:
            improved = True
        elif maximize:
            improved = float(value) > previous_cost + 1e-12
        else:
            improved = float(value) < previous_cost - 1e-12

        rows.append(
            {
                "step": len(rows) + 1,
                "trial_id": history_id,
                "time": _trial.end_time,
                "incumbent_value": float(value),
                "incumbent_cost": cost,
                "improved": bool(improved),
            }
        )
        previous_cost = float(value)

    if not rows:
        raise RuntimeError(
            f"No eligible trials for objective '{obj.name}', budget={budget}, "
            "seed={seed}. Check run data."
        )

    return pd.DataFrame(rows)
