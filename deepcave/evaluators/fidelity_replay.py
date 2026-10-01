# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Fidelity Replay

This module provides utilities to replay a multi-fidelity optimization run as
a Successive Halving (SH) / Hyperband (HB) style fidelity ladder.

The budgets of a run are mapped onto rungs (either the run's own budget levels
or an eta-geometric ladder). For every rung, the replay tracks how many
configurations were evaluated, the best (incumbent) cost, how many
configurations the successive-halving simulation would promote to the next
rung, how well the actually evaluated configurations cover the promoted ones,
and how consistently ranks transfer between consecutive rungs.

Configurations that would be promoted but lack an evaluation on the next rung
are handled by a missing policy:

- ``carry_last`` (default): carry the last known cost forward;
- ``skip``: drop the configuration from the ladder;
- ``worst``: treat the configuration as if it had the worst observed cost on
  the next rung.

## Functions
    - infer_eta: Infer the eta ratio from a budget ladder.
    - collect_rung_costs: Collect the per-rung costs of a run.
    - replay: Simulate the fidelity ladder and derive per-rung KPIs.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from deepcave.runs import AbstractRun, Status

# Missing policies.
CARRY_LAST = "carry_last"
SKIP = "skip"
WORST = "worst"
MISSING_POLICIES = [CARRY_LAST, SKIP, WORST]


def infer_eta(budgets: Sequence[float]) -> Optional[int]:
    """
    Infer the eta ratio from a geometric budget ladder.

    Parameters
    ----------
    budgets : Sequence[float]
        The budget values of a run (at least two, sorted ascending or not).

    Returns
    -------
    Optional[int]
        The constant integer ratio between consecutive budgets, or None if
        the budgets do not form a constant geometric ladder with an integer
        ratio greater than one.
    """
    unique = sorted(set(float(budget) for budget in budgets))
    if len(unique) < 2:
        return None

    ratios = []
    for prev, cur in zip(unique[:-1], unique[1:]):
        if prev <= 0:
            return None
        ratios.append(cur / prev)

    rounded = round(ratios[0])
    if rounded < 2:
        return None
    for ratio in ratios:
        if abs(ratio - rounded) > 1e-6 * max(ratio, rounded):
            return None

    return int(rounded)

def collect_rung_costs(
    run: AbstractRun,
    objective_id: int = 0,
    budgets: Optional[Sequence[float]] = None,
) -> Dict[float, Dict[int, float]]:
    """
    Collect the (seed-averaged) costs per budget level of a run.

    Parameters
    ----------
    run : AbstractRun
        The run to collect the costs from.
    objective_id : int, optional
        The objective to collect the costs for. Default is 0.
    budgets : Optional[Sequence[float]], optional
        The budget values to use as rungs. By default None, which means all
        non-combined budgets of the run are used.

    Returns
    -------
    Dict[float, Dict[int, float]]
        A mapping from budget value to a mapping from config id to the
        averaged cost of the config on that budget.
    """
    if budgets is None:
        budgets = [run.get_budget(id) for id in run.get_budget_ids(include_combined=False)]

    rung_set = set(float(budget) for budget in budgets)
    # Single pass over the history: accumulate sum and count per
    # (budget, config id) to average over seeds.
    acc: Dict[float, Dict[int, List[float]]] = {rung: {} for rung in rung_set}
    for trial in run.history:
        if trial.status != Status.SUCCESS:
            continue

        rung = float(trial.budget)
        if rung not in rung_set:
            continue

        cost = trial.costs[objective_id]
        if cost is None:
            continue

        cell = acc[rung].setdefault(trial.config_id, [0.0, 0.0])
        cell[0] += float(cost)
        cell[1] += 1

    rung_costs: Dict[float, Dict[int, float]] = {}
    for rung in rung_set:
        rung_costs[rung] = {
            config_id: cell_sum / cell_count
            for config_id, (cell_sum, cell_count) in acc[rung].items()
        }

    return rung_costs

def replay(
    rung_costs: Dict[float, Dict[int, float]],
    eta: int = 3,
    optimize: str = "lower",
    missing_policy: str = CARRY_LAST,
) -> Dict[str, Any]:
    """
    Replay a run as a Successive Halving style fidelity ladder.

    Parameters
    ----------
    rung_costs : Dict[float, Dict[int, float]]
        A mapping from budget value (rung) to a mapping from config id to
        cost, as returned by ``collect_rung_costs``. Rungs are processed in
        ascending budget order.
    eta : int, optional
        The promotion ratio of the successive halving simulation. On every
        rung, only the top ``1/eta`` of the active configurations is promoted
        to the next rung. Default is 3.
    optimize : str, optional
        "lower" if lower costs are better, "upper" otherwise. Default is
        "lower".
    missing_policy : str, optional
        How to handle promoted configurations without an evaluation on the
        next rung: "carry_last", "skip" or "worst". Default is "carry_last".

    Returns
    -------
    Dict[str, Any]
        A JSON serializable dictionary with per-rung KPIs:

        - ``rungs``: the ascending budget values;
        - ``n_evaluated``: number of evaluated configs per rung;
        - ``n_active``: number of active (promoted) configs per rung;
        - ``n_promoted``: number of configs promoted to the next rung;
        - ``best_evaluated``: best cost among the evaluated configs;
        - ``best_active``: best cost among the active configs;
        - ``coverage``: fraction of active configs with a real evaluation;
        - ``spearman``: rank correlation between consecutive rungs (or None);
        - ``best_config_id`` / ``best_cost``: incumbent on the top rung.
    """
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}.")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(
            f"Unknown missing policy '{missing_policy}'. Choose from {MISSING_POLICIES}."
        )

    sign = 1.0 if optimize == "lower" else -1.0
    rungs = sorted(rung_costs.keys())

    active: Dict[int, float] = {}  # config id -> last known cost (minimization)
    n_evaluated: List[int] = []
    n_active: List[int] = []
    n_promoted: List[int] = []
    best_evaluated: List[Optional[float]] = []
    best_active: List[Optional[float]] = []
    coverage: List[Optional[float]] = []
    spearman: List[Optional[float]] = []
    best_config_id: Optional[int] = None
    best_cost: Optional[float] = None

    for i, rung in enumerate(rungs):
        evaluated = {config_id: sign * cost for config_id, cost in rung_costs[rung].items()}
        prev_active = dict(active)

        # Update the active set with the evaluations of this rung.
        current: Dict[int, Tuple[float, str]] = {}
        if missing_policy == SKIP:
            for config_id in list(active.keys()):
                if config_id in evaluated:
                    current[config_id] = (evaluated[config_id], "evaluated")
        else:
            for config_id, last_cost in active.items():
                if config_id in evaluated:
                    current[config_id] = (evaluated[config_id], "evaluated")
                elif missing_policy == WORST and len(evaluated) > 0:
                    worst = max(evaluated.values())
                    current[config_id] = (worst, "worst")
                else:  # carry_last
                    current[config_id] = (last_cost, "carried")

        # New configs may also enter the ladder on every rung.
        for config_id, cost in evaluated.items():
            if config_id not in current:
                current[config_id] = (cost, "evaluated")

        # Rank correlation between consecutive rungs (shared, real evals).
        if i > 0:
            shared = [
                config_id
                for config_id in current
                if current[config_id][1] == "evaluated"
                and config_id in prev_active
                and config_id in rung_costs[rungs[i - 1]]
            ]
            if len(shared) >= 2:
                c1 = [prev_active[config_id] for config_id in shared]
                c2 = [current[config_id][0] for config_id in shared]
                if len(set(c1)) > 1 and len(set(c2)) > 1:
                    correlation = stats.spearmanr(c1, c2).correlation
                    spearman.append(float(correlation))
                else:
                    spearman.append(None)
            else:
                spearman.append(None)

        n_evaluated.append(len(evaluated))
        n_active.append(len(current))

        if len(evaluated) > 0:
            best_evaluated.append(sign * min(evaluated.values()))
        else:
            best_evaluated.append(None)

        if len(current) > 0:
            best_active.append(sign * min(cost for cost, _ in current.values()))
        else:
            best_active.append(None)

        n_real = sum(1 for source in current.values() if source[1] == "evaluated")
        if len(current) > 0:
            coverage.append(n_real / len(current))
        else:
            coverage.append(None)

        # Successive halving promotion for the next rung.
        if i < len(rungs) - 1:
            ranked = sorted(current.items(), key=lambda item: item[1][0])
            n_survivors = max(1, int(np.ceil(len(ranked) / eta)))
            survivors = ranked[:n_survivors]
            active = {config_id: cost for config_id, (cost, _) in survivors}
            n_promoted.append(len(active))

            best_config_id = int(survivors[0][0])
            best_cost = float(survivors[0][1][0])
        else:
            n_promoted.append(0)
            if len(current) > 0:
                ranked = sorted(current.items(), key=lambda item: item[1][0])
                best_config_id = int(ranked[0][0])
                best_cost = float(ranked[0][1][0])

    return {
        "rungs": rungs,
        "eta": eta,
        "missing_policy": missing_policy,
        "n_evaluated": n_evaluated,
        "n_active": n_active,
        "n_promoted": n_promoted,
        "best_evaluated": best_evaluated,
        "best_active": best_active,
        "coverage": coverage,
        "spearman": spearman,
        "best_config_id": best_config_id,
        "best_cost": sign * best_cost if best_cost is not None else None,
    }
