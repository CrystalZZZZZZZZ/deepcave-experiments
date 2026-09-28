"""Utilities for building a time-aware configuration footprint."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from deepcave.evaluators.footprint import Footprint
from deepcave.runs import AbstractRun, Status
from deepcave.runs.objective import Objective


def _trial_records(
    run: AbstractRun,
    objective: Objective,
    budget: Union[int, float],
) -> List[Dict[str, Any]]:
    """Return successful trials at ``budget`` in completion order."""
    ordered = sorted(enumerate(run.history), key=lambda item: item[1].end_time)
    records = []
    first_seen: Dict[int, int] = {}
    best_cost: Optional[float] = None
    for history_id, trial in ordered:
        if trial.status != Status.SUCCESS or trial.budget != budget:
            continue
        cost = trial.costs[run.get_objective_id(objective)]
        if cost is None or not np.isfinite(cost):
            continue
        cost = float(cost)
        if trial.config_id not in first_seen:
            first_seen[trial.config_id] = len(records)
        if best_cost is None or (
            objective.optimize == "lower" and cost < best_cost
        ) or (objective.optimize == "upper" and cost > best_cost):
            best_cost = cost
        records.append(
            {
                "trial_id": int(history_id),
                "order": len(records),
                "config_id": int(trial.config_id),
                "cost": cost,
                "cost_best": float(best_cost),
                "first_seen": int(first_seen[trial.config_id]),
                "time": float(trial.end_time),
            }
        )
    return records


def _surface(data: Any) -> Dict[str, Any]:
    return {"x": data[0], "y": data[1], "z": data[2]}


def build_cloud(
    run: AbstractRun,
    objective: Objective,
    budget: Union[int, float],
    details: float = 0.5,
) -> Dict[str, Any]:
    """Build a frozen MDS footprint and its trial-order metadata."""
    evaluator = Footprint(run)
    evaluator.calculate(objective, budget)
    point_meta: List[Dict[str, Any]] = []
    for category in ("configs", "borders", "supports", "incumbents"):
        x, y, config_ids = evaluator.get_points(category)
        for x_value, y_value, config_id in zip(x, y, config_ids):
            point_meta.append(
                {
                    "category": category,
                    "config_id": int(config_id),
                    "x": float(x_value),
                    "y": float(y_value),
                }
            )

    records = _trial_records(run, objective, budget)
    first_seen = {record["config_id"]: record["first_seen"] for record in records}
    for point in point_meta:
        point["first_seen"] = first_seen.get(point["config_id"])

    return {
        "coords": [[point["x"], point["y"]] for point in point_meta],
        "point_meta": point_meta,
        "surface": _surface(evaluator.get_surface(details=details, performance=True)),
        "area_surface": _surface(evaluator.get_surface(details=details, performance=False)),
        "incumbent_prefix": [
            {
                "trial_id": record["trial_id"],
                "order": record["order"],
                "config_id": record["config_id"],
                "cost": record["cost_best"],
            }
            for record in records
        ],
        "t_max": max(len(records) - 1, 0),
        "notes": [],
    }


def subset(cloud: Dict[str, Any], trial: Optional[int] = None) -> Dict[str, Any]:
    """Return the replay-visible points and incumbent prefix at ``trial``."""
    t_max = int(cloud.get("t_max", 0))
    selected_trial = t_max if trial is None else max(0, min(int(trial), t_max))
    points = []
    for point in cloud["point_meta"]:
        first_seen = point.get("first_seen")
        visible = first_seen is not None and first_seen <= selected_trial
        points.append({**point, "visible": visible})
    prefix = [
        item for item in cloud["incumbent_prefix"] if item["order"] <= selected_trial
    ]
    return {"trial": selected_trial, "points": points, "incumbent_prefix": prefix}
