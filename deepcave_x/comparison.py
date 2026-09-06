"""Multi-run / multi-optimizer trajectory comparison."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .trajectory import extract_trajectory


@dataclass
class TrajectorySummary:
    """Summary metrics for one optimizer trajectory."""

    label: str
    n_steps: int
    n_improvements: int
    total_time: float
    final_value: float
    final_cost: float
    last_improvement_step: Optional[int]


def _build_summary(trajectory: pd.DataFrame, label: str) -> TrajectorySummary:
    improved_rows = trajectory.loc[trajectory["improved"]]
    last_improvement_step = (
        int(improved_rows["step"].max()) if len(improved_rows) > 0 else None
    )
    return TrajectorySummary(
        label=label,
        n_steps=int(len(trajectory)),
        n_improvements=int(improved_rows["improved"].sum()),
        total_time=float(trajectory["time"].max() - trajectory["time"].min()),
        final_value=float(trajectory["incumbent_value"].iloc[-1]),
        final_cost=float(trajectory["incumbent_cost"].iloc[-1]),
        last_improvement_step=last_improvement_step,
    )


def summarize_trajectories(
    trajectories: Sequence[pd.DataFrame], labels: Sequence[str]
) -> pd.DataFrame:
    """Create one row per trajectory with comparison-relevant metrics."""
    summaries = [
        _build_summary(trajectory, label)
        for trajectory, label in zip(trajectories, labels)
    ]
    best_cost = min(item.final_cost for item in summaries)
    rows = []
    for item in summaries:
        rows.append(
            {
                "label": item.label,
                "n_steps": item.n_steps,
                "n_improvements": item.n_improvements,
                "total_time": item.total_time,
                "final_value": item.final_value,
                "final_cost": item.final_cost,
                "gap_to_best": item.final_cost - best_cost,
                "last_improvement_step": item.last_improvement_step,
            }
        )
    return pd.DataFrame(rows)


def _value_at_step(trajectory: pd.DataFrame, step: int) -> Optional[float]:
    mask = trajectory["step"] <= step
    if not mask.any():
        return None
    return float(trajectory.loc[mask, "incumbent_cost"].iloc[-1])


class TrajectoryComparison:
    """Aligned comparison view for multiple incumbent trajectories."""

    def __init__(
        self,
        trajectories: Sequence[pd.DataFrame],
        labels: Sequence[str],
        objective_label: str = "objective cost",
    ):
        if len(trajectories) != len(labels):
            raise ValueError("trajectories and labels must have equal length")
        if not trajectories:
            raise ValueError("at least one trajectory required")
        self.trajectories = list(trajectories)
        self.labels = list(labels)
        self.objective_label = objective_label

    def aligned_frame(self) -> pd.DataFrame:
        """Build a long-form DataFrame with per-step values for all trajectories."""
        max_step = max(t["step"].max() for t in self.trajectories)
        rows = []
        for trajectory, label in zip(self.trajectories, self.labels):
            for step in range(1, int(max_step) + 1):
                value = _value_at_step(trajectory, step)
                rows.append(
                    {
                        "step": step,
                        "label": label,
                        "incumbent_cost": value,
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        return summarize_trajectories(self.trajectories, self.labels)

    def plot(self, title: Optional[str] = None) -> go.Figure:
        """Step-plot the incumbent cost of every optimizer."""
        fig = go.Figure()
        for trajectory, label in zip(self.trajectories, self.labels):
            fig.add_trace(
                go.Scatter(
                    x=trajectory["step"],
                    y=trajectory["incumbent_cost"],
                    mode="lines",
                    name=label,
                    line_shape="hv",
                )
            )
        fig.update_layout(
            title=title or "Multi-optimizer incumbent trajectory comparison",
            xaxis_title="Evaluated configurations (trial index)",
            yaxis_title=self.objective_label,
            legend_title="Optimizer",
            template="plotly_white",
        )
        return fig
