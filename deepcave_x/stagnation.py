"""Optimization bottleneck / stagnation detection."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go


@dataclass
class StagnationSegment:
    """A contiguous interval without incumbent improvement."""

    start_step: int
    end_step: int
    duration_steps: int
    cost_at_start: float
    cost_at_end: float

    @property
    def relative_gap(self) -> float:
        if abs(self.cost_at_start) < 1e-12:
            return float("inf")
        return float((self.cost_at_end - self.cost_at_start) / self.cost_at_start)


def detect_stagnation(
    trajectory: pd.DataFrame,
    min_window: int = 5,
) -> List[StagnationSegment]:
    """Detect plateau intervals between incumbent improvements.

    ``min_window`` is the minimum number of consecutive *evaluated trials*
    without improvement that should count as a stagnation segment.
    """
    required = ["step", "incumbent_cost", "improved"]
    missing = [col for col in required if col not in trajectory.columns]
    if missing:
        raise ValueError(f"trajectory missing columns: {missing}")

    improved_steps = list(trajectory.loc[trajectory["improved"], "step"])
    if not improved_steps:
        # Never improved at all.
        if len(trajectory) >= min_window:
            start = int(trajectory["step"].iloc[0])
            end = int(trajectory["step"].iloc[-1])
            return [
                StagnationSegment(
                    start_step=start,
                    end_step=end,
                    duration_steps=end - start + 1,
                    cost_at_start=float(trajectory["incumbent_cost"].iloc[0]),
                    cost_at_end=float(trajectory["incumbent_cost"].iloc[-1]),
                )
            ]
        return []

    segments: List[StagnationSegment] = []
    boundaries = [int(trajectory["step"].iloc[0]) - 1] + improved_steps
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1] - 1
        duration = end - start + 1
        if duration >= min_window:
            start_row = trajectory.loc[trajectory["step"] == start]
            end_row = trajectory.loc[trajectory["step"] == end]
            segments.append(
                StagnationSegment(
                    start_step=start,
                    end_step=end,
                    duration_steps=duration,
                    cost_at_start=float(start_row["incumbent_cost"].iloc[0]),
                    cost_at_end=float(end_row["incumbent_cost"].iloc[0]),
                )
            )

    # Trailing segment after the last improvement.
    last_improvement = improved_steps[-1]
    tail = trajectory.loc[trajectory["step"] > last_improvement]
    if len(tail) >= min_window:
        segments.append(
            StagnationSegment(
                start_step=last_improvement + 1,
                end_step=int(tail["step"].iloc[-1]),
                duration_steps=len(tail),
                cost_at_start=float(tail["incumbent_cost"].iloc[0]),
                cost_at_end=float(tail["incumbent_cost"].iloc[-1]),
            )
        )

    return segments


def stagnation_advice(
    trajectory: pd.DataFrame,
    segments: List[StagnationSegment],
    optimizer_label: Optional[str] = None,
) -> List[str]:
    """Generate human-readable suggestions from detected segments."""
    prefix = f"[{optimizer_label}] " if optimizer_label else ""
    if not segments:
        return [prefix + "No stagnation detected within the current window."]

    total_steps = len(trajectory)
    stagnant_steps = sum(seg.duration_steps for seg in segments)
    advice = []
    advice.append(
        prefix
        + f"Detected {len(segments)} stagnation segment(s) covering "
        f"{stagnant_steps}/{total_steps} evaluated trials."
    )

    if stagnant_steps / max(total_steps, 1) > 0.5:
        advice.append(
            prefix
            + "More than half of the run contains no improvement: consider "
            "restarting with a different seed, enlarging exploration, or "
            "restricting the search space."
        )
    if any(seg.duration_steps >= 10 for seg in segments):
        advice.append(
            prefix
            + "A long plateau was found: verify that important hyperparameters "
            "are not fixed to a too narrow range."
        )
    return advice


def plot_stagnation(
    trajectory: pd.DataFrame,
    segments: List[StagnationSegment],
    title: Optional[str] = None,
) -> go.Figure:
    """Plot the incumbent cost and mark stagnation intervals."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trajectory["step"],
            y=trajectory["incumbent_cost"],
            mode="lines+markers",
            name="incumbent cost",
            line_shape="hv",
        )
    )
    for seg in segments:
        fig.add_vrect(
            x0=seg.start_step - 0.5,
            x1=seg.end_step + 0.5,
            fillcolor="red",
            opacity=0.12,
            line_width=0,
            annotation_text=f"stagnation {seg.duration_steps}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title or "Incumbent trajectory with stagnation intervals",
        xaxis_title="Evaluated trials",
        yaxis_title="incumbent cost (lower is better)",
        template="plotly_white",
    )
    return fig
