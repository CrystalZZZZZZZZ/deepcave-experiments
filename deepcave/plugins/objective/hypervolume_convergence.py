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
# Hypervolume Convergence

This module provides utilities for visualizing the convergence of the
dominated hypervolume over the course of an optimization run.

The hypervolume of the (incrementally maintained) Pareto front is evaluated at
up to 400 checkpoints along the submission order of the trials. Multiple runs
can be compared in one figure, and a summary table reports the final
hypervolume, the area under the convergence curve, and the point at which 90%
of the final hypervolume was reached.

## Classes
    - HypervolumeConvergence: A plugin to visualize the hypervolume convergence.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config, notification
from deepcave.evaluators.hypervolume import (
    auc_trapezoid,
    hypervolume_series,
    make_ref_point,
    normalize_points,
    time_to_threshold,
    to_minimization,
)
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, Status, check_equality
from deepcave.runs.exceptions import NotMergeableError, RunInequality
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, save_image

# Maximum number of hypervolume evaluations along the convergence curve.
MAX_CHECKPOINTS = 400


def _short_tier(label: str) -> str:
    """
    Shorten an algorithm tier label for table/badge display.

    Parameters
    ----------
    label : str
        The full algorithm tier label.

    Returns
    -------
    str
        A shortened label.
    """
    if label.startswith("2D exact"):
        return "Exact (2D)"
    if label.startswith("3D exact"):
        return "Exact (3D)"
    if label.startswith("1D exact"):
        return "Exact (1D)"
    if label.startswith("WFG"):
        return "WFG (exact)"
    if label.startswith("Monte Carlo"):
        return "Monte Carlo (approx.)"
    if label.startswith("empty"):
        return "-"
    return label


class HypervolumeConvergence(DynamicPlugin):
    """
    A plugin to visualize the hypervolume convergence.

    Properties
    ----------
    objective_options : List[Dict[str, Any]]
        A list of dictionaries of the objective options.
    budget_options : List[Dict[str, Any]]
        A list of dictionaries of the budget options.
    """

    id = "hypervolume_convergence"
    name = "Hypervolume Convergence"
    icon = "fas fa-chart-area"
    help = "plugins/hypervolume_convergence.html"

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """
        Check if the runs are compatible.

        This function is needed if all selected runs need something in common
        (e.g. budget or objective).
        Since this function is called before the layout is created,
        it can be also used to set common values for the plugin.

        If the runs are not mergeable, they still should be displayed
        but with a corresponding warning message.

        Parameters
        ----------
        runs : List[AbstractRun]
            A list containing the selected runs.

        Raises
        ------
        NotMergeableError
            If the objectives of the runs are not equal.
        """
        try:
            check_equality(runs, objectives=True, budgets=True)
        except NotMergeableError as e:
            run_inequality = e.args[1]
            if run_inequality == RunInequality.INEQ_BUDGET:
                notification.update("The budgets of the runs are not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_CONFIGSPACE:
                notification.update(
                    "The configuration spaces of the runs are not equal.", color="warning"
                )
            elif run_inequality == RunInequality.INEQ_META:
                notification.update("The meta data of the runs is not equal.", color="warning")
            elif run_inequality == RunInequality.INEQ_OBJECTIVE:
                raise NotMergeableError("The objectives of the selected runs cannot be merged.")

        # Set some attributes here
        # It is necessary to get the run with the smallest budget and objective options
        # as first comparative value, else there is gonna be an index problem
        objective_options = []
        budget_options = []
        for run in runs:
            objective_names = run.get_objective_names()
            objective_ids = run.get_objective_ids()
            objective_options.append(get_select_options(objective_names, objective_ids))

            budgets = run.get_budgets(human=True)
            budget_ids = run.get_budget_ids()
            budget_options.append(get_select_options(budgets, budget_ids))
        self.objective_options = min(objective_options, key=len)
        self.budget_options = min(budget_options, key=len)

    @staticmethod
    def get_input_layout(register: Callable) -> List[dbc.Row]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[dbc.Row]
            Layouts for the input block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective 1"),
                            dbc.Select(
                                id=register("objective_id_1", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Objective 2",
                                    help_button(
                                        "The hypervolume is computed on the selected pair of "
                                        "objectives. Select the same objective twice to use "
                                        "**all** objectives of the run instead (requires at "
                                        "least two objectives)."
                                    ),
                                ]
                            ),
                            dbc.Select(
                                id=register("objective_id_2", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            help_button(
                                "Budget refers to the multi-fidelity budget. Only trials on "
                                "this budget are considered for the hypervolume."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Reference point factor",
                                    help_button(
                                        "The reference point is the worst value of each "
                                        "objective multiplied by this factor. It bounds the "
                                        "dominated hypervolume."
                                    ),
                                ]
                            ),
                            dbc.Input(
                                id=register("ref_factor", "value", type=float),
                                type="number",
                                min=1.0,
                                step=0.1,
                                placeholder="Reference point factor ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Monte Carlo front threshold",
                                    help_button(
                                        "For more than 4 objectives, the exact WFG algorithm "
                                        "is replaced by a Monte Carlo approximation as soon "
                                        "as the Pareto front exceeds this many points."
                                    ),
                                ]
                            ),
                            dbc.Input(
                                id=register("mc_threshold", "value", type=int),
                                type="number",
                                min=1,
                                step=1,
                                placeholder="Monte Carlo threshold ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the filter block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layouts for the filter block.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("X-Axis"),
                            dbc.Select(
                                id=register("xaxis", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Normalize objectives",
                                    help_button(
                                        "Scales each objective to [0, 1] (using the worst and "
                                        "best value of the run) before computing the "
                                        "hypervolume. Recommended and enabled by default, "
                                        "because it makes the hypervolume values of different "
                                        "runs (and objectives with different scales) "
                                        "comparable."
                                    ),
                                ]
                            ),
                            dbc.Select(
                                id=register("normalize", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        This method is necessary to pre-load contents for the inputs.
        So, if the plugin is called for the first time or there are no results in the cache,
        the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Any]
            The content to be filled.
        """
        return {
            "objective_id_1": {
                "options": self.objective_options,
                "value": self.objective_options[0]["value"],
            },
            "objective_id_2": {
                "options": self.objective_options,
                "value": self.objective_options[-1]["value"],
            },
            "budget_id": {
                "options": self.budget_options,
                "value": self.budget_options[-1]["value"],
            },
            "xaxis": {
                "options": [
                    {"label": "Evaluated trials", "value": "trials"},
                    {"label": "Wallclock time [s]", "value": "times"},
                    {"label": "Logarithmic Time", "value": "times_log"},
                ],
                "value": "trials",
            },
            "normalize": {"options": get_select_options(binary=True), "value": True},
            "ref_factor": {"value": 1.1},
            "mc_threshold": {"value": 500},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run : AbstractRun
            The selected run to process.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.
        """
        budget = run.get_budget(inputs["budget_id"])
        objective_ids = sorted(run.get_objective_ids())

        # If both selects are equal, fall back to all objectives.
        if inputs["objective_id_1"] == inputs["objective_id_2"]:
            selected_ids = objective_ids
        else:
            selected_ids = [inputs["objective_id_1"], inputs["objective_id_2"]]

        objectives = [run.get_objective(objective_id) for objective_id in selected_ids]

        error = None
        if len(selected_ids) < 2:
            error = (
                "The hypervolume requires at least two objectives, but this run has only "
                f"{len(selected_ids)}."
            )
        elif any(objective is None for objective in objectives):
            error = "The selected objectives are not available in this run."

        if error is not None:
            return {
                "error": error,
                "indices": [],
                "check_times": [],
                "hvs": [],
                "method": "-",
                "objective_names": [],
                "n_trials": 0,
                "final_hv": None,
                "ref_point": [],
            }

        optimize = [objective.optimize for objective in objectives]
        objective_names = [objective.name for objective in objectives]

        # Collect the cost vectors in submission order. Duplicated evaluations
        # of the same configuration are counted only once (first success).
        points: List[List[float]] = []
        times: List[float] = []
        seen_config_ids = set()

        start_time = min((trial.start_time for trial in run.history), default=0.0)
        for trial in run.history:
            if trial.budget != budget or trial.status != Status.SUCCESS:
                continue

            if trial.config_id in seen_config_ids:
                continue

            costs = [trial.costs[objective_id] for objective_id in selected_ids]
            if any(cost is None for cost in costs):
                continue

            seen_config_ids.add(trial.config_id)
            points.append([float(cost) for cost in costs])
            times.append(float(trial.end_time - start_time))

        if len(points) == 0:
            return {
                "error": "No successful trials found on the selected budget.",
                "indices": [],
                "check_times": [],
                "hvs": [],
                "method": "-",
                "objective_names": objective_names,
                "n_trials": 0,
                "final_hv": None,
                "ref_point": [],
            }

        # Transform to minimization and optionally normalize to [0, 1]. If
        # normalization is enabled, the worst value of each objective maps to
        # exactly 1.0, hence every run shares the same reference point
        # (1.1 ** d by default), which keeps runs comparable.
        points_min = to_minimization(points, optimize)
        if inputs["normalize"]:
            mins = np.min(points_min, axis=0)
            maxs = np.max(points_min, axis=0)
            points_min = normalize_points(points_min, mins, maxs)

        ref_point = make_ref_point(np.max(points_min, axis=0), factor=inputs["ref_factor"])

        indices, hvs, method = hypervolume_series(
            points_min,
            ref_point,
            max_checkpoints=MAX_CHECKPOINTS,
            mc_threshold=inputs["mc_threshold"],
        )

        return {
            "error": None,
            "indices": [int(index) + 1 for index in indices],
            "check_times": [float(times[index]) for index in indices],
            "hvs": [float(hv) for hv in hvs],
            "method": method,
            "objective_names": objective_names,
            "n_trials": len(points),
            "final_hv": float(hvs[-1]) if len(hvs) > 0 else None,
            "ref_point": ref_point.tolist(),
        }

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """
        Get the layouts for the output block.

        Parameters
        ----------
        register : Callable
            Method to register outputs.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            The layouts for the output block.
        """
        return [
            html.Div(id=register("alert", "children"), className="mb-1"),
            dcc.Graph(
                id=register("graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
            html.Div(id=register("table", "children"), className="mt-2"),
        ]

    @staticmethod
    def load_outputs(  # type: ignore
        runs: List[AbstractRun],
        inputs: Dict[str, Any],
        outputs: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        """
        Read in the raw data and prepare them for the layout.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to 'load_inputs'
        or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        runs : List[AbstractRun]
            The selected runs.
        inputs : Dict[str, Any]
            The input and filter values from the user.
        outputs : Dict[str, Dict[str, Any]]
            The raw outputs from the runs.

        Returns
        -------
        List[Any]
            The rendered alert, figure and summary table.
        """
        alert = None
        results: List[Tuple[int, AbstractRun, Dict[str, Any], float, Optional[float], list]] = []

        for idx, run in enumerate(runs):
            data = outputs.get(run.id)
            if data is None:
                continue

            if data.get("error"):
                alert = dbc.Alert(data["error"], color="warning")
                continue

            if data["final_hv"] is None:
                continue

            x = data["indices"]
            if inputs["xaxis"] in ("times", "times_log"):
                x = data["check_times"]

            auc = auc_trapezoid(data["hvs"], x)
            t90 = time_to_threshold(data["hvs"], x, 0.9)
            results.append((idx, run, data, auc, t90, x))

        if len(results) == 0:
            if alert is None:
                alert = dbc.Alert("No data available for the selected runs.", color="warning")
            return [alert, go.Figure(), html.Div()]

        # Sort by final hypervolume (descending). The first row is the best run.
        results = sorted(results, key=lambda result: result[2]["final_hv"], reverse=True)
        best_idx, best_run, best_data, _, _, _ = results[0]

        traces = []
        for idx, run, data, _, _, x in sorted(results, key=lambda result: result[0]):
            traces.append(
                go.Scatter(
                    x=x,
                    y=data["hvs"],
                    name=run.name,
                    line_shape="hv",
                    mode="lines+markers",
                    marker=dict(symbol="circle", size=4),
                    line=dict(color=get_color(idx)),
                    hovertemplate="%{x:.0f}: HV=%{y:.4f}<extra>" + run.name + "</extra>",
                )
            )

        xaxis: Dict[str, Any] = {"title": "Number of evaluated trials"}
        if inputs["xaxis"] == "times":
            xaxis = {"title": "Wallclock time [s]"}
        elif inputs["xaxis"] == "times_log":
            xaxis = {"title": "Wallclock time [s]", "type": "log"}

        best_tier = _short_tier(best_data["method"])
        layout = go.Layout(
            xaxis=xaxis,
            yaxis=dict(title="Hypervolume"),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
            legend=dict(orientation="h", y=-0.2),
            annotations=[
                dict(
                    text=f"Best: {best_run.name} (HV={best_data['final_hv']:.4f}, {best_tier})",
                    xref="paper",
                    yref="paper",
                    x=1.0,
                    y=1.06,
                    xanchor="right",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=config.FIGURE_FONT_SIZE, color=get_color(best_idx)),
                )
            ],
        )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "hypervolume_convergence.pdf")

        # Summary table. The first row contains the best run (and the tier badge).
        header = html.Thead(
            html.Tr(
                [
                    html.Th("Run"),
                    html.Th("Final HV"),
                    html.Th("AUC"),
                    html.Th("t@90% HV"),
                    html.Th("Algorithm"),
                ]
            )
        )
        rows = []
        for row_idx, (idx, run, data, auc, t90, _) in enumerate(results):
            tier = _short_tier(data["method"])
            first_cell = run.name
            if row_idx == 0:
                first_cell = html.Span(
                    [
                        run.name + " ",
                        dbc.Badge(tier, color="success", className="ms-1"),
                    ]
                )

            rows.append(
                html.Tr(
                    [
                        html.Td(first_cell),
                        html.Td(f"{data['final_hv']:.4f}"),
                        html.Td(f"{auc:.4f}" if auc is not None else "-"),
                        html.Td(f"{t90:.1f}" if t90 is not None else "-"),
                        html.Td(tier if row_idx > 0 else "best"),
                    ],
                    className="table-success" if row_idx == 0 else None,
                )
            )

        table = dbc.Table([header, html.Tbody(rows)], bordered=True, striped=True, size="sm")

        return [alert, figure, table]
