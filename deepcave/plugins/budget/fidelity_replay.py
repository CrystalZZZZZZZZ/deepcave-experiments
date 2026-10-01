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

This module provides utilities for replaying a multi-fidelity run as a
Successive Halving / Hyperband style fidelity ladder.

For every fidelity (budget) level ("rung"), the replay shows how many
configurations were evaluated, the best cost, how many configurations the
successive halving simulation would promote, how well the evaluated
configurations cover the promoted ones, and how consistently ranks transfer
between consecutive rungs. Groups are supported: each member run gets its own
KPI rows plus a Mean row.

## Classes
    - FidelityReplay: A plugin to replay the fidelity ladder of a run.
"""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html

from deepcave import config, notification
from deepcave.evaluators.fidelity_replay import (
    CARRY_LAST,
    SKIP,
    WORST,
    collect_rung_costs,
    infer_eta,
    replay,
)
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, save_image

# Number of KPIs aggregated in the Mean row of the group table.
_MEAN_KEYS = ("best_cost", "coverage_top", "spearman_top")


def _member_kpis(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the top-level KPIs from a replay result.

    Parameters
    ----------
    result : Dict[str, Any]
        The replay result.

    Returns
    -------
    Dict[str, Any]
        The final best cost, the coverage on the top rung and the rank
        correlation on the last transition.
    """
    coverage_top = None
    for value in reversed(result["coverage"]):
        if value is not None:
            coverage_top = value
            break

    spearman_top = None
    for value in reversed(result["spearman"]):
        if value is not None:
            spearman_top = value
            break

    return {
        "best_cost": result["best_cost"],
        "coverage_top": coverage_top,
        "spearman_top": spearman_top,
    }


class FidelityReplay(DynamicPlugin):
    """
    A plugin to replay the fidelity ladder of a run.

    Properties
    ----------
    objective_options : List[Dict[str, Any]]
        A list of dictionaries of the objective options.
    """

    id = "fidelity_replay"
    name = "Fidelity Replay"
    icon = "fas fa-layer-group"
    help = "plugins/fidelity_replay.html"
    activate_run_selection = True

    @staticmethod
    def check_run_compatibility(run: AbstractRun) -> bool:
        """
        Check if the run has more than one budget and is compatible.

        Parameters
        ----------
        run : AbstractRun
            The run to be checked.

        Returns
        -------
        bool
            True if the run is compatible, otherwise False.
        """
        if len(run.get_budgets()) < 2:
            notification.update(
                f"{run.name} can not be selected because it needs at least two budgets."
            )
            return False

        return True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the input block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layouts for the input block.
        """
        return [
            html.Div(
                [
                    dbc.Label("Objective"),
                    dbc.Select(
                        id=register("objective_id", ["value", "options"], type=int),
                        placeholder="Select objective ...",
                    ),
                ],
            ),
        ]

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Dict[str, Any]]:  # type: ignore
        """
        Work like 'load_inputs' but called after inputs have changed.

        Parameters
        ----------
        run
            The run to get the objective from.
        _
            The previous inputs (unused).
        inputs
            The inputs containing the objective id and a value.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary with the changes.
        """
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)

        value = inputs["objective_id"]["value"]
        if value is None:
            value = objective_ids[0]

        return {
            "objective_id": {
                "options": objective_options,
                "value": value,
            },
        }

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
                            dbc.Label(
                                [
                                    "Eta (promotion ratio)",
                                    help_button(
                                        "The promotion ratio of the simulated successive "
                                        "halving: on every rung only the top 1/eta of the "
                                        "active configurations is promoted. If *Auto* is "
                                        "selected, eta is inferred from the budget ladder "
                                        "(falls back to 3 if the budgets are not geometric)."
                                    ),
                                ]
                            ),
                            dbc.Select(
                                id=register("eta", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Missing policy",
                                    help_button(
                                        "How to handle promoted configurations without an "
                                        "evaluation on the next rung: **carry_last** keeps "
                                        "the last known cost, **skip** drops the "
                                        "configuration from the ladder, **worst** treats it "
                                        "as if it had the worst observed cost."
                                    ),
                                ]
                            ),
                            dbc.Select(
                                id=register("missing_policy", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in 'get_input_layout' and 'get_filter_layout'.

        Returns
        -------
        Dict[str, Any]
            The content to be filled.
        """
        return {
            "eta": {
                "options": [
                    {"label": "Auto (infer from budgets)", "value": "auto"},
                    {"label": "2", "value": "2"},
                    {"label": "3", "value": "3"},
                    {"label": "4", "value": "4"},
                ],
                "value": "auto",
            },
            "missing_policy": {
                "options": [
                    {"label": "Carry last cost", "value": CARRY_LAST},
                    {"label": "Skip config", "value": SKIP},
                    {"label": "Treat as worst", "value": WORST},
                ],
                "value": CARRY_LAST,
            },
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
            The selected run (or group) to process.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            A serialized dictionary.
        """
        objective_id = inputs["objective_id"]
        objective = run.get_objective(objective_id)

        budgets = [run.get_budget(id) for id in run.get_budget_ids(include_combined=False)]
        rung_costs = collect_rung_costs(run, objective_id=objective_id, budgets=budgets)

        # Resolve eta: "auto" infers the ratio from the budget ladder.
        eta_value = inputs["eta"]
        eta_inferred = eta_value == "auto"
        if eta_inferred:
            effective_eta = infer_eta(budgets)
            if effective_eta is None:
                effective_eta = 3
        else:
            effective_eta = int(eta_value)

        result = replay(
            rung_costs,
            eta=effective_eta,
            optimize=objective.optimize,
            missing_policy=inputs["missing_policy"],
        )
        result["eta_inferred"] = eta_inferred

        output: Dict[str, Any] = {
            "rungs": result["rungs"],
            "objective_name": objective.name,
            "replay": result,
            "is_group": False,
            "members": {},
        }

        # For groups, replay every member run as well (the group itself is
        # the merged run and is handled as the main replay above).
        if isinstance(run, Group):
            output["is_group"] = True
            for member in run.get_runs():
                member_objective_id = member.get_objective_id(objective)
                member_costs = collect_rung_costs(
                    member, objective_id=member_objective_id, budgets=budgets
                )
                member_result = replay(
                    member_costs,
                    eta=effective_eta,
                    optimize=objective.optimize,
                    missing_policy=inputs["missing_policy"],
                )
                output["members"][member.name] = member_result

        return output

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
            html.Div(id=register("banner", "children"), className="mb-1"),
            dcc.Graph(
                id=register("graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
            html.Div(id=register("table", "children"), className="mt-2"),
        ]

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[Any]:  # type: ignore
        """
        Read in the raw data and prepare them for the layout.

        Parameters
        ----------
        run
            The selected run.
        inputs : Dict[str, Any]
            The input and filter values from the user.
        outputs : Dict[str, Any]
            The raw outputs from the process method.

        Returns
        -------
        List[Any]
            The rendered banner, figure and rung KPI table.
        """
        replay_data = outputs["replay"]
        rungs = replay_data["rungs"]
        objective_name = outputs["objective_name"]

        banner = html.Div(
            [
                dbc.Badge(
                    f"eta={replay_data['eta']}"
                    + (" (inferred)" if replay_data["eta_inferred"] else " (manual)"),
                    color="primary",
                    className="me-1",
                ),
                dbc.Badge(f"policy={replay_data['missing_policy']}", color="secondary"),
            ]
        )

        traces = []

        # Best evaluated cost per rung (dashed) and best active cost (solid).
        traces.append(
            go.Scatter(
                x=rungs,
                y=replay_data["best_evaluated"],
                name=f"Best evaluated ({objective_name})",
                mode="lines+markers",
                line=dict(color=get_color(0), dash="dash"),
            )
        )
        traces.append(
            go.Scatter(
                x=rungs,
                y=replay_data["best_active"],
                name=f"Best active ({objective_name})",
                mode="lines+markers",
                line=dict(color=get_color(0)),
            )
        )

        # For groups, add the best active cost of every member run.
        for idx, (name, member) in enumerate(outputs["members"].items()):
            traces.append(
                go.Scatter(
                    x=member["rungs"],
                    y=member["best_active"],
                    name=name,
                    mode="lines+markers",
                    line=dict(color=get_color(idx + 1), width=1),
                    opacity=0.6,
                )
            )

        layout = go.Layout(
            xaxis=dict(title="Fidelity (budget)", type="log"),
            yaxis=dict(title=f"Best cost ({objective_name})"),
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
            legend=dict(orientation="h", y=-0.2),
        )

        figure = go.Figure(data=traces, layout=layout)
        save_image(figure, "fidelity_replay.pdf")

        # Rung KPI table.
        header = html.Thead(
            html.Tr(
                [
                    html.Th("Rung (budget)"),
                    html.Th("#Evaluated"),
                    html.Th("#Active"),
                    html.Th("#Promoted"),
                    html.Th(f"Best evaluated ({objective_name})"),
                    html.Th(f"Best active ({objective_name})"),
                    html.Th("Coverage"),
                    html.Th("Rank corr. (Spearman)"),
                ]
            )
        )
        rows = []
        for i, rung in enumerate(rungs):
            rows.append(
                html.Tr(
                    [
                        html.Th(str(rung)),
                        html.Td(str(replay_data["n_evaluated"][i])),
                        html.Td(str(replay_data["n_active"][i])),
                        html.Td(str(replay_data["n_promoted"][i])),
                        html.Td(
                            f"{replay_data['best_evaluated'][i]:.4f}"
                            if replay_data["best_evaluated"][i] is not None
                            else "-"
                        ),
                        html.Td(
                            f"{replay_data['best_active'][i]:.4f}"
                            if replay_data["best_active"][i] is not None
                            else "-"
                        ),
                        html.Td(
                            f"{replay_data['coverage'][i] * 100:.0f}%"
                            if replay_data["coverage"][i] is not None
                            else "-"
                        ),
                        html.Td(
                            f"{replay_data['spearman'][i - 1]:.2f}"
                            if i > 0 and replay_data["spearman"][i - 1] is not None
                            else "-"
                        ),
                    ]
                )
            )

        table = dbc.Table([header, html.Tbody(rows)], bordered=True, striped=True, size="sm")

        table_children: List[Any] = [table]

        # For groups, add a member KPI table (with a Mean row) below the
        # rung KPI table.
        if outputs["is_group"]:
            member_header = html.Thead(
                html.Tr(
                    [
                        html.Th("Member"),
                        html.Th(f"Best cost @ top rung ({objective_name})"),
                        html.Th("Coverage @ top rung"),
                        html.Th("Rank corr. (last transition)"),
                    ]
                )
            )
            member_rows = []
            kpis = {}
            for name, member in outputs["members"].items():
                kpis[name] = _member_kpis(member)
                member_rows.append(
                    html.Tr(
                        [
                            html.Td(name),
                            html.Td(
                                f"{kpis[name]['best_cost']:.4f}"
                                if kpis[name]["best_cost"] is not None
                                else "-"
                            ),
                            html.Td(
                                f"{kpis[name]['coverage_top'] * 100:.0f}%"
                                if kpis[name]["coverage_top"] is not None
                                else "-"
                            ),
                            html.Td(
                                f"{kpis[name]['spearman_top']:.2f}"
                                if kpis[name]["spearman_top"] is not None
                                else "-"
                            ),
                        ]
                    )
                )

            # Mean row over all members.
            mean_cells = [html.Td(html.B("Mean"))]
            for key in _MEAN_KEYS:
                values = [kpi[key] for kpi in kpis.values() if kpi[key] is not None]
                if len(values) > 0:
                    mean = float(np.mean(values))
                    text = f"{mean:.2f}" if key == "spearman_top" else f"{mean:.4f}"
                    if key == "coverage_top":
                        text = f"{mean * 100:.0f}%"
                    mean_cells.append(html.Td(text))
                else:
                    mean_cells.append(html.Td("-"))
            member_rows.append(html.Tr(mean_cells, className="table-primary"))

            member_table = dbc.Table(
                [member_header, html.Tbody(member_rows)], bordered=True, striped=True, size="sm"
            )
            table_children += [html.H5("Group members"), member_table]

        return [banner, figure, html.Div(table_children)]
