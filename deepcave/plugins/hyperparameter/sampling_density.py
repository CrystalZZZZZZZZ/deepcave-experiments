"""Interactive sampling density plugin."""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from ConfigSpace.hyperparameters import Constant

from deepcave import config
from deepcave.evaluators.sampling_density import calculate
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import get_hyperparameter_ticks, save_image


class SamplingDensity(DynamicPlugin):
    """Show one-dimensional marginal or two-dimensional joint sample density."""

    id = "sampling_density"
    name = "Sampling Density"
    icon = "fas fa-chart-area"
    help = "plugins/sampling_density.html"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[dbc.Row]:
        """Return controls which trigger density calculation."""
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            help_button("Only successful trials at the selected budget are shown."),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Mode"),
                            dbc.Select(
                                id=register("mode", ["value", "options"]),
                                placeholder="Select mode ...",
                            ),
                        ],
                        md=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Hyperparameter"),
                            dbc.Select(
                                id=register("hp1", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Second hyperparameter"),
                            dbc.Select(
                                id=register("hp2", ["value", "options"]),
                                placeholder="Select hyperparameter ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mt-3",
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[html.Div]:
        """Return display-only filters."""
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Show uniform baseline"),
                            dbc.Checklist(
                                id=register("show_uniform", ["value"]),
                                options=[{"label": "", "value": True}],
                                value=[True],
                                switch=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show rug"),
                            dbc.Checklist(
                                id=register("show_rug", ["value"]),
                                options=[{"label": "", "value": True}],
                                value=[True],
                                switch=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show evaluated points"),
                            dbc.Checklist(
                                id=register("show_points", ["value"]),
                                options=[{"label": "", "value": True}],
                                value=[True],
                                switch=True,
                            ),
                        ],
                        md=4,
                    ),
                ]
            )
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """Return initial values for the plugin controls."""
        return {
            "objective_id": {"options": [], "value": None},
            "budget_id": {"options": [], "value": None},
            "mode": {
                "options": [
                    {"label": "Single hyperparameter", "value": "single"},
                    {"label": "Hyperparameter pair", "value": "pair"},
                ],
                "value": "single",
            },
            "hp1": {"options": [], "value": None},
            "hp2": {"options": [], "value": None},
            "show_uniform": {"value": [True]},
            "show_rug": {"value": [True]},
            "show_points": {"value": [True]},
        }

    def load_dependency_inputs(self, run, _, inputs) -> Dict[str, Any]:  # type: ignore
        """Populate objective, budget, and hyperparameter choices for the selected run."""
        objective_ids = run.get_objective_ids()
        budget_ids = run.get_budget_ids()
        hp_names = [
            name
            for name, hyperparameter in run.configspace.items()
            if not isinstance(hyperparameter, Constant)
        ]

        objective_value = inputs["objective_id"]["value"]
        budget_value = inputs["budget_id"]["value"]
        mode_value = inputs["mode"]["value"] or "single"
        hp1_value = inputs["hp1"]["value"]
        hp2_value = inputs["hp2"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0] if objective_ids else None
        if budget_value not in budget_ids:
            budget_value = budget_ids[-1] if budget_ids else None
        if hp1_value not in hp_names:
            hp1_value = hp_names[0] if hp_names else None
        if hp2_value not in hp_names or hp2_value == hp1_value:
            hp2_value = next((name for name in hp_names if name != hp1_value), None)

        return {
            "objective_id": {
                "options": get_select_options(run.get_objective_names(), objective_ids),
                "value": objective_value,
            },
            "budget_id": {
                "options": get_select_options(run.get_budgets(human=True), budget_ids),
                "value": budget_value,
            },
            "mode": {
                "options": [
                    {"label": "Single hyperparameter", "value": "single"},
                    {"label": "Hyperparameter pair", "value": "pair"},
                ],
                "value": mode_value,
            },
            "hp1": {"options": get_select_options(hp_names), "value": hp1_value},
            "hp2": {"options": get_select_options(hp_names), "value": hp2_value},
        }

    @staticmethod
    def process(run: AbstractRun, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a JSON-serializable density payload."""
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])
        return calculate(
            run=run,
            objective=objective,
            budget=budget,
            mode=inputs["mode"],
            hp1_name=inputs["hp1"],
            hp2_name=inputs.get("hp2"),
        )

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """Return the Plotly output component."""
        return dcc.Graph(
            register("graph", "figure"),
            style={"height": config.FIGURE_HEIGHT},
            config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """Convert evaluator output into a Plotly figure."""
        mode = outputs["mode"]

        def enabled(value: Any) -> bool:
            return value is True or (isinstance(value, list) and True in value)

        show_uniform = enabled(inputs.get("show_uniform", [True]))
        show_rug = enabled(inputs.get("show_rug", [True]))
        show_points = enabled(inputs.get("show_points", [True]))
        figure = go.Figure()

        if mode == "single":
            hp = run.configspace[outputs["hp"]]
            tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=6, include_nan=False)
            figure.add_trace(
                go.Scatter(
                    x=outputs["grid"], y=outputs["pdf"], mode="lines", name="Observed density"
                )
            )
            if show_uniform:
                figure.add_trace(
                    go.Scatter(
                        x=outputs["grid"],
                        y=outputs["uniform"],
                        mode="lines",
                        name="Uniform baseline",
                        line={"dash": "dash"},
                    )
                )
            if show_rug:
                baseline = max(outputs["uniform"] + outputs["pdf"] + [1.0]) * 0.02
                figure.add_trace(
                    go.Scatter(
                        x=outputs["rug"],
                        y=[-baseline] * len(outputs["rug"]),
                        mode="markers",
                        name="Samples",
                        marker={"symbol": "line-ns-open", "size": 9},
                    )
                )
            if outputs.get("incumbent") is not None:
                figure.add_vline(
                    x=outputs["incumbent"], line_dash="dot", line_color="#d62728", annotation_text="Incumbent"
                )
            figure.update_xaxes(title_text=outputs["hp"], tickvals=tickvals, ticktext=ticktext)
            figure.update_yaxes(title_text="Density")
        else:
            x_edges = np.asarray(outputs["x_edges"])
            y_edges = np.asarray(outputs["y_edges"])
            x = (x_edges[:-1] + x_edges[1:]) / 2
            y = (y_edges[:-1] + y_edges[1:]) / 2
            figure.add_trace(
                go.Heatmap(x=x, y=y, z=outputs["counts"], colorscale="Viridis", name="Density")
            )
            if show_points:
                points = np.asarray(outputs["points"])
                figure.add_trace(
                    go.Scatter(
                        x=points[:, 0], y=points[:, 1], mode="markers", name="Evaluated points",
                        marker={"size": 5, "color": "rgba(255,255,255,0.6)"},
                    )
                )
            if outputs.get("incumbent") is not None:
                figure.add_trace(
                    go.Scatter(
                        x=[outputs["incumbent"][0]], y=[outputs["incumbent"][1]],
                        mode="markers", name="Incumbent", marker={"symbol": "star", "size": 13},
                    )
                )
            hp1 = run.configspace[outputs["hp1"]]
            hp2 = run.configspace[outputs["hp2"]]
            x_ticks, x_text = get_hyperparameter_ticks(hp1, ticks=6, include_nan=False)
            y_ticks, y_text = get_hyperparameter_ticks(hp2, ticks=6, include_nan=False)
            figure.update_xaxes(title_text=outputs["hp1"], tickvals=x_ticks, ticktext=x_text)
            figure.update_yaxes(title_text=outputs["hp2"], tickvals=y_ticks, ticktext=y_text)

        figure.update_layout(margin=config.FIGURE_MARGIN, font={"size": config.FIGURE_FONT_SIZE})
        save_image(figure, "sampling_density.pdf")
        return figure