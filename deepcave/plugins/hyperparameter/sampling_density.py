"""Interactive sampling density plugin."""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, OrdinalHyperparameter

from deepcave import config
from deepcave.evaluators.sampling_density import calculate
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import save_image


class SamplingDensity(DynamicPlugin):
    """Show one-dimensional marginal sample density."""

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
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                [
                                    "Hyperparameter",
                                    help_button(
                                        "Plots use the original hyperparameter values directly. "
                                        "A curve point is [original value, density]."
                                    ),
                                ]
                            ),
                            dbc.Select(
                                id=register("hp1", ["value", "options"]),
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
                ]
            )
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """Return initial values for the plugin controls."""
        return {
            "objective_id": {"options": [], "value": None},
            "budget_id": {"options": [], "value": None},
            "hp1": {"options": [], "value": None},
            "show_uniform": {"value": [True]},
            "show_rug": {"value": [True]},
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
        hp1_value = inputs["hp1"]["value"]
        if objective_value not in objective_ids:
            objective_value = objective_ids[0] if objective_ids else None
        if budget_value not in budget_ids:
            budget_value = budget_ids[-1] if budget_ids else None
        if hp1_value not in hp_names:
            hp1_value = hp_names[0] if hp_names else None

        return {
            "objective_id": {
                "options": get_select_options(run.get_objective_names(), objective_ids),
                "value": objective_value,
            },
            "budget_id": {
                "options": get_select_options(run.get_budgets(human=True), budget_ids),
                "value": budget_value,
            },
            "hp1": {"options": get_select_options(hp_names), "value": hp1_value},
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
            hp1_name=inputs["hp1"],
        )

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """Return the Plotly graph and its distribution summary."""
        return [
            dcc.Graph(
                register("graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
            html.Div(id=register("summary", "children"), className="mt-2"),
        ]

    @staticmethod
    def load_outputs(run, inputs, outputs) -> go.Figure:  # type: ignore
        """Convert evaluator output into a Plotly figure."""
        def enabled(value: Any) -> bool:
            return value is True or (isinstance(value, list) and True in value)

        show_uniform = enabled(inputs.get("show_uniform", [True]))
        show_rug = enabled(inputs.get("show_rug", [True]))
        figure = go.Figure()
        hp = run.configspace[outputs["hp"]]
        tickvals, ticktext = SamplingDensity._raw_ticks(hp)
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
                x=outputs["incumbent"],
                line_dash="dot",
                line_color="#d62728",
                annotation_text="Incumbent",
            )
        figure.update_xaxes(title_text=outputs["hp"], tickvals=tickvals, ticktext=ticktext)
        figure.update_yaxes(title_text="Density in original value space")

        figure.update_layout(
            margin=config.FIGURE_MARGIN,
            font={"size": config.FIGURE_FONT_SIZE},
        )
        save_image(figure, "sampling_density.pdf")
        return [figure, SamplingDensity._build_summary(outputs)]

    @staticmethod
    def _raw_ticks(hp):
        """Return axis ticks in the original hyperparameter value space."""
        if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
            labels = [str(value) for value in (
                hp.choices if isinstance(hp, CategoricalHyperparameter) else hp.sequence
            )]
            return list(range(len(labels))), labels
        return [hp.lower, (hp.lower + hp.upper) / 2, hp.upper], [
            str(hp.lower),
            str((hp.lower + hp.upper) / 2),
            str(hp.upper),
        ]

    @staticmethod
    def _build_summary(outputs: Dict[str, Any]) -> html.Div:
        """Build an alert-style panel of distribution metrics."""
        summary = outputs["summary"]
        metrics: List[Any] = []

        def add_metrics(name: str, details: Dict[str, Any]) -> None:
            metrics.extend(
                [
                    html.Li(f"{name}: {details['sample_count']} observations"),
                    html.Li(f"{name}: edge-share (outer 10%): {details['edge_fraction']:.1%}"),
                ]
            )
            if "max_bin_share" in details:
                metrics.extend(
                    [
                        html.Li(f"{name}: maximum 10-bin share: {details['max_bin_share']:.1%}"),
                        html.Li(
                            f"{name}: occupied 10-bin share: "
                            f"{details['occupied_bin_fraction']:.1%}"
                        ),
                    ]
                )
            if "preferred_category" in details:
                metrics.extend(
                    [
                        html.Li(
                            f"{name}: most frequent category: "
                            f"{details['preferred_category']} ({details['preferred_share']:.1%})"
                        ),
                        html.Li(
                            f"{name}: unvisited categories: "
                            f"{', '.join(details['missing_categories']) or 'none'}"
                        ),
                    ]
                )

        add_metrics(outputs["hp"], summary)

        text = [
            html.B(f"Distribution metrics · {summary['sample_count']} configurations"),
            html.Ul(metrics, className="mb-0 mt-1"),
        ]
        return html.Div(dbc.Alert(text, color="info", className="mb-0"))