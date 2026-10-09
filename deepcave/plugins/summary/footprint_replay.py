"""Static plugin for replaying the configuration footprint over time."""

from typing import Any, Callable, Dict, List

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc

from deepcave import config
from deepcave.evaluators.footprint_replay import build_cloud, subset
from deepcave.plugins.static import StaticPlugin
from deepcave.runs import Status
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import get_hovertext_from_config, save_image


class FootprintReplay(StaticPlugin):
    """Replay how evaluated configurations appeared in a footprint."""

    id = "footprint_replay"
    name = "Footprint Replay"
    icon = "fas fa-play-circle"
    help = "plugins/footprint_replay.html"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register: Callable) -> List[Any]:
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
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Label("Details"),
            help_button("Controls the resolution of the surface plot."),
            dcc.Slider(
                id=register("details", "value", type=float),
                min=0.1,
                max=0.9,
                step=0.4,
                marks={0.1: "Low", 0.5: "Medium", 0.9: "High"},
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Any]:
        return [
            dbc.Label("Trial"),
            dcc.Slider(
                id=register("trial", ["value", "max", "marks"]),
                min=0,
                max=0,
                step=1,
                marks={0: "0"},
            ),
            dbc.Label("Show unvisited configurations"),
            dbc.Select(
                id=register("show_unvisited", ["value", "options"]),
                placeholder="Select ...",
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "details": {"value": 0.5},
            "trial": {"value": 0},
            "show_unvisited": {"options": get_select_options(binary=True), "value": "true"},
        }

    def load_dependency_inputs(self, run, previous_inputs, inputs) -> Dict[str, Any]:
        objective_ids = run.get_objective_ids()
        budget_ids = run.get_budget_ids()
        selected_budget_id = inputs.get("budget_id", {}).get("value") or budget_ids[-1]
        budget = run.get_budget(selected_budget_id)
        trial_count = sum(
            trial.status == Status.SUCCESS and trial.budget == budget for trial in run.history
        )
        trial_max = max(trial_count - 1, 0)
        trial_value = inputs.get("trial", {}).get("value", 0) or 0
        return {
            "objective_id": {
                "options": get_select_options(run.get_objective_names(), objective_ids),
                "value": inputs["objective_id"]["value"] or objective_ids[0],
            },
            "budget_id": {
                "options": get_select_options(run.get_budgets(human=True), budget_ids),
                "value": inputs["budget_id"]["value"] or budget_ids[-1],
            },
            "trial": {
                "value": min(trial_value, trial_max),
                "max": trial_max,
                "marks": get_select_options(
                    [str(index) for index in range(trial_max + 1)],
                    list(range(trial_max + 1)),
                ),
            },
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        objective = run.get_objective(inputs["objective_id"])
        budget = run.get_budget(inputs["budget_id"])
        return build_cloud(run, objective, budget, details=inputs["details"])

    @staticmethod
    def get_output_layout(register: Callable) -> dbc.Tabs:
        return dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Graph(
                        id=register("graph", "figure"),
                        style={"height": config.FIGURE_HEIGHT},
                        config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
                    ),
                    label="Performance",
                ),
            ]
        )

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[Any]:  # type: ignore
        replay = subset(outputs, inputs.get("trial"))
        show_unvisited = inputs.get("show_unvisited", True)
        objective = run.get_objective(inputs["objective_id"])
        budget = run.get_budget(inputs["budget_id"])
        visible = [point for point in replay["points"] if point["visible"]]
        unvisited = [point for point in replay["points"] if not point["visible"]]

        def point_trace(name, points, color, symbol, size, opacity=1.0):
            config_points = [point for point in points if point["config_id"] >= 0]
            return go.Scatter(
                name=name,
                x=[point["x"] for point in points],
                y=[point["y"] for point in points],
                mode="markers",
                marker={"color": color, "symbol": symbol, "size": size, "opacity": opacity},
                hovertext=[
                    get_hovertext_from_config(run, point["config_id"], budget)
                    for point in config_points
                ],
                hoverinfo="text",
            )

        traces = []
        if show_unvisited:
            traces.append(point_trace("Unvisited", unvisited, "lightgray", "x", 7, 0.45))
        traces.append(point_trace("Evaluated", visible, "#e67e22", "x", 8))

        coordinate_by_id = {
            point["config_id"]: (point["x"], point["y"])
            for point in outputs["point_meta"]
            if point["config_id"] >= 0
        }
        prefix = replay["incumbent_prefix"]
        line_points = [
            coordinate_by_id[item["config_id"]]
            for item in prefix
            if item["config_id"] in coordinate_by_id
        ]
        if line_points:
            current = line_points[-1]
            traces.append(
                go.Scatter(
                    name="Current incumbent",
                    x=[current[0]],
                    y=[current[1]],
                    mode="markers",
                    marker={"color": "#c0392b", "symbol": "triangle-up", "size": 14},
                    hoverinfo="skip",
                )
            )

        surface = outputs["surface"]
        figure = go.Figure(
            data=[
                go.Heatmap(
                    x=surface["x"], y=surface["y"], z=surface["z"],
                    zsmooth="best", hoverinfo="skip", colorscale="blues",
                    colorbar={
                        "title": {"text": objective.name, "side": "right"},
                        "x": 1.04,
                        "len": 0.8,
                    },
                )
            ] + traces,
            layout=go.Layout(
                title={
                    "text": f"Trial {replay['trial']} / {outputs['t_max']}",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 0.98,
                    "yanchor": "top",
                },
                xaxis={"title": None, "tickvals": []},
                yaxis={"title": None, "tickvals": []},
                legend={
                    "orientation": "h",
                    "x": 0,
                    "y": -0.18,
                    "xanchor": "left",
                    "yanchor": "top",
                },
                margin={
                    **config.FIGURE_MARGIN,
                    "t": max(config.FIGURE_MARGIN.get("t", 0), 60),
                    "r": max(config.FIGURE_MARGIN.get("r", 0), 120),
                    "b": max(config.FIGURE_MARGIN.get("b", 0), 90),
                },
                font={"size": config.FIGURE_FONT_SIZE},
            ),
        )
        save_image(figure, "footprint_replay.pdf")
        return [figure]