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
# BottleneckDiagnosis

This module provides utilities for detecting and visualizing stagnation
intervals (bottlenecks) in the incumbent curve of a run.

The detection itself is implemented in `deepcave.evaluators.stagnation`.
This plugin renders the incumbent curve, highlights stagnation intervals as
colored background bands, summarizes the current state of the search in a
banner (including bilingual advice), and refreshes itself while a run is
still being evaluated.

## Classes
    - BottleneckDiagnosis: A plugin to detect and visualize stagnation intervals.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import callback_context, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from deepcave import config, interactive, notification
from deepcave.constants import COMBINED_SEED
from deepcave.evaluators.stagnation import (
    INEFFECTIVE_EXPLORATION,
    LOCAL_OPTIMUM_STALL,
    OK,
    STATE_LABELS,
    STATE_LABELS_ZH,
    WARMING_UP,
    annotate_advice,
    build_records,
    detect_full,
)
from deepcave.plugins import Plugin
from deepcave.plugins.dynamic import DynamicPlugin
from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.exceptions import NotMergeableError, RunInequality
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.utils.layout import get_select_options, help_button
from deepcave.utils.styled_plotty import get_color, save_image


# Background colors (r, g, b) of the stagnation bands, by state.
SEGMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    WARMING_UP: (108, 117, 125),  # gray
    LOCAL_OPTIMUM_STALL: (253, 126, 20),  # orange
    INEFFECTIVE_EXPLORATION: (13, 110, 253),  # blue
}

# Band opacity, by severity.
SEVERITY_OPACITY: Dict[str, float] = {
    "warning": 0.14,
    "critical": 0.28,
}

XAXIS_OPTIONS = [
    {"label": "Time", "value": "times"},
    {"label": "Time (log)", "value": "times_log"},
    {"label": "Trials", "value": "trials"},
]

# Minimum number of successful trials needed for a meaningful curve.
MIN_CURVE_TRIALS = 3


def _to_int(value: Any, default: int) -> int:
    """Convert a (possibly string) value to int, falling back to `default`."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    """Convert a (possibly string) value to float, falling back to `default`."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_float(value: Any, digits: int = 4) -> str:
    """Format an optional float for display."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "-"


def _fallback_records(
    run: AbstractRun, objective: Objective, budget: Any, seed: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Build a minimal incumbent curve without novelty statistics.

    This is used as a graceful degradation if the full record building fails
    (e.g. for group runs whose configuration mapping is ambiguous): the curve
    is still shown, but no stagnation detection is performed.

    Parameters
    ----------
    run : AbstractRun
        The run to process.
    objective : Objective
        The objective to track.
    budget : Any
        The budget to filter for.
    seed : Optional[int]
        The seed to filter for. None means all seeds.

    Returns
    -------
    List[Dict[str, Any]]
        Chronological records with keys ``order``, ``time``, ``cost`` and
        ``cost_best``.
    """
    objectives = run.get_objectives()
    objective_id = objectives.index(objective)
    lower_is_better = getattr(objective, "optimize", "lower") != "upper"

    trials = []
    for trial_id, trial in enumerate(run.get_trials()):
        if trial.status != Status.SUCCESS:
            continue
        if budget is not None and trial.budget != budget:
            continue
        if seed is not None and trial.seed != seed:
            continue
        if trial.costs is None or trial.costs[objective_id] is None:
            continue
        trials.append((float(trial.end_time or 0.0), trial_id, trial))
    trials.sort(key=lambda item: (item[0], item[1]))

    records: List[Dict[str, Any]] = []
    best: Optional[float] = None
    for end_time, trial_id, trial in trials:
        cost = float(trial.costs[objective_id])
        if not lower_is_better:
            cost = -cost
        if best is None or cost < best:
            best = cost

        records.append(
            {
                "trial_id": int(trial_id),
                "order": len(records),
                "time": end_time,
                "cost": cost,
                "cost_best": float(best),
                "config_id": int(trial.config_id),
            }
        )

    return records


class BottleneckDiagnosis(DynamicPlugin):
    """
    A plugin to detect and visualize stagnation intervals of a run.

    The incumbent curve is rendered together with colored background bands
    marking stagnation intervals. A banner above the figure summarizes the
    current state of the search (including bilingual advice on how to
    resolve the bottleneck) and a list below the figure details every
    detected interval.

    While a run is still being evaluated, the plugin refreshes itself
    together with the global update interval.
    """

    id = "bottleneck_diagnosis"
    name = "Bottleneck Diagnosis"
    icon = "fas fa-hourglass-half"
    help = "plugins/bottleneck_diagnosis.html"

    use_cache = False
    activate_run_selection = True

    def __init__(self) -> None:
        """Initialize the plugin and reset the realtime signature."""
        super().__init__()

        # Signature (inputs + run content) of the last computed result.
        # Used to skip redundant recomputations of the realtime callback.
        self._realtime_signature: Optional[Tuple[Any, ...]] = None

    def check_runs_compatibility(self, runs: List[AbstractRun]) -> None:
        """
        Check if the runs are compatible.

        This function is needed if all selected runs need something in common
        (e.g. budget or objective). Since this function is called before the
        layout is created, it can be also used to set common values for the
        plugin.

        Runs with different objectives cannot be merged at all. Other
        inequalities (e.g. different budgets) only trigger a warning: the
        detection is run-specific and each selected run is processed
        separately.

        Parameters
        ----------
        runs : List[AbstractRun]
            The selected runs to check.

        Raises
        ------
        NotMergeableError
            If the objectives of the runs are not equal.
        """
        try:
            check_equality(runs, objectives=True, budgets=True)
        except NotMergeableError as e:
            run_inequality = e.args[1]
            if run_inequality == RunInequality.INEQ_OBJECTIVE:
                raise NotMergeableError(
                    "The objectives of the selected runs cannot be merged."
                )
            notification.update(
                "The selected runs are not fully mergeable "
                "(e.g. different budgets). The first run is used as reference.",
                color="warning",
            )

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
                            help_button(
                                "Budget refers to the multi-fidelity budget. "
                                "The stagnation detection always evaluates the trials "
                                "of the selected budget only."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Seed"),
                            help_button(
                                "If 'All (avg)' is selected, the costs are averaged "
                                "per configuration over all seeds (mirroring the "
                                "combined seed of the cost-over-time plugin)."
                            ),
                            dbc.Select(
                                id=register("seed_id", ["value", "options"], type=int),
                                placeholder="Select seed ...",
                            ),
                        ],
                        md=4,
                    ),
                ],
            ),
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
                        md=4,
                    ),
                ],
                className="mt-1",
            ),
            html.Details(
                [
                    html.Summary(
                        html.B("Detection Settings"),
                        style={"cursor": "pointer"},
                        className="text-muted",
                    ),
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Window k"),
                                            dbc.Input(
                                                id=register("window_k", "value", type=int),
                                                type="number",
                                                min=2,
                                                step=1,
                                                placeholder="20",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("eps_noise"),
                                            dbc.Input(
                                                id=register("eps_noise", "value", type=float),
                                                type="number",
                                                step=0.0001,
                                                placeholder="0.001",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("theta_e"),
                                            dbc.Input(
                                                id=register("theta_e", "value", type=float),
                                                type="number",
                                                min=0,
                                                max=1,
                                                step=0.05,
                                                placeholder="0.5",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="mt-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("tau_q"),
                                            dbc.Input(
                                                id=register("tau_q", "value", type=float),
                                                type="number",
                                                min=0.5,
                                                max=0.99,
                                                step=0.05,
                                                placeholder="0.75",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("min_trials"),
                                            dbc.Input(
                                                id=register("min_trials", "value", type=int),
                                                type="number",
                                                min=1,
                                                step=1,
                                                placeholder="20",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="mt-1",
                            ),
                        ],
                        className="pl-2",
                    ),
                ],
                className="mt-2",
            ),
        ]

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load the content for the defined inputs in 'get_input_layout'.

        This method is necessary to pre-load contents for the inputs. So, if
        the plugin is called for the first time or there are no results in the
        cache, the plugin gets its content from this method.

        Returns
        -------
        Dict[str, Any]
            The content to be filled.
        """
        return {
            "objective_id": {"options": [], "value": None},
            "budget_id": {"options": [], "value": None},
            "seed_id": {"options": [], "value": None},
            "xaxis": {"options": XAXIS_OPTIONS, "value": "times"},
            "window_k": {"value": 20},
            "eps_noise": {"value": 0.001},
            "theta_e": {"value": 0.5},
            "tau_q": {"value": 0.75},
            "min_trials": {"value": 20},
        }

    def load_dependency_inputs(  # type: ignore
        self, run, previous_inputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Work like 'load_inputs' but called after inputs have changed.

        Note
        ----
        Only the changes have to be returned.
        The returned dictionary will be merged with the inputs.

        Parameters
        ----------
        run :
            The selected run.
        previous_inputs : Dict[str, Any]
            Previous content of the inputs.
            Not used in this specific function.
        inputs : Dict[str, Any]
            The current content of the inputs.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the changes.
        """
        if run is None:
            return {}

        # Objectives
        objectives = run.get_objectives()
        objective_options = get_select_options(
            labels=[objective.name for objective in objectives],
            values=list(range(len(objectives))),
        )
        objective_value = _to_int(inputs["objective_id"]["value"], -1)
        if not 0 <= objective_value < len(objectives):
            objective_value = 0

        # Budgets (real budgets only; the detection is budget-specific)
        budgets = run.get_budgets(include_combined=False)
        budget_labels = run.get_budgets(human=True, include_combined=False)
        budget_options = get_select_options(
            labels=[str(label) for label in budget_labels],
            values=list(range(len(budgets))),
        )
        budget_value = _to_int(inputs["budget_id"]["value"], -1)
        if not 0 <= budget_value < len(budgets):
            budget_value = max(len(budgets) - 1, 0)

        # Seeds (plus 'All (avg)' if the run contains more than one seed)
        seeds = run.get_seeds(include_combined=False)
        seed_options: List[Dict[str, Any]] = []
        if len(seeds) > 1:
            seed_options += [{"label": "All (avg)", "value": COMBINED_SEED}]
        seed_options += get_select_options(labels=seeds, values=seeds)

        seed_value = inputs["seed_id"]["value"]
        valid_seeds = [option["value"] for option in seed_options]
        if seed_value not in valid_seeds:
            if len(seeds) > 1:
                seed_value = COMBINED_SEED
            elif len(seeds) == 1:
                seed_value = seeds[0]
            else:
                seed_value = None

        return {
            "objective_id": {"options": objective_options, "value": objective_value},
            "budget_id": {"options": budget_options, "value": budget_value},
            "seed_id": {"options": seed_options, "value": seed_value},
        }

    @staticmethod
    def process(run, inputs) -> Dict[str, Any]:  # type: ignore
        """
        Return raw data based on a run and input data.

        Warning
        -------
        The returned data must be JSON serializable.

        Note
        ----
        The passed inputs are cleaned and therefore differs compared to
        'load_inputs' or 'load_dependency_inputs'.
        Please see '_clean_inputs' for more information.

        Parameters
        ----------
        run :
            The selected run.
        inputs : Dict[str, Any]
            The input data.

        Returns
        -------
        Dict[str, Any]
            The detection result, see `deepcave.evaluators.stagnation`.
        """
        objectives = run.get_objectives()
        objective_id = _to_int(inputs.get("objective_id"), 0)
        if not 0 <= objective_id < len(objectives):
            objective_id = 0
        objective = objectives[objective_id]

        budgets = run.get_budgets(include_combined=False)
        budget_id = _to_int(inputs.get("budget_id"), -1)
        if not 0 <= budget_id < len(budgets):
            budget_id = max(len(budgets) - 1, 0)
        budget = budgets[budget_id] if budgets else None

        seed_id = inputs.get("seed_id")
        seed: Optional[int] = None
        if seed_id is not None and seed_id != "" and _to_int(seed_id, COMBINED_SEED) != COMBINED_SEED:
            seed = _to_int(seed_id, COMBINED_SEED)

        window_k = max(2, _to_int(inputs.get("window_k"), 20))
        eps_noise = max(1e-6, _to_float(inputs.get("eps_noise"), 1e-3))
        theta_e = min(max(_to_float(inputs.get("theta_e"), 0.5), 0.0), 1.0)
        tau_q = min(max(_to_float(inputs.get("tau_q"), 0.75), 0.5), 0.99)
        min_trials = max(1, _to_int(inputs.get("min_trials"), 20))

        multi_budget = budget is not None and budgets and budget != budgets[-1]

        try:
            records = build_records(run, objective, budget, seed=seed)
            result = detect_full(
                records,
                window_k=window_k,
                eps_noise=eps_noise,
                theta_e=theta_e,
                tau_q=tau_q,
                min_trials=min_trials,
            )
            result = annotate_advice(result, multi_budget=multi_budget)
        except Exception as e:  # noqa: BLE001
            # The record building can fail e.g. if the configuration mapping of
            # a group run is ambiguous. In this case, degrade gracefully and
            # show the incumbent curve only (without detection).
            result = {
                "state": OK,
                "n_trials": 0,
                "segments": [],
                "advice": [],
                "records": _fallback_records(run, objective, budget, seed),
                "curve_only": True,
                "error": str(e),
            }

        # Meta information for the output layout
        result["objective_name"] = objective.name
        result["budget"] = float(budget) if budget is not None else None
        result["seed"] = seed
        result["is_group"] = getattr(run, "prefix", "") == "group"

        return result

    @staticmethod
    def get_output_layout(register: Callable) -> List[Any]:
        """
        Get the layout for the output block.

        Parameters
        ----------
        register : Callable
            Method to register (user) variables.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        List[Any]
            Layouts for the output block.
        """
        return [
            html.Div(id=register("banner", "children"), className="mb-1"),
            dcc.Graph(
                id=register("graph", "figure"),
                style={"height": config.FIGURE_HEIGHT},
                config={"toImageButtonOptions": {"scale": config.FIGURE_DOWNLOAD_SCALE}},
            ),
            html.Div(id=register("segments", "children"), className="mt-2"),
        ]

    @staticmethod
    def load_outputs(run, inputs, outputs) -> List[Any]:  # type: ignore
        """
        Get the raw outputs and update the layout.

        Parameters
        ----------
        run :
            The selected run.
        inputs : Dict[str, Any]
            The input data.
        outputs : Dict[str, Any]
            The raw outputs from the process method.

        Returns
        -------
        List[Any]
            The rendered banner, figure and segment list.
        """
        banner = _render_banner(run, outputs)
        figure = _render_figure(outputs, xaxis=inputs.get("xaxis", "times"))
        segments = _render_segments(outputs)

        return [banner, figure, segments]

    @interactive
    def register_callbacks(self) -> None:
        """
        Register the callbacks of the plugin.

        In addition to the basic plugin callbacks (input updates, raw data
        dialog, help dialog), the standard dynamic output callback is extended
        by the global update interval: if the content of the selected run
        changed on disk (e.g. because it is still being evaluated), the
        outputs are recomputed. Otherwise the callback is skipped to keep the
        refresh lightweight.
        """
        # Basic plugin callbacks only (the output callback below replaces
        # the one of the DynamicPlugin).
        Plugin.register_callbacks(self)
        from deepcave import app, c, rc

        outputs = []
        for id, attribute in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [Input(self.get_internal_id("update-button"), "n_clicks")]
        # Realtime: also trigger on the global update interval.
        inputs += [Input("global-update", "n_intervals")]
        for id, attribute, *_ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))

        @app.callback(outputs, inputs)  # type: ignore
        def plugin_output_update(update_clicks: Any, n_intervals: Any, *inputs_list: Any) -> Any:
            """
            Update the outputs (manually or on changed run content).

            Parameters
            ----------
            update_clicks : Any
                Clicks of the update button (manual re-detection).
            n_intervals : Any
                Ticks of the global update interval.
            *inputs_list : Any
                Input values from user.

            Returns
            -------
            Any
                The raw outputs.
            """
            # Map the list `inputs_list` to a dict s.t. it's easier to access
            # them.
            inputs = self._list_to_dict(list(inputs_list), input=True)

            # Give feedback instead of silently doing nothing when no run is
            # selected. Otherwise the graph stays empty without any hint and
            # looks like it is "loading forever".
            run_value = (inputs.get("run") or {}).get("value")
            if run_value is None or run_value == "":
                hint = dbc.Alert(
                    "Please select a run above to display the bottleneck diagnosis.",
                    color="info",
                    className="mt-2",
                )
                figure = go.Figure()
                figure.add_annotation(
                    text="No run selected",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font={"size": 18, "color": "#888888"},
                )
                return [hint, figure, html.Div()]

            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            cleaned_inputs = self._clean_inputs(inputs)

            # Guard against stale run ids (e.g. restored from the cache after
            # the working directory or the run selection changed). Show a hint
            # instead of letting the callback fail with "Run not found."
            try:
                runs = self.get_selected_runs(inputs)
            except RuntimeError:
                alert = dbc.Alert(
                    "The previously selected run is no longer available. "
                    "Please select a run again.",
                    color="warning",
                    className="mt-2",
                )
                return [alert, go.Figure(), html.Div()]

            # Skip the recompute if neither the inputs nor the content of the
            # selected runs changed. Important: only skip when the callback was
            # triggered *solely* by the global update interval. Page loads
            # (no trigger) and input/run changes must always (re)render.
            # Otherwise the outputs would stay blank when the user switches
            # back to a previously shown run or refreshes the page, because
            # `self._realtime_signature` is server-side state which survives
            # both, and every subsequent interval tick would keep the
            # (PreventUpdate) blank state forever.
            signature = (inputs_key, tuple(_run_signature(run) for run in runs))

            triggered = callback_context.triggered
            if triggered:
                triggered_ids = []
                for first in triggered:
                    if isinstance(first, dict):
                        triggered_ids.append(first.get("prop_id", ""))
                    else:
                        triggered_ids.append(getattr(first, "prop_id", ""))

                interval_only = all("global-update" in prop_id for prop_id in triggered_ids)
                if interval_only and signature == self._realtime_signature:
                    raise PreventUpdate()

            self._realtime_signature = signature

            # Follow the DynamicPlugin pattern: use per-run caches via
            # rc.get/rc.set. The caches are invalidated automatically by
            # RunCaches.update when the hash of a run changed, so a run which
            # is still evaluated is recomputed on the next interval tick.
            # Do NOT call rc.clear() here: on the global update interval this
            # would wipe the caches of all plugins and force a full recompute
            # (plus a run handler update) on every tick.
            raw_outputs = {}
            for run in runs:
                run_outputs = rc.get(run, self.id, inputs_key)
                if run_outputs is None:
                    self.logger.debug(f"Process {run.name}.")
                    run_outputs = self.process(run, cleaned_inputs)

                    # Cache it
                    if self.use_cache:
                        rc.set(run, self.id, inputs_key, value=run_outputs)
                else:
                    self.logger.debug(f"Found outputs from {run.name} in cache.")

                raw_outputs[run.id] = run_outputs

            # Save for modal
            self.raw_outputs = raw_outputs

            # Cache last inputs
            c.set("last_inputs", self.id, value=inputs)

            try:
                return self._process_raw_outputs(inputs, raw_outputs)
            except RuntimeError:
                # The run disappeared between resolving and rendering (e.g.
                # because the selection changed). Degrade gracefully instead
                # of raising an internal server error.
                alert = dbc.Alert(
                    "The previously selected run is no longer available. "
                    "Please select a run again.",
                    color="warning",
                    className="mt-2",
                )
                return [alert, go.Figure(), html.Div()]


def _run_signature(run: AbstractRun) -> Tuple[Any, ...]:
    """
    Build a lightweight content signature of a run.

    The signature changes if the run file(s) were modified (e.g. new trials
    were written while the run is still being evaluated).

    Parameters
    ----------
    run : AbstractRun
        The run to describe.

    Returns
    -------
    Tuple[Any, ...]
        The signature of the run.
    """
    try:
        latest_change = float(run.latest_change)
    except Exception:  # noqa: BLE001
        latest_change = None

    try:
        n_trials = len(run.history)
    except Exception:  # noqa: BLE001
        n_trials = None

    return (run.id, latest_change, n_trials)


def _state_color(state: str, severity: str) -> str:
    """Map a detection state and severity to a bootstrap alert color."""
    if state == OK:
        return "success"
    if state == WARMING_UP:
        return "secondary"
    if severity == "critical":
        return "danger"

    return "warning"


def _render_banner(run: Any, outputs: Dict[str, Any]) -> html.Div:
    """
    Render the banner summarizing the current state and advice.

    Parameters
    ----------
    run :
        The selected run.
    outputs : Dict[str, Any]
        The raw outputs from the process method.

    Returns
    -------
    html.Div
        The banner layout.
    """
    if outputs.get("curve_only"):
        message = (
            "The stagnation detection is not available for this run "
            "(the configuration mapping could not be resolved). "
            "The incumbent curve is shown without detection."
        )
        if outputs.get("error"):
            message += f" Reason: {outputs['error']}"
        return html.Div(
            dbc.Alert(
                [
                    html.B("Current state: Curve only · 仅显示曲线"),
                    html.Br(),
                    message,
                ],
                color="warning",
                className="mb-0",
            )
        )

    state = outputs.get("state", WARMING_UP)
    n_trials = outputs.get("n_trials", 0)
    segments = outputs.get("segments", [])
    advice = outputs.get("advice", [])
    last_window = outputs.get("last_window", {}) or {}

    if n_trials < MIN_CURVE_TRIALS:
        return html.Div(
            dbc.Alert(
                [
                    html.B(f"Warming up · {STATE_LABELS_ZH[WARMING_UP]}"),
                    html.Br(),
                    f"Only {n_trials} successful trial(s) available. "
                    "At least "
                    f"{MIN_CURVE_TRIALS} trials are needed to draw the curve.",
                ],
                color="secondary",
                className="mb-0",
            )
        )

    # Severity of the ongoing (or last) segment determines the banner color.
    severity = "info"
    if segments:
        reference_segment = segments[-1]
        severity = reference_segment.get("severity", "warning")
    color = _state_color(state, severity)

    badges = [
        dbc.Badge(f"{n_trials} trials", color="light", text_color="dark", className="me-1")
    ]
    if segments:
        ongoing = segments[-1].get("ongoing", False)
        badges += [
            dbc.Badge(
                "ongoing" if ongoing else "resolved",
                color="info" if ongoing else "light",
                text_color="dark" if not ongoing else "white",
                className="me-1",
            ),
            dbc.Badge(
                f"{len(segments)} segment(s)",
                color="light",
                text_color="dark",
                className="me-1",
            ),
        ]
    badges += [
        dbc.Badge(
            f"k={outputs.get('window_k_effective', '?')}",
            color="light",
            text_color="dark",
            className="me-1",
        )
    ]

    children: List[Any] = [
        html.B(f"Current state: {STATE_LABELS.get(state, state)} · {STATE_LABELS_ZH.get(state, state)}"),
        html.Span(badges, className="ms-2"),
    ]

    if last_window:
        children += [
            html.Small(
                f"Last window: δ={_format_float(last_window.get('delta'))}, "
                f"e={_format_float(last_window.get('e'), 2)}, "
                f"m={last_window.get('m', '-')}",
                className="d-block text-muted",
            )
        ]

    if advice:
        advice_items = []
        for item in advice:
            advice_items += [
                html.Li(
                    [
                        dbc.Badge(item.get("action", "?"), color="dark", className="me-1"),
                        html.Span(item.get("advice_en", "")),
                        html.Br(),
                        html.Small(
                            item.get("advice_zh", ""), className="text-muted"
                        ),
                    ]
                )
            ]
        children += [html.Ul(advice_items, className="mb-0 mt-1")]

    return html.Div(dbc.Alert(children, color=color, className="mb-0"))


def _segment_bounds(
    segment: Dict[str, Any], xaxis: str, records: List[Dict[str, Any]]
) -> Tuple[Any, Any]:
    """
    Compute the x-coordinates of a stagnation band.

    Parameters
    ----------
    segment : Dict[str, Any]
        The stagnation segment.
    xaxis : str
        The selected x-axis ('times', 'times_log' or 'trials').
    records : List[Dict[str, Any]]
        The records of the run (index == order).

    Returns
    -------
    Tuple[Any, Any]
        The (x0, x1) coordinates of the band.
    """
    if not records:
        return 0, 0

    start = min(max(segment.get("start_order", 0), 0), len(records) - 1)
    end = min(max(segment.get("end_order", start), start), len(records) - 1)

    if xaxis == "trials":
        return start, end + 1

    # Time axes: use the (possibly non-equidistant) timestamps.
    return records[start]["time"], records[end]["time"]


def _render_figure(outputs: Dict[str, Any], xaxis: str) -> go.Figure:
    """
    Render the incumbent curve with the stagnation bands.

    Parameters
    ----------
    outputs : Dict[str, Any]
        The raw outputs from the process method.
    xaxis : str
        The selected x-axis ('times', 'times_log' or 'trials').

    Returns
    -------
    go.Figure
        The figure.
    """
    records = outputs.get("records", [])
    segments = outputs.get("segments", [])
    objective_name = outputs.get("objective_name", "Cost")

    if xaxis == "trials":
        x_values = [record["order"] for record in records]
        x_title = "Number of evaluated trials"
    else:
        x_values = [record["time"] for record in records]
        x_title = "Wallclock time [s]"

    figure = go.Figure()

    if len(records) < MIN_CURVE_TRIALS:
        figure.update_layout(
            title="Incumbent " + objective_name,
            xaxis={"title": x_title, "type": "log" if xaxis == "times_log" else "linear"},
            yaxis={"title": objective_name},
            margin=config.FIGURE_MARGIN,
            font=dict(size=config.FIGURE_FONT_SIZE),
            annotations=[
                {
                    "text": "Not enough successful trials.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return figure

    # Raw evaluated costs (context for the incumbent curve)
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=[record["cost"] for record in records],
            name="Evaluated costs",
            mode="markers",
            marker={"color": get_color(0, 0.2), "size": 5},
            hovertext=[
                f"Trial {record['order']}, config {record['config_id']}"
                for record in records
            ],
            hoverinfo="y+x+text",
        )
    )

    # Incumbent curve
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=[record["cost_best"] for record in records],
            name="Incumbent",
            mode="lines",
            line={"color": get_color(0), "width": 2},
            line_shape="hv",
        )
    )

    # Stagnation bands
    present_types = []
    for segment in segments:
        seg_type = segment.get("type", WARMING_UP)
        if seg_type not in present_types:
            present_types += [seg_type]

        rgb = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS[WARMING_UP])
        opacity = SEVERITY_OPACITY.get(segment.get("severity", "warning"), 0.14)

        x0, x1 = _segment_bounds(segment, xaxis, records)

        line = {"width": 0}
        if segment.get("ongoing", False):
            line = {
                "width": 1.5,
                "color": f"rgba{rgb + (0.9,)}",
                "dash": "dash",
            }
        elif segment.get("severity") == "critical":
            line = {"width": 1.5, "color": f"rgba{rgb + (0.9,)}"}

        figure.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=f"rgba{rgb + (opacity,)}",
            line=line,
            layer="below",
        )

    # Legend entries for the band types
    for seg_type in present_types:
        rgb = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS[WARMING_UP])
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": f"rgba{rgb + (0.6,)}",
                    "symbol": "square",
                    "size": 10,
                },
                name=STATE_LABELS.get(seg_type, seg_type),
            )
        )

    figure.update_layout(
        title="Incumbent " + objective_name,
        xaxis={"title": x_title, "type": "log" if xaxis == "times_log" else "linear"},
        yaxis={"title": objective_name},
        margin=config.FIGURE_MARGIN,
        font=dict(size=config.FIGURE_FONT_SIZE),
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    save_image(figure, "bottleneck_diagnosis.pdf")

    return figure


def _render_segments(outputs: Dict[str, Any]) -> html.Div:
    """
    Render the list of detected stagnation segments.

    Parameters
    ----------
    outputs : Dict[str, Any]
        The raw outputs from the process method.

    Returns
    -------
    html.Div
        The segment list layout.
    """
    segments = outputs.get("segments", [])
    records = outputs.get("records", [])

    if outputs.get("curve_only"):
        return html.Div()

    if not segments:
        return html.Div(
            html.Small(
                "No stagnation interval detected. · 未检测到停滞区间。",
                className="text-muted",
            )
        )

    items = []
    for segment in segments:
        seg_type = segment.get("type", WARMING_UP)
        rgb = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS[WARMING_UP])
        color = f"rgba{rgb + (1.0,)}"

        severity = segment.get("severity", "warning")
        ongoing = bool(segment.get("ongoing", False))

        badges: List[Any] = [
            dbc.Badge(
                severity,
                color="danger" if severity == "critical" else "warning",
                className="me-1",
            )
        ]
        if ongoing:
            badges += [dbc.Badge("ongoing", color="info", className="me-1")]

        header = [
            html.B(f"Trials {segment.get('start_order', '?')}–{segment.get('end_order', '?')}"),
            html.Span(f" ({segment.get('length', '?')} trials)", className="text-muted"),
            html.Span(
                f" · {STATE_LABELS.get(seg_type, seg_type)}"
                f" · {STATE_LABELS_ZH.get(seg_type, seg_type)}",
                className="ms-1",
            ),
            html.Span(badges, className="ms-2"),
        ]

        details = f"δ={_format_float(segment.get('delta'))} · e={_format_float(segment.get('e'), 2)} · m={segment.get('m', '-')}"
        if records:
            details += f" · time {records[max(segment.get('start_order', 0), 0)]['time']:.0f}s"

        children: List[Any] = [
            html.Div(header),
            html.Small(details, className="text-muted d-block"),
        ]

        advice = segment.get("advice", [])
        if advice:
            first = advice[0]
            children += [
                html.Small(
                    [
                        dbc.Badge(
                            first.get("action", "?"),
                            color="dark",
                            className="me-1",
                        ),
                        first.get("advice_en", ""),
                    ],
                    className="d-block mt-1",
                )
            ]

        items += [
            dbc.ListGroupItem(
                children,
                style={"borderLeft": f"4px solid {color}"},
            )
        ]

    return html.Div(
        [
            html.H6("Detected stagnation segments"),
            dbc.ListGroup(items),
        ]
    )





