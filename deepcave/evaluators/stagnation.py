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
# Stagnation

This module detects stagnation intervals in the cost-over-time curve of a run
and provides bilingual (English/Chinese) advice rules for common bottlenecks.

The detector combines four complementary components:

1. Relative incumbent improvement rate over a sliding window.
2. A Mann-Whitney U test on the raw cost distribution (noise robustness).
3. A novelty rate based on nearest-neighbor distances of sampled configurations.
4. A segment state machine that confirms and closes stagnation intervals.

## Functions
    build_records: Convert a run into a chronological list of trial records.
    detect_full: Detect stagnation intervals on a list of records.
    annotate_advice: Attach bilingual advice to a detection result.
    get_advice: Return matching bilingual advice rules for a context.

## Classes
    StagnationDetector: Streaming stagnation detector.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.stats import mannwhitneyu

from deepcave.constants import COMBINED_SEED
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


# Possible states of a window (and of the whole run).
OK = "ok"
WARMING_UP = "warming_up"
LOCAL_OPTIMUM_STALL = "local_optimum_stall"
INEFFECTIVE_EXPLORATION = "ineffective_exploration"
STALL_STATES = (LOCAL_OPTIMUM_STALL, INEFFECTIVE_EXPLORATION)

STATE_LABELS = {
    OK: "Improving",
    WARMING_UP: "Warming up",
    LOCAL_OPTIMUM_STALL: "Local optimum stall",
    INEFFECTIVE_EXPLORATION: "Ineffective exploration",
}

STATE_LABELS_ZH = {
    OK: "持续改进",
    WARMING_UP: "预热中",
    LOCAL_OPTIMUM_STALL: "局部最优停滞",
    INEFFECTIVE_EXPLORATION: "无效探索",
}

# Nearest-neighbor distances of new configurations are only tracked for the
# last `DEFAULT_MAX_DIST_HISTORY` configurations (adaptive novelty threshold).
DEFAULT_MAX_DIST_HISTORY = 200

# Minimum number of new configurations before the novelty threshold is used.
MIN_DIST_SAMPLES = 5


def build_records(
    run: AbstractRun,
    objective: Objective,
    budget: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a run into a chronological list of trial records.

    Only successful trials at the given budget (and seed) are considered.
    Costs are normalized s.t. lower is better. If all seeds are considered
    (seed is None or COMBINED_SEED), the per-configuration average cost is used,
    mirroring the "Combined" seed of the cost-over-time plugin.

    Parameters
    ----------
    run : AbstractRun
        The run to process.
    objective : Objective
        The objective to track.
    budget : Optional[Union[int, float]], optional
        The budget to filter for. By default None (highest budget).
    seed : Optional[int], optional
        The seed to filter for. None or COMBINED_SEED means all seeds.

    Returns
    -------
    List[Dict[str, Any]]
        Chronologically ordered records with the keys ``trial_id``, ``order``,
        ``time``, ``cost``, ``cost_best``, ``config_id``, ``is_new_config``
        and ``nn_dist``.
    """
    if budget is None:
        budgets = run.get_budgets(include_combined=False)
        if len(budgets) == 0:
            return []
        budget = budgets[-1]

    objective_id = run.get_objective_id(objective)
    if objective_id is None:
        return []

    if seed == COMBINED_SEED:
        seed = None

    # Encoded configuration vectors (used for the novelty distances).
    hp_names = list(run.configspace.keys())
    vectors: Dict[int, np.ndarray] = {}
    try:
        encoded = run.get_encoded_data(
            objectives=objective,
            budget=budget,
            seed=seed,
            statuses=Status.SUCCESS,
            include_config_ids=True,
        )
        if encoded is not None and not encoded.empty:
            columns = ["config_id"] + hp_names
            if all(column in encoded.columns for column in columns):
                data = encoded[columns].to_numpy(dtype=float)
                for row in data:
                    vectors[int(row[0])] = row[1:]
    except Exception:
        vectors = {}

    dim = len(hp_names)
    norm = float(np.sqrt(dim)) if dim > 0 else 1.0

    # Successful trials at the selected budget (and seed), ordered by end time.
    trials = []
    for trial_id, trial in enumerate(run.history):
        if trial.status != Status.SUCCESS:
            continue
        if trial.budget != budget:
            continue
        if seed is not None and trial.seed != seed:
            continue
        cost = trial.costs[objective_id] if trial.costs is not None else None
        if cost is None or not np.isfinite(cost):
            continue
        trials.append((float(trial.end_time), trial_id, trial))
    trials.sort(key=lambda item: (item[0], item[1]))

    # Per-configuration costs (averaged over seeds if all seeds are considered).
    cost_of: Dict[int, float] = {}
    if seed is None:
        sums: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for _, _, trial in trials:
            value = float(trial.costs[objective_id])
            sums[trial.config_id] = sums.get(trial.config_id, 0.0) + value
            counts[trial.config_id] = counts.get(trial.config_id, 0) + 1
        cost_of = {config_id: sums[config_id] / counts[config_id] for config_id in sums}
    else:
        cost_of = {t.config_id: float(t.costs[objective_id]) for _, _, t in trials}

    lower_is_better = getattr(objective, "optimize", "lower") != "upper"

    records: List[Dict[str, Any]] = []
    best: Optional[float] = None
    seen: Dict[int, bool] = {}
    history_vecs: List[np.ndarray] = []

    for end_time, trial_id, trial in trials:
        cost = cost_of[trial.config_id]
        if not lower_is_better:
            cost = -cost

        config_id = int(trial.config_id)
        if config_id not in seen:
            seen[config_id] = True
            vector = vectors.get(config_id)
            if vector is None or not history_vecs:
                nn_dist = 1.0
            else:
                reference = np.asarray(history_vecs[-DEFAULT_MAX_DIST_HISTORY:])
                nn_dist = float(np.min(np.linalg.norm(reference - vector, axis=1)) / norm)
            if vector is not None:
                history_vecs.append(vector)
            is_new = True
        else:
            nn_dist = 0.0
            is_new = False

        if best is None or cost < best:
            best = cost

        records.append(
            {
                "trial_id": int(trial_id),
                "order": len(records),
                "time": float(end_time),
                "cost": float(cost),
                "cost_best": float(best),
                "config_id": config_id,
                "is_new_config": bool(is_new),
                "nn_dist": float(nn_dist),
            }
        )

    return records


class StagnationDetector:
    """
    Streaming stagnation detector.

    Records can be added incrementally via `update` (e.g. while a run is still
    being evaluated) and the current detection result is obtained via `detect`.
    The detection itself is deterministic and recomputed from scratch, s.t.
    streaming and offline results are always identical.

    Properties
    ----------
    records : List[Dict[str, Any]]
        The records added so far.
    """

    def __init__(
        self,
        window_k: int = 20,
        eps_noise: float = 1e-3,
        theta_e: float = 0.5,
        tau_q: float = 0.75,
        min_trials: int = 20,
        confirm_windows: int = 2,
        max_dist_history: int = DEFAULT_MAX_DIST_HISTORY,
    ) -> None:
        """
        Prepare the detector.

        Parameters
        ----------
        window_k : int, optional
            Sliding window size (capped at 10% of the number of trials).
        eps_noise : float, optional
            Relative improvement rate below which a window counts as stagnant.
        theta_e : float, optional
            Novelty rate threshold separating local optimum stalls
            (below) from ineffective exploration (above).
        tau_q : float, optional
            Quantile of the nearest-neighbor distance history used as the
            adaptive novelty threshold.
        min_trials : int, optional
            Minimum number of successful trials before stagnation is detected.
        confirm_windows : int, optional
            Consecutive stagnant windows required to confirm a segment.
        max_dist_history : int, optional
            Number of recent nearest-neighbor distances considered for tau.
        """
        self.window_k = max(3, int(window_k))
        self.eps_noise = float(eps_noise)
        self.theta_e = float(theta_e)
        self.tau_q = float(np.clip(float(tau_q), 0.0, 1.0))
        self.min_trials = max(1, int(min_trials))
        self.confirm_windows = max(1, int(confirm_windows))
        self.max_dist_history = int(max_dist_history)

        self._records: List[Dict[str, Any]] = []

    def update(self, record: Dict[str, Any]) -> None:
        """
        Add a record (as created by `build_records`).

        Parameters
        ----------
        record : Dict[str, Any]
            The record to add.
        """
        self._records.append(dict(record))

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Return the records added so far."""
        return self._records

    def _effective_window(self, n_trials: int) -> int:
        """Cap the window size at 10% of the number of trials (at least 3)."""
        return max(3, min(self.window_k, n_trials // 10))

    def _significant_improvement(self, records: List[Dict[str, Any]], index: int) -> bool:
        """Check whether the incumbent improved significantly at ``index``."""
        if index <= 0:
            return False
        before = records[index - 1]["cost_best"]
        after = records[index]["cost_best"]
        return (before - after) >= self.eps_noise * max(abs(before), 1e-6)


    def _classify_window(
        self,
        records: List[Dict[str, Any]],
        index: int,
        window: int,
        dist_history: List[float],
    ) -> Dict[str, Any]:
        """
        Classify the sliding window ending at ``index``.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            All records so far.
        index : int
            Index of the last record of the window.
        window : int
            The (effective) window size.
        dist_history : List[float]
            Nearest-neighbor distances of recently seen configurations.

        Returns
        -------
        Dict[str, Any]
            Window classification with ``state``, ``delta``, ``e``, ``tau``,
            ``m`` and ``dist_improving``.
        """
        k_eff = min(window, index + 1)
        base = records[index - k_eff] if index - k_eff >= 0 else records[0]

        # Component 1: relative incumbent improvement rate over the window.
        best_before = base["cost_best"]
        best_now = records[index]["cost_best"]
        delta = (best_before - best_now) / max(abs(best_before), 1e-6)

        # Number of (even tiny) incumbent improvements inside the window.
        improvements = 0
        for i in range(index - k_eff + 1, index + 1):
            if i == 0 or records[i]["cost_best"] < records[i - 1]["cost_best"]:
                improvements += 1

        if delta >= self.eps_noise:
            return {
                "state": OK,
                "delta": float(delta),
                "e": None,
                "tau": None,
                "m": improvements,
                "dist_improving": False,
            }

        # Component 2: Mann-Whitney U test on the raw cost distribution.
        # Partial windows (at the start of a run) cannot claim stagnation.
        if k_eff < window:
            return {
                "state": OK,
                "delta": float(delta),
                "e": None,
                "tau": None,
                "m": improvements,
                "dist_improving": False,
            }

        dist_improving = False
        if k_eff >= 6:
            costs = [records[i]["cost"] for i in range(index - k_eff + 1, index + 1)]
            half = k_eff // 2
            first, second = costs[:half], costs[k_eff - half:]
            if len(first) >= 3 and len(second) >= 3:
                try:
                    _, p_value = mannwhitneyu(second, first, alternative="less")
                    dist_improving = bool(p_value < 0.05)
                except ValueError:
                    dist_improving = False

        if dist_improving:
            return {
                "state": OK,
                "delta": float(delta),
                "e": None,
                "tau": None,
                "m": improvements,
                "dist_improving": True,
            }

        # Component 3: novelty rate of the configurations sampled in the window.
        if len(dist_history) >= MIN_DIST_SAMPLES:
            tau = float(np.quantile(dist_history, self.tau_q))
        else:
            tau = 0.0

        new_configs = [
            records[i]
            for i in range(index - k_eff + 1, index + 1)
            if records[i]["is_new_config"]
        ]
        if new_configs:
            e = sum(1 for record in new_configs if record["nn_dist"] >= tau) / len(new_configs)
        else:
            # Sampling is fully concentrated on already known configurations.
            e = 0.0

        state = LOCAL_OPTIMUM_STALL if e < self.theta_e else INEFFECTIVE_EXPLORATION
        return {
            "state": state,
            "delta": float(delta),
            "e": float(e),
            "tau": float(tau),
            "m": improvements,
            "dist_improving": False,
        }


    def detect(self) -> Dict[str, Any]:
        """
        Detect stagnation intervals on the records added so far.

        Returns
        -------
        Dict[str, Any]
            The detection result containing the overall ``state``, the
            per-trial ``records``, the confirmed ``segments``, the
            ``last_window`` statistics and the current ``segment``.
        """
        records = self._records
        n = len(records)

        result: Dict[str, Any] = {
            "state": WARMING_UP,
            "n_trials": n,
            "window_k_effective": 0,
            "params": {
                "window_k": self.window_k,
                "eps_noise": self.eps_noise,
                "theta_e": self.theta_e,
                "tau_q": self.tau_q,
                "min_trials": self.min_trials,
            },
            "records": records,
            "segments": [],
            "last_window": None,
            "had_stall": False,
            "ongoing": False,
            "segment": None,
        }
        if n == 0:
            return result

        window = self._effective_window(n)

        # Not enough trials yet: the whole run is still warming up.
        if n < self.min_trials:
            result["window_k_effective"] = window
            return result

        # Components 1-3: classify every window.
        states: List[str] = [OK] * n
        windows: List[Dict[str, Any]] = [dict() for _ in range(n)]
        dist_history: List[float] = []
        for index, record in enumerate(records):
            if record["is_new_config"]:
                dist_history.append(record["nn_dist"])
                if len(dist_history) > self.max_dist_history:
                    dist_history.pop(0)
            window_result = self._classify_window(records, index, window, dist_history)
            states[index] = window_result["state"]
            windows[index] = window_result

        # Component 4: segment state machine.
        segments: List[Dict[str, Any]] = []
        index = 0
        while index < n:
            if states[index] in STALL_STATES:
                end = index
                while end < n and states[end] == states[index]:
                    end += 1

                # A stagnation interval is only confirmed after enough windows.
                if end - index >= self.confirm_windows:
                    # Anchor the segment at the last significant improvement.
                    anchor = None
                    for candidate in range(index, 0, -1):
                        if self._significant_improvement(records, candidate):
                            anchor = candidate
                            break
                    start = anchor + 1 if anchor is not None else 0
                    if segments:
                        start = max(start, segments[-1]["end_order"] + 1)
                    start = min(start, index)

                    segment = {
                        "type": states[index],
                        "start_order": start,
                        "end_order": end - 1,
                        "length": end - start,
                        "start_time": records[start]["time"],
                        "end_time": records[end - 1]["time"],
                        "start_trial_id": records[start]["trial_id"],
                        "end_trial_id": records[end - 1]["trial_id"],
                        "delta": windows[end - 1]["delta"],
                        "e": windows[end - 1]["e"],
                        "tau": windows[end - 1]["tau"],
                        "m": windows[end - 1]["m"],
                        "ongoing": end == n,
                    }
                    fraction = segment["length"] / n
                    if segment["length"] >= 3 * window or fraction > 0.3:
                        segment["severity"] = "critical"
                    else:
                        segment["severity"] = "warning"
                    segments.append(segment)
                index = end
            else:
                index += 1

        result["state"] = states[-1]
        result["window_k_effective"] = window
        result["segments"] = segments
        result["last_window"] = windows[-1]
        result["had_stall"] = len(segments) > 0
        result["ongoing"] = bool(segments) and segments[-1]["ongoing"]
        if result["ongoing"]:
            result["segment"] = segments[-1]

        return result


def detect_full(records: List[Dict[str, Any]], **params: Any) -> Dict[str, Any]:
    """
    Detect stagnation intervals on a complete list of records.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Records as created by `build_records`.
    **params : Any
        Parameters passed to `StagnationDetector`.

    Returns
    -------
    Dict[str, Any]
        The detection result, see `StagnationDetector.detect`.
    """
    detector = StagnationDetector(**params)
    for record in records:
        detector.update(record)
    return detector.detect()


class _SafeFormatDict(dict):
    """Dict that keeps unknown placeholders instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format(template: str, context: Dict[str, Any]) -> str:
    """Format a template with missing keys kept as placeholders."""
    try:
        return template.format_map(_SafeFormatDict(context))
    except Exception:
        return template


ADVICE_RULES: List[Dict[str, Any]] = [
    {
        "id": "too_few_success",
        "severity": "info",
        "action": "CONTINUE",
        "condition": lambda ctx: ctx["n_trials"] == 0,
        "advice_en": (
            "No successful trials found for the selected objective, budget and seed. "
            "Check whether the run has finished at least one successful evaluation."
        ),
        "advice_zh": (
            "未找到所选目标、预算与随机种子下的成功试验。"
            "请检查该运行是否至少完成了一次成功评估。"
        ),
    },
    {
        "id": "warming_up",
        "severity": "info",
        "action": "CONTINUE",
        "condition": lambda ctx: ctx["state"] == WARMING_UP,
        "advice_en": (
            "Only {n_trials} trials have been evaluated so far "
            "(at least {min_trials} are required). Stagnation detection starts "
            "after the warm-up phase."
        ),
        "advice_zh": (
            "目前仅完成 {n_trials} 次试验（至少需要 {min_trials} 次）。"
            "停滞检测将在预热阶段结束后启动。"
        ),
    },
    {
        "id": "stop_early",
        "severity": "critical",
        "action": "STOP",
        "condition": lambda ctx: (
            ctx["ongoing"]
            and ctx["seg_frac"] > 0.5
            and ctx["severity"] == "critical"
            and ctx["state"] in STALL_STATES
        ),
        "advice_en": (
            "Stagnation has lasted for {seg_frac:.0%} of the run and is still ongoing. "
            "If the optimization budget is tight, stopping this run and starting "
            "a new one is likely more efficient."
        ),
        "advice_zh": (
            "停滞已持续整个运行的 {seg_frac:.0%} 且仍在继续。"
            "若优化预算有限，建议终止本次运行并重新启动，通常更划算。"
        ),
    },
    {
        "id": "local_optimum_critical",
        "severity": "critical",
        "action": "RESTART",
        "condition": lambda ctx: (
            ctx["state"] == LOCAL_OPTIMUM_STALL and ctx["severity"] == "critical"
        ),
        "advice_en": (
            "The optimizer appears trapped in a local optimum: similar configurations "
            "were sampled for {seg_len} trials ({seg_frac:.0%} of the run) without improving "
            "the incumbent (relative improvement {delta:.2%}, novelty rate {e:.0%}). "
            "Restart with a different initialization or reparameterize the search space "
            "(e.g. log-scale transforms)."
        ),
        "advice_zh": (
            "优化器疑似陷入局部最优：{seg_len} 次试验（占整个运行的 {seg_frac:.0%}）都在相似配置附近采样，"
            "未能改进当前最优（相对改进 {delta:.2%}，新颖率 {e:.0%}）。"
            "建议换用不同初始化重启，或对搜索空间重新参数化（如改用对数尺度）。"
        ),
    },
    {
        "id": "ineffective_critical",
        "severity": "critical",
        "action": "EXPLOIT",
        "condition": lambda ctx: (
            ctx["state"] == INEFFECTIVE_EXPLORATION and ctx["severity"] == "critical"
        ),
        "advice_en": (
            "Exploration is ineffective: {seg_len} diverse trials "
            "({seg_frac:.0%} of the run, novelty rate {e:.0%}) did not improve the incumbent "
            "(relative improvement {delta:.2%}). Switch to a more exploitative strategy and "
            "focus the remaining budget on promising regions."
        ),
        "advice_zh": (
            "探索收效甚微：{seg_len} 次多样化试验（占整个运行的 {seg_frac:.0%}，新颖率 {e:.0%}）"
            "均未改进当前最优（相对改进 {delta:.2%}）。建议转向更强利用（exploitation）的策略，"
            "把剩余预算集中到有希望的区域。"
        ),
    },
    {
        "id": "local_optimum_stall",
        "severity": "warning",
        "action": "EXPLORE",
        "condition": lambda ctx: ctx["state"] == LOCAL_OPTIMUM_STALL,
        "advice_en": (
            "The search samples near known configurations (novelty rate {e:.0%}) without "
            "improving the incumbent for {seg_len} trials. Increase exploration, e.g. via "
            "more random restarts or a higher exploration factor."
        ),
        "advice_zh": (
            "搜索在已知配置附近反复采样（新颖率 {e:.0%}），已连续 {seg_len} 次试验未改进当前最优。"
            "建议增强探索，例如增加随机重启次数或提高探索系数。"
        ),
    },
    {
        "id": "ineffective_exploration",
        "severity": "warning",
        "action": "EXPLOIT",
        "condition": lambda ctx: ctx["state"] == INEFFECTIVE_EXPLORATION,
        "advice_en": (
            "Diverse configurations are sampled (novelty rate {e:.0%}) but the incumbent has "
            "not improved for {seg_len} trials. Increase exploitation, e.g. lower the "
            "exploration factor, and consider cheaper fidelities to screen configurations."
        ),
        "advice_zh": (
            "采样到的配置足够多样（新颖率 {e:.0%}），但已连续 {seg_len} 次试验未改进当前最优。"
            "建议加强利用，例如降低探索系数，并考虑用更低保真度先筛选配置。"
        ),
    },
    {
        "id": "recovering",
        "severity": "info",
        "action": "CONTINUE",
        "condition": lambda ctx: ctx["state"] == OK and ctx["had_stall"],
        "advice_en": (
            "The search recovered from a previous stagnation interval and is improving "
            "again. The current settings appear to work - continue."
        ),
        "advice_zh": "搜索已从此前的停滞区间恢复并重新开始改进。当前设置有效，可继续运行。",
    },
    {
        "id": "noise_level_progress",
        "severity": "info",
        "action": "CONTINUE",
        "condition": lambda ctx: ctx["state"] == OK and ctx["dist_improving"],
        "advice_en": (
            "The incumbent is flat, but the underlying cost distribution is still improving "
            "(Mann-Whitney U p < 0.05). This looks like noise-level progress - continue, "
            "but keep monitoring."
        ),
        "advice_zh": (
            "当前最优曲线走平，但底层代价分布仍在改善（Mann-Whitney U 检验 p < 0.05）。"
            "这属于噪声量级的进展，可继续运行并持续观察。"
        ),
    },
    {
        "id": "healthy",
        "severity": "info",
        "action": "CONTINUE",
        "condition": lambda ctx: ctx["state"] == OK,
        "advice_en": (
            "No stagnation detected: the incumbent improved by {delta:.2%} within the "
            "last window. Continue."
        ),
        "advice_zh": "未检测到停滞：最近一个窗口内当前最优改进了 {delta:.2%}。请继续运行。",
    },
    {
        "id": "noise_floor",
        "severity": "warning",
        "action": "ADJUST_FIDELITY",
        "condition": lambda ctx: ctx["state"] in STALL_STATES and ctx["m"] > 0,
        "advice_en": (
            "{m} incumbent improvements in the last window stayed below the noise threshold "
            "(eps_noise={eps_noise}). The objective may be too noisy at this budget - "
            "consider a higher fidelity or averaging over more seeds."
        ),
        "advice_zh": (
            "最近窗口内有 {m} 次最优改进，但均低于噪声阈值（eps_noise={eps_noise}）。"
            "当前预算下目标函数噪声可能过大，建议提高保真度或对更多随机种子取平均。"
        ),
    },
    {
        "id": "multi_budget",
        "severity": "info",
        "action": "ADJUST_FIDELITY",
        "condition": lambda ctx: ctx["multi_budget"],
        "advice_en": (
            "This analysis uses a budget below the highest budget of the run. "
            "Stagnation at lower budgets can be normal - switch to the highest "
            "budget for a final assessment."
        ),
        "advice_zh": (
            "当前分析使用的预算低于该运行的最高预算。低预算下出现停滞属于正常现象，"
            "最终评估请切换到最高预算。"
        ),
    },
]


def get_advice(
    result: Dict[str, Any],
    segment: Optional[Dict[str, Any]] = None,
    multi_budget: bool = False,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return matching bilingual advice rules for a detection result.

    The rules are evaluated in the order of `ADVICE_RULES` (highest priority
    first) and the first ``limit`` matches are returned. English and Chinese
    texts are always returned together.

    Parameters
    ----------
    result : Dict[str, Any]
        Detection result, see `StagnationDetector.detect`.
    segment : Optional[Dict[str, Any]], optional
        Segment to give advice for. If not given, the current (ongoing)
        segment or the overall state of the run is used.
    multi_budget : bool, optional
        Whether the analyzed budget is below the highest budget of the run.
    limit : int, optional
        Maximum number of returned rules. By default 3.

    Returns
    -------
    List[Dict[str, Any]]
        Matching rules with ``id``, ``severity``, ``action``, ``advice_en``
        and ``advice_zh``.
    """
    if segment is None:
        segment = result.get("segment")
        if segment is None and result.get("segments"):
            segment = result["segments"][-1]

    last_window = result.get("last_window") or {}
    params = result.get("params") or {}
    n_trials = result.get("n_trials", 0)

    if segment is not None:
        state = segment["type"]
        seg_len = segment.get("length", 0)
        seg_frac = seg_len / n_trials if n_trials > 0 else 0.0
        severity = segment.get("severity", "warning")
        ongoing = bool(segment.get("ongoing", False))
        delta = segment.get("delta", last_window.get("delta"))
        e_value = segment.get("e", last_window.get("e"))
        m_value = segment.get("m", last_window.get("m"))
    else:
        state = result.get("state", WARMING_UP)
        seg_len = 0
        seg_frac = 0.0
        severity = "info"
        ongoing = False
        delta = last_window.get("delta")
        e_value = last_window.get("e")
        m_value = last_window.get("m")

    context: Dict[str, Any] = {
        "state": state,
        "n_trials": n_trials,
        "min_trials": params.get("min_trials", 20),
        "seg_len": seg_len,
        "seg_frac": seg_frac,
        "severity": severity,
        "ongoing": ongoing,
        "delta": delta if delta is not None else 0.0,
        "e": e_value if e_value is not None else 0.0,
        "m": m_value if m_value is not None else 0,
        "k": result.get("window_k_effective", 0),
        "eps_noise": params.get("eps_noise", 1e-3),
        "theta_e": params.get("theta_e", 0.5),
        "had_stall": bool(result.get("had_stall", False)),
        "dist_improving": bool(last_window.get("dist_improving", False)),
        "multi_budget": bool(multi_budget),
    }

    advice: List[Dict[str, Any]] = []
    for rule in ADVICE_RULES:
        try:
            if not rule["condition"](context):
                continue
        except Exception:
            continue

        advice.append(
            {
                "id": rule["id"],
                "severity": rule["severity"],
                "action": rule["action"],
                "advice_en": _format(rule["advice_en"], context),
                "advice_zh": _format(rule["advice_zh"], context),
            }
        )
        if len(advice) >= limit:
            break

    return advice


def annotate_advice(
    result: Dict[str, Any], multi_budget: bool = False, limit: int = 3
) -> Dict[str, Any]:
    """
    Attach bilingual advice to a detection result (in-place and returned).

    The overall advice is stored under ``result["advice"]`` and per-segment
    advice under ``segment["advice"]``.

    Parameters
    ----------
    result : Dict[str, Any]
        Detection result, see `StagnationDetector.detect`.
    multi_budget : bool, optional
        Whether the analyzed budget is below the highest budget of the run.
    limit : int, optional
        Maximum number of returned rules per context.

    Returns
    -------
    Dict[str, Any]
        The given result with attached advice.
    """
    result["advice"] = get_advice(result, multi_budget=multi_budget, limit=limit)
    for segment in result.get("segments", []):
        segment["advice"] = get_advice(
            result, segment=segment, multi_budget=multi_budget, limit=limit
        )
    return result
