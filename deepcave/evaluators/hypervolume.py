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
# Hypervolume

This module provides pure numpy utilities to compute the hypervolume indicator
and its convergence over time.

All points are expected in *minimization* space (smaller is better).
Use :func:`to_minimization` to convert objectives which are maximized.

The exact algorithm is selected based on the number of objectives:
- 1D: closed form.
- 2D: exact sweep in O(n log n).
- 3D: exact sweep over the third objective (slices are solved exactly in 2D).
- 4D and higher: exact WFG algorithm (While, Bradstreet, Barone 2012). If the
  Pareto front grows beyond ``mc_threshold`` points, a Monte Carlo approximation
  is used instead to bound the runtime.

## Functions
    - to_minimization: Flip objectives which are maximized.
    - dominates: Check whether a point dominates another (minimization).
    - pareto_filter: Return the non-dominated subset of points.
    - hv_2d: Exact hypervolume for 2 objectives.
    - hv_3d: Exact hypervolume for 3 objectives.
    - hv_wfg: Exact hypervolume for an arbitrary number of objectives (WFG).
    - hv_monte_carlo: Monte Carlo hypervolume approximation.
    - compute_hypervolume: Dispatch to the appropriate algorithm.
    - make_ref_point: Build a reference point from the joint worst values.
    - normalize_points: Normalize points to the unit box.
    - hypervolume_series: Incremental hypervolume after each trial.
    - auc_trapezoid: Trapezoidal AUC of a series.
    - time_to_threshold: First time a series reaches a fraction of its final value.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "to_minimization",
    "dominates",
    "pareto_filter",
    "hv_2d",
    "hv_3d",
    "hv_wfg",
    "hv_monte_carlo",
    "compute_hypervolume",
    "make_ref_point",
    "normalize_points",
    "hypervolume_series",
    "auc_trapezoid",
    "time_to_threshold",
]


def to_minimization(
    costs: Union[np.ndarray, Sequence[float]], optimize: Union[List[str], str]
) -> np.ndarray:
    """
    Convert costs to minimization space.

    Objectives with ``optimize == "upper"`` are maximized and therefore negated.

    Parameters
    ----------
    costs : Union[np.ndarray, Sequence[float]]
        A single cost vector (d,) or a matrix (n, d).
    optimize : Union[List[str], str]
        Either a single value or one value per objective: "lower" or "upper".

    Returns
    -------
    np.ndarray
        The costs in minimization space.
    """
    points = np.atleast_2d(np.asarray(costs, dtype=float))
    if isinstance(optimize, str):
        # A single value applies to all objectives.
        optimize = [optimize] * points.shape[1]

    signs = np.array([-1.0 if o == "upper" else 1.0 for o in optimize])
    if points.shape[1] != len(signs):
        raise ValueError(
            f"Number of objectives ({points.shape[1]}) does not match optimize ({len(signs)})."
        )

    return points * signs


def dominates(point: np.ndarray, other: np.ndarray) -> bool:
    """
    Check whether ``point`` dominates ``other`` in minimization space.

    Parameters
    ----------
    point : np.ndarray
        The first point (d,).
    other : np.ndarray
        The second point (d,).

    Returns
    -------
    bool
        True if ``point`` is not worse in all and better in at least one objective.
    """
    point = np.asarray(point, dtype=float)
    other = np.asarray(other, dtype=float)
    return bool(np.all(point <= other) and np.any(point < other))


def _feasible(points: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Return only points which contribute to the hypervolume (all coordinates < ref)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    ref_point = np.asarray(ref_point, dtype=float)
    return points[np.all(points < ref_point, axis=1)]


def pareto_filter(points: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    """
    Return the non-dominated subset of points (minimization space).

    Duplicates and dominated points are removed. The order is not guaranteed.

    Parameters
    ----------
    points : Union[np.ndarray, Sequence[Sequence[float]]]
        The points (n, d).

    Returns
    -------
    np.ndarray
        The Pareto front (k, d) with k <= n.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) == 0:
        return points

    # Sort lexicographically for a faster O(n^2) filter with early exit.
    order = np.lexsort(tuple(points[:, i] for i in range(points.shape[1] - 1, -1, -1)))
    points = points[order]

    front: List[np.ndarray] = []
    for i, point in enumerate(points):
        dominated = False
        for front_point in front:
            if np.all(front_point <= point):
                dominated = True
                break
        if not dominated:
            front.append(point)
    return np.array(front)


def hv_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute the exact hypervolume for two objectives in O(n log n).

    Parameters
    ----------
    front : np.ndarray
        Points (n, 2) in minimization space. Dominated points are allowed.
    ref_point : np.ndarray
        The reference point (2,).

    Returns
    -------
    float
        The hypervolume.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    front = pareto_filter(_feasible(front, ref_point))
    if len(front) == 0:
        return 0.0

    # Sort by first objective ascending; on a Pareto front the second objective
    # is then strictly decreasing.
    order = np.lexsort((front[:, 1], front[:, 0]))
    front = front[order]

    hv = 0.0
    for i in range(len(front)):
        x_next = front[i + 1, 0] if i + 1 < len(front) else ref_point[0]
        hv += (x_next - front[i, 0]) * (ref_point[1] - front[i, 1])
    return float(hv)


def hv_3d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute the exact hypervolume for three objectives via slicing.

    The front is swept along the third objective; every slab is solved exactly
    in 2D. Complexity is O(n^2 log n), which is fast enough for typical run
    histories.

    Parameters
    ----------
    front : np.ndarray
        Points (n, 3) in minimization space. Dominated points are allowed.
    ref_point : np.ndarray
        The reference point (3,).

    Returns
    -------
    float
        The hypervolume.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    front = pareto_filter(_feasible(front, ref_point))
    if len(front) == 0:
        return 0.0

    order = np.argsort(front[:, 2], kind="stable")
    front = front[order]
    zs = front[:, 2]

    hv = 0.0
    active = np.empty((0, 3))
    previous_z: Optional[float] = None
    for z in np.unique(zs):
        if previous_z is not None:
            hv += (z - previous_z) * hv_2d(active[:, :2], ref_point[:2])
        active = np.vstack([active, front[zs == z]])
        previous_z = float(z)

    if previous_z is not None:
        hv += (ref_point[2] - previous_z) * hv_2d(active[:, :2], ref_point[:2])
    return float(hv)


def hv_wfg(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute the exact hypervolume for an arbitrary number of objectives (WFG).

    Implements the WFG algorithm (While, Bradstreet, Barone 2012) using the
    incremental inclusion-exclusion scheme. Worst case complexity is
    exponential in the number of objectives; therefore it should only be used
    for a moderate number of dimensions and front sizes.

    Parameters
    ----------
    front : np.ndarray
        Points (n, d) in minimization space. Dominated points are allowed.
    ref_point : np.ndarray
        The reference point (d,).

    Returns
    -------
    float
        The hypervolume.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    front = pareto_filter(_feasible(front, ref_point))
    if len(front) == 0:
        return 0.0

    # Sort ascending on the first objective (recommended for the WFG slicing).
    order = np.lexsort(tuple(front[:, i] for i in range(front.shape[1] - 1, -1, -1)))
    return _wfg_recursive(front[order], ref_point)


def _wfg_recursive(points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Recursive helper for :func:`hv_wfg`.

    Uses the incremental scheme: the hypervolume of a point set is the sum of
    the exclusive volumes of its points, where the exclusive volume of point
    k is measured against the union of the *previously processed* points
    (telescological inclusion-exclusion). The clipped point sets are pruned to
    their non-dominated subset, which is essential for practical performance.
    """
    n = len(points)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.prod(ref_point - points[0]))

    hv = 0.0
    processed: List[np.ndarray] = []
    for k in range(n):
        point = points[k]
        hv += float(np.prod(ref_point - point))
        if processed:
            # Intersect the processed boxes [p_j, ref] with the box of the
            # current point [point, ref]. In minimization space the boxes
            # extend to the reference point, hence the elementwise maximum
            # (in maximization space this corresponds to the "limit" minimum).
            limited = np.maximum(np.array(processed), point)
            limited = limited[np.all(limited < ref_point, axis=1)]
            if len(limited) > 0:
                limited = pareto_filter(limited)
                if len(limited) > 0:
                    hv -= _wfg_recursive(limited, ref_point)
        processed.append(point)
    return hv


def hv_monte_carlo(
    front: np.ndarray, ref_point: np.ndarray, n_samples: int = 100_000, seed: int = 0
) -> float:
    """
    Approximate the hypervolume via Monte Carlo sampling.

    Samples are drawn uniformly from the box spanned by the ideal point (the
    per-objective minimum of the front) and the reference point. The
    hypervolume is the box volume times the fraction of dominated samples.

    Parameters
    ----------
    front : np.ndarray
        Points (n, d) in minimization space. Dominated points are allowed.
    ref_point : np.ndarray
        The reference point (d,).
    n_samples : int
        The number of samples. Default is 100_000.
    seed : int
        The seed for the random generator. Default is 0.

    Returns
    -------
    float
        The approximated hypervolume.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    front = pareto_filter(_feasible(front, ref_point))
    if len(front) == 0:
        return 0.0

    lows = front.min(axis=0)
    box_volume = float(np.prod(ref_point - lows))

    rng = np.random.default_rng(seed)
    dominated = np.zeros(n_samples, dtype=bool)
    chunk_size = 10_000
    offset = 0
    remaining = n_samples
    while remaining > 0:
        size = min(chunk_size, remaining)
        samples = rng.uniform(low=lows, high=ref_point, size=(size, len(ref_point)))
        for point in front:
            dominated[offset : offset + size] |= np.all(samples >= point, axis=1)
        offset += size
        remaining -= size

    return float(box_volume * dominated.mean())


def compute_hypervolume(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    ref_point: Union[np.ndarray, Sequence[float]],
    mc_threshold: int = 500,
    mc_samples: int = 100_000,
    seed: int = 0,
) -> Tuple[float, str]:
    """
    Compute the hypervolume and dispatch to the appropriate algorithm.

    Parameters
    ----------
    points : Union[np.ndarray, Sequence[Sequence[float]]]
        Points (n, d) in minimization space. Dominated points are allowed.
    ref_point : Union[np.ndarray, Sequence[float]]
        The reference point (d,).
    mc_threshold : int
        For d >= 4, use Monte Carlo if the Pareto front exceeds this many
        points. Default is 500.
    mc_samples : int
        The number of Monte Carlo samples. Default is 100_000.
    seed : int
        The seed for the Monte Carlo fallback. Default is 0.

    Returns
    -------
    Tuple[float, str]
        The hypervolume and a human readable label of the algorithm tier.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    front = pareto_filter(_feasible(np.asarray(points, dtype=float), ref_point))
    d = ref_point.shape[0]

    if len(front) == 0:
        return 0.0, "empty front"

    if d == 1:
        return float(ref_point[0] - front[:, 0].min()), "1D exact"

    if d == 2:
        return hv_2d(front, ref_point), "2D exact"

    if d == 3:
        return hv_3d(front, ref_point), "3D exact"

    if len(front) > mc_threshold:
        return hv_monte_carlo(front, ref_point, n_samples=mc_samples, seed=seed), (
            f"Monte Carlo approx (d={d}, N={len(front)}, {mc_samples} samples)"
        )

    return hv_wfg(front, ref_point), f"WFG exact (d={d}, N={len(front)})"


def make_ref_point(worst: Union[np.ndarray, Sequence[float]], factor: float = 1.1) -> np.ndarray:
    """
    Build a reference point from the joint worst objective values.

    The factor is applied multiplicatively for positive values and
    divisively for negative values so that the reference point is always
    strictly worse than the worst observation.

    Parameters
    ----------
    worst : Union[np.ndarray, Sequence[float]]
        The per-objective worst (maximal in minimization space) values (d,).
    factor : float
        The slack factor, must be greater than 1. Default is 1.1.

    Returns
    -------
    np.ndarray
        The reference point (d,).
    """
    if factor <= 1.0:
        raise ValueError("The reference point factor must be greater than 1.")

    worst = np.asarray(worst, dtype=float)
    return np.where(worst > 0, worst * factor, worst / factor)


def normalize_points(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    mins: Union[np.ndarray, Sequence[float]],
    maxs: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Normalize points per objective to [0, 1].

    Objectives with zero width (min == max) are mapped to 0.

    Parameters
    ----------
    points : Union[np.ndarray, Sequence[Sequence[float]]]
        The points (n, d) in minimization space.
    mins : Union[np.ndarray, Sequence[float]]
        The per-objective minima (d,).
    maxs : Union[np.ndarray, Sequence[float]]
        The per-objective maxima (d,).

    Returns
    -------
    np.ndarray
        The normalized points (n, d).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)

    ranges = maxs - mins
    zero_width = ranges <= 0
    ranges = np.where(zero_width, 1.0, ranges)
    normalized = (points - mins) / ranges
    # Zero-width objectives are pinned to 0.
    normalized[:, zero_width] = 0.0
    return normalized


def hypervolume_series(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    ref_point: Union[np.ndarray, Sequence[float]],
    max_checkpoints: int = 400,
    mc_threshold: int = 500,
    mc_samples: int = 100_000,
    seed: int = 0,
) -> Tuple[List[int], List[float], str]:
    """
    Incrementally compute the hypervolume after each trial (event-driven).

    The trials are given in submission order. The Pareto front is maintained
    incrementally and the hypervolume is recomputed at (up to)
    ``max_checkpoints`` evenly spaced steps to bound the runtime.

    Parameters
    ----------
    points : Union[np.ndarray, Sequence[Sequence[float]]]
        The cost vectors (n, d) in minimization space in submission order.
    ref_point : Union[np.ndarray, Sequence[float]]
        The reference point (d,).
    max_checkpoints : int
        The maximum number of evaluation steps. Default is 400.
    mc_threshold : int
        For d >= 4, use Monte Carlo if the Pareto front exceeds this many
        points. Default is 500.
    mc_samples : int
        The number of Monte Carlo samples. Default is 100_000.
    seed : int
        The seed for the Monte Carlo fallback. Default is 0.

    Returns
    -------
    Tuple[List[int], List[float], str]
        The trial indices at which the hypervolume was evaluated, the
        hypervolume values, and the algorithm label of the last evaluation.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = len(points)
    if n == 0:
        return [0], [0.0], "empty run"

    if n <= max_checkpoints:
        checkpoints = list(range(n))
    else:
        checkpoints = sorted(set(np.linspace(0, n - 1, max_checkpoints).astype(int).tolist()))

    frontier: List[np.ndarray] = []
    indices: List[int] = []
    hvs: List[float] = []
    method = ""

    for i in checkpoints:
        point = points[i]
        if np.all(point < ref_point) and not any(dominates(q, point) for q in frontier):
            frontier = [q for q in frontier if not dominates(point, q)]
            frontier.append(point)

        hv, method = compute_hypervolume(
            frontier, ref_point, mc_threshold=mc_threshold, mc_samples=mc_samples, seed=seed
        )
        indices.append(int(i))
        hvs.append(hv)

    return indices, hvs, method


def auc_trapezoid(
    values: Union[np.ndarray, Sequence[float]], x: Union[np.ndarray, Sequence[float]]
) -> float:
    """
    Compute the area under a series via the trapezoidal rule.

    Parameters
    ----------
    values : Union[np.ndarray, Sequence[float]]
        The series values.
    x : Union[np.ndarray, Sequence[float]]
        The x positions (e.g. normalized times).

    Returns
    -------
    float
        The area under the series. 0.0 if fewer than two points are given.
    """
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(values) < 2:
        return float(values[0]) if len(values) == 1 else 0.0
    return float(np.trapz(values, x))


def time_to_threshold(
    values: Union[np.ndarray, Sequence[float]],
    x: Union[np.ndarray, Sequence[float]],
    fraction: float = 0.9,
) -> Optional[float]:
    """
    Return the first x position at which the series reaches ``fraction`` of its
    final value.

    Parameters
    ----------
    values : Union[np.ndarray, Sequence[float]]
        The (monotonically non-decreasing) series values.
    x : Union[np.ndarray, Sequence[float]]
        The x positions (e.g. wallclock times).
    fraction : float
        The fraction of the final value. Default is 0.9.

    Returns
    -------
    Optional[float]
        The x position or None if the threshold is never reached.
    """
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(values) == 0:
        return None

    threshold = fraction * values[-1]
    hits = np.nonzero(values >= threshold)[0]
    if len(hits) == 0:
        return None
    return float(x[hits[0]])



    front: List[np.ndarray] = []
    for i, point in enumerate(points):
        dominated = False
        for front_point in front:
            if np.all(front_point <= point):
                dominated = True
                break
        if not dominated:
            front.append(point)
    return np.array(front)
