"""DeepCAVE-X: DeepCAVE-based extension for HPO trajectory diagnosis.

The package treats DeepCAVE as a dependency library: run data is read with
``DeepCAVERun.from_path`` and all analysis is built on top of its Run API.
"""

from .runs import find_run_directories, label_from_path, load_runs
from .trajectory import extract_trajectory
from .comparison import TrajectoryComparison, summarize_trajectories
from .importance import random_forest_importance
from .stagnation import StagnationSegment, detect_stagnation

__version__ = "0.1.0"

__all__ = [
    "find_run_directories",
    "label_from_path",
    "load_runs",
    "extract_trajectory",
    "TrajectoryComparison",
    "summarize_trajectories",
    "random_forest_importance",
    "StagnationSegment",
    "detect_stagnation",
]
