"""Helpers for discovering and loading DeepCAVE native runs."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from deepcave.runs.converters.deepcave import DeepCAVERun


def find_run_directories(root: Union[str, Path]) -> List[Path]:
    """Recursively find folders that look like a DeepCAVE native run.

    A native run must contain ``history.jsonl`` and its sibling files.
    """
    root = Path(root)
    if not root.exists():
        return []

    candidates = []
    for history_file in sorted(root.rglob("history.jsonl")):
        run_dir = history_file.parent
        if (
            (run_dir / "configspace.json").exists()
            and (run_dir / "configs.json").exists()
            and (run_dir / "origins.json").exists()
            and (run_dir / "meta.json").exists()
        ):
            candidates.append(run_dir)
    return candidates


def label_from_path(run_dir: Union[str, Path], root: Optional[Path] = None) -> str:
    """Create a short, human-readable label for a run folder.

    Example structure::

        experiments/digits/SMAC_seed0/run

    is labelled ``digits/SMAC_seed0`` instead of the verbose full relative path.
    """
    run_dir = Path(run_dir).resolve()
    root = root.resolve() if root is not None else run_dir

    if root == run_dir:
        return run_dir.name

    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        return run_dir.name

    parts = list(rel.parts)
    if parts and parts[-1] == "run":
        parts = parts[:-1]
    if not parts:
        return run_dir.name
    return "/".join(parts)


def load_runs(
    root: Union[str, Path],
    labels: Optional[Dict[Path, str]] = None,
) -> Dict[str, DeepCAVERun]:
    """Load every native run below ``root``.

    Returns ``{label: DeepCAVERun}``. Labels come from ``labels`` if provided,
    otherwise from the folder structure.
    """
    root = Path(root)
    run_dirs = find_run_directories(root)
    result: Dict[str, DeepCAVERun] = {}

    for run_dir in run_dirs:
        if labels is not None and run_dir in labels:
            label = labels[run_dir]
        else:
            label = label_from_path(run_dir, root=root)
        result[label] = DeepCAVERun.from_path(run_dir)

    return result
