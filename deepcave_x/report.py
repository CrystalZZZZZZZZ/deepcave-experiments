"""Generate a Markdown + figure report from an experiment folder."""

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .comparison import TrajectoryComparison
from .importance import random_forest_importance
from .runs import load_runs
from .stagnation import detect_stagnation, plot_stagnation, stagnation_advice
from .trajectory import extract_trajectory


def _write_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), width=1200, height=650, scale=1.2)
    except Exception:
        path = path.with_suffix(".html")
        fig.write_html(str(path))
    return path


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def run_report(
    experiment_root: Path,
    objective_name: str,
    output_dir: Path,
    min_window: int = 5,
    with_importance: bool = True,
) -> Path:
    """Load runs under ``experiment_root`` and produce a diagnostic report."""
    experiment_root = Path(experiment_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, object] = load_runs(experiment_root)
    if not runs:
        raise RuntimeError(f"No DeepCAVE native runs found under {experiment_root}")

    trajectories = {}
    skipped = []
    for label, run in runs.items():
        if run.get_objective(objective_name) is None:
            skipped.append((label, "objective not found"))
            continue
        try:
            trajectories[label] = extract_trajectory(
                run, objective=objective_name
            )
        except Exception as exc:
            skipped.append((label, str(exc)))

    if not trajectories:
        raise RuntimeError(f"No trajectory could be extracted; skipped={skipped}")

    labels = list(trajectories.keys())
    traj_list = [trajectories[label] for label in labels]

    # Multi-optimizer comparison
    comparison = TrajectoryComparison(traj_list, labels, objective_label=objective_name)
    comparison_summary = comparison.summary()
    comparison_fig = comparison.plot(
        title=f"Incumbent trajectory comparison on '{objective_name}'"
    )
    comparison_path = _write_figure(
        comparison_fig, output_dir / "trajectory_comparison.png"
    )

    # Stagnation analysis per run
    stagnation_rows = []
    segment_figures = []
    for label, trajectory in zip(labels, traj_list):
        segments = detect_stagnation(trajectory, min_window=min_window)
        advice = stagnation_advice(trajectory, segments, optimizer_label=label)
        for seg in segments:
            stagnation_rows.append(
                {
                    "label": label,
                    "start_step": seg.start_step,
                    "end_step": seg.end_step,
                    "duration_steps": seg.duration_steps,
                    "relative_gap": seg.relative_gap,
                }
            )
        fig = plot_stagnation(
            trajectory,
            segments,
            title=f"Stagnation: {label} ({objective_name})",
        )
        fig_path = _write_figure(
            fig, output_dir / f"stagnation_{label.replace('/', '_')}.png"
        )
        segment_figures.append((label, fig_path, advice))

    stagnation_df = pd.DataFrame(stagnation_rows)

    # Optional RF importance for each run
    importance_tables = {}
    if with_importance:
        for label, run in runs.items():
            if label not in trajectories:
                continue
            try:
                importance_tables[label] = random_forest_importance(
                    run, objective=objective_name
                )
            except Exception as exc:
                importance_tables[label] = None
                print(f"[warn] importance failed for {label}: {exc}")

    # Markdown report
    lines = [
        "# DeepCAVE-X diagnostic report",
        "",
        f"- experiment root: `{experiment_root}`",
        f"- objective: `{objective_name}`",
        f"- stagnation min window: `{min_window}` trials",
        "",
        "## Trajectory comparison summary",
        "",
    ]
    lines.append(_df_to_markdown(comparison_summary))
    lines += ["", f"![trajectory comparison](./{comparison_path.name})", ""]

    if len(stagnation_df) > 0:
        lines += ["## Stagnation segments", ""]
        lines.append(_df_to_markdown(stagnation_df))
        lines.append("")

    lines += ["## Per-run stagnation figures", ""]
    for label, fig_path, advice in segment_figures:
        lines.append(f"### {label}")
        lines.append(f"![{label}](./{fig_path.name})")
        for item in advice:
            lines.append(f"- {item}")
        lines.append("")

    if importance_tables:
        lines += ["## Random forest importance", ""]
        for label, table in importance_tables.items():
            lines.append(f"### {label}")
            if table is None:
                lines.append("- importance computation failed (see logs)")
            else:
                lines.append(_df_to_markdown(table))
            lines.append("")

    lines += [f"- skipped runs: {skipped}", ""]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--objective", type=str, default="accuracy")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--min-window", type=int, default=5)
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args()

    report_path = run_report(
        experiment_root=args.experiment_root,
        objective_name=args.objective,
        output_dir=args.output,
        min_window=args.min_window,
        with_importance=not args.no_importance,
    )
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
