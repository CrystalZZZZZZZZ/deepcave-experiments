# DeepCAVE-X

DeepCAVE-X is an extension package built **on top of DeepCAVE** (as a dependency,
not a fork). It uses DeepCAVE's Python/API layer to read native HPO runs and adds
three diagnostic capabilities planned in the assignment:

1. **Multi-optimizer trajectory comparison**
2. **Hyperparameter importance** (Random Forest on DeepCAVE-encoded data)
3. **Stagnation / bottleneck diagnosis**

The package itself does not run SMAC or Optuna. Trajectory data is expected to be
produced by the baseline notebook
[`01_smac_optuna_deepcave_baseline.ipynb`](01_smac_optuna_deepcave_baseline.ipynb),
which records runs in DeepCAVE native format.

## Structure

```text
codespace/
├── 01_smac_optuna_deepcave_baseline.ipynb   # data generation
├── experiments/                             # produced by the notebook
├── deepcave_x/
│   ├── runs.py              # find/load native runs
│   ├── trajectory.py        # per-trial incumbent trajectory extraction
│   ├── comparison.py        # multi-run trajectory metrics + plot
│   ├── importance.py        # RF importance (plus DeepCAVE plugin bridge)
│   ├── stagnation.py        # plateau detection + suggestions + plot
│   └── report.py            # CLI report generator
└── pyproject.toml
```

## Install

```bash
conda activate deepcave
cd codespace
pip install -e .
```

If you also need the data generation dependencies (SMAC/Optuna/OpenML), install
`requirements.txt` as well.

## Quick usage after running the notebook

```python
from deepcave_x import load_runs, extract_trajectory, TrajectoryComparison

runs = load_runs("experiments/digits")
trajectories = {
    label: extract_trajectory(run, objective="accuracy")
    for label, run in runs.items()
}

comparison = TrajectoryComparison(
    list(trajectories.values()),
    list(trajectories.keys()),
    objective_label="1 - balanced accuracy (lower better)",
)
comparison.summary()
comparison.plot()
```

## CLI report

```bash
python -m deepcave_x.report \
  --experiment-root experiments \
  --objective accuracy \
  --output reports
```

Output: `reports/report.md`, trajectory comparison figure, per-run stagnation
figures and RF importance tables.

## Design note

DeepCAVE is intentionally kept as an external dependency. New modules only call
its public Run/plugin APIs (`DeepCAVERun.from_path`, `get_encoded_data`,
`get_incumbent`, plugin `generate_*`/`load_outputs`), which keeps the package
upgradeable and lets later work register real DeepCAVE plugins if needed.
