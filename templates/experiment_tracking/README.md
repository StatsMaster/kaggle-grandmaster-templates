# Experiment tracking: run experiments like a budget

**The lesson (finale through-line #4):** compute democratized — Kaggle hands out free GPUs — so the bottleneck moved to *iteration speed and discipline*: fast loops, knowing when to stop, and never re-running an experiment you can't reproduce. Grandmasters log everything because memory lies and leaderboards mislead.

**When to reach for it:** from experiment #1. If you can't diff two runs, you don't have two experiments — you have vibes.

**The tool:** `tracker.py` — a minimal logger (params, metrics, artifacts → local JSONL). Zero dependencies beyond the stdlib. If `mlflow` is installed it mirrors there too, so this drops into existing setups.

```python
from tracker import Experiment

with Experiment("lgbm_baseline", {"lr": 0.05, "num_leaves": 63}) as exp:
    ... train ...
    exp.log_metric("oof_auc", 0.9123, step=fold)
    exp.log_artifact("runs/tabular_20261004/oof.csv")
# -> runs/experiments/lgbm_baseline_<ts>.jsonl
```

**Grandmaster rule:** one experiment = one idea changed. If you changed three things and the score moved, you learned nothing.
