"""Lean blend: OOF-optimized weights + rank averaging.

Usage:
    python templates/blending/blend.py
TODO: OOF_DIR with one oof.csv per model (columns: pred) + test_<name>.csv files,
      and y_true.npy with the training targets in row order.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# TODO: your OOF predictions, one file per model, same row order.
OOF_DIR = Path("runs/blend_inputs")
TASK = "binary"  # "binary" | "regression"


def load_oofs():
    files = sorted(OOF_DIR.glob("oof_*.csv"))
    assert files, f"no oof_*.csv in {OOF_DIR}"
    names = [f.stem.replace("oof_", "") for f in files]
    mat = np.column_stack([pd.read_csv(f)["pred"].values for f in files])
    y = np.load(OOF_DIR / "y_true.npy")
    print("[blend] OOF correlation:\n", pd.DataFrame(mat, columns=names).corr().round(3))
    return names, mat, y


def score(weights, mat, y):
    w = np.clip(weights, 0, None)
    w /= w.sum()
    blend = mat @ w
    return roc_auc_score(y, blend) if TASK == "binary" else -np.sqrt(np.mean((y - blend) ** 2))


def optimize_weights(mat, y):
    n = mat.shape[1]
    res = minimize(lambda w: -score(w, mat, y), x0=np.ones(n) / n,
                   method="Nelder-Mead", options={"maxiter": 2000})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def rank_average(mat):
    ranks = np.column_stack([pd.Series(col).rank(pct=True).values for col in mat.T])
    return ranks.mean(axis=1)


def main():
    names, mat, y = load_oofs()
    w = optimize_weights(mat, y)
    print("[blend] weights:", dict(zip(names, w.round(4))))
    print(f"[blend] OOF score: {score(w, mat, y):.5f}")
    print(f"[blend] rank-average OOF score: "
          f"{score(np.ones(len(names)) / len(names), np.column_stack([pd.Series(c).rank(pct=True).values for c in mat.T]), y):.5f}")

    # apply to test predictions (test_<name>.csv, same order as oof files)
    test_mat = np.column_stack([
        pd.read_csv(OOF_DIR / f"test_{n}.csv")["pred"].values for n in names])
    blended = test_mat @ w
    pd.DataFrame({"pred": blended}).to_csv("runs/blended_submission.csv", index=False)
    print("[done] runs/blended_submission.csv")


if __name__ == "__main__":
    main()
