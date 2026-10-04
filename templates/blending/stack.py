"""2-level stacking: a meta-model learns which base model to trust where.

Base models' OOF predictions become meta-features. The meta-model trains on
OOF (never on in-fold predictions) and predicts on base models' test preds.

Usage:
    python templates/blending/stack.py
TODO: same OOF_DIR layout as blend.py.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

OOF_DIR = Path("runs/blend_inputs")
TASK = "binary"  # "binary" | "regression"
META_MODEL = "ridge"  # "ridge" | "logreg" | "lgbm"  (keep it simple: linear first)


def load():
    files = sorted(OOF_DIR.glob("oof_*.csv"))
    names = [f.stem.replace("oof_", "") for f in files]
    X_meta = np.column_stack([pd.read_csv(f)["pred"].values for f in files])
    y = np.load(OOF_DIR / "y_true.npy")
    X_test_meta = np.column_stack([
        pd.read_csv(OOF_DIR / f"test_{n}.csv")["pred"].values for n in names])
    return names, X_meta, y, X_test_meta


def make_meta():
    if META_MODEL == "ridge":
        return Ridge(alpha=1.0)
    if META_MODEL == "logreg":
        return LogisticRegression(C=1.0, max_iter=1000)
    # lgbm: small GBDT meta-learner for non-linear trust patterns
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03,
                             num_leaves=7, verbose=-1)


def main():
    names, X_meta, y, X_test_meta = load()
    # Grandmaster default: start linear. A fancy meta-model that overfits OOF
    # is worse than a weighted average — check blend.py first.
    meta = make_meta()
    meta.fit(X_meta, y)
    test_pred = meta.predict(X_test_meta)
    if TASK == "binary":
        test_pred = np.clip(test_pred, 0, 1)
    if hasattr(meta, "coef_"):
        print("[stack] meta coefs:", dict(zip(names, np.round(meta.coef_, 4))))
    pd.DataFrame({"pred": test_pred}).to_csv("runs/stacked_submission.csv", index=False)
    print("[done] runs/stacked_submission.csv")


if __name__ == "__main__":
    main()
