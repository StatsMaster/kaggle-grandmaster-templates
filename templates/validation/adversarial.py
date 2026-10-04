"""Adversarial validation: can a model distinguish train rows from test rows?

- AUC ~ 0.5  -> train and test come from the same distribution. K-fold is fine.
- AUC -> 1.0 -> distribution shift. Your validation MUST reproduce the shift
  (see splits.py), and the top features below show you where the drift lives.

Usage:
    python templates/validation/adversarial.py
TODO: DATA_DIR / ID_COL to match your competition.
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("data")
ID_COL = "id"  # TODO


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    features = [c for c in train.columns if c != ID_COL and c in test.columns]
    # TODO: drop the target column if it shares a name pattern; keep only
    # features present in BOTH files.

    X = pd.concat([train[features], test[features]], ignore_index=True)
    y = [0] * len(train) + [1] * len(test)  # 0=train, 1=test

    aucs, imp = [], pd.Series(0.0, index=features)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1},
                      dtr, 1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(roc_auc_score(y[va], m.predict(X.iloc[va])))
        imp += pd.Series(m.feature_importance(importance_type="gain"), index=features) / 5

    print(f"[adversarial] AUC={sum(aucs) / len(aucs):.4f}")
    print("[adversarial] most drifted features:")
    print(imp.sort_values(ascending=False).head(10).round(1))
    print("\nInterpretation: ~0.50 = no shift, K-fold is honest. "
          ">0.65 = shift; fix your validation before tuning anything.")


if __name__ == "__main__":
    main()
