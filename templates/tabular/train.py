"""Grandmaster tabular pipeline: LightGBM + Optuna + honest OOF + seed averaging.

Usage:
    python templates/tabular/train.py
Artifacts (OOF preds, test preds, importances, params) -> runs/tabular_<timestamp>/

TODO: set DATA_DIR / TARGET_COL / ID_COL / TASK in config.py first.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find config.py next to this file

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from config import TabularConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------------
def load_data(cfg: TabularConfig):
    train = pd.read_csv(Path(cfg.DATA_DIR) / "train.csv")
    test = pd.read_csv(Path(cfg.DATA_DIR) / "test.csv")
    features = [c for c in train.columns if c not in (cfg.TARGET_COL, cfg.ID_COL)]
    # TODO: your feature engineering goes here (or in a features.py you import).
    # Grandmaster rule: iterate features against OOF, never against a single split.
    X, y = train[features], train[cfg.TARGET_COL].values
    return X, y, test[features], test[cfg.ID_COL].values, features


def make_splits(cfg: TabularConfig, y, groups=None):
    if cfg.GROUP_COL:
        return list(GroupKFold(cfg.N_SPLITS).split(y, y, groups))
    if cfg.TASK == "binary":
        return list(StratifiedKFold(cfg.N_SPLITS, shuffle=True,
                                    random_state=cfg.RANDOM_STATE).split(y, y))
    return list(KFold(cfg.N_SPLITS, shuffle=True,
                      random_state=cfg.RANDOM_STATE).split(y))


def objective(trial, cfg, X_tr, y_tr, X_va, y_va):
    params = dict(cfg.LGBM_PARAMS)
    params.update({
        "objective": {"binary": "binary", "regression": "regression",
                      "multiclass": "multiclass"}[cfg.TASK],
        "num_class": len(np.unique(y_tr)) if cfg.TASK == "multiclass" else 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    })
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=cfg.NUM_BOOST_ROUND,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False)])
    pred = model.predict(X_va, num_iteration=model.best_iteration)
    if cfg.TASK == "binary":
        return roc_auc_score(y_va, pred)
    if cfg.TASK == "regression":
        return -mean_squared_error(y_va, pred)  # Optuna maximizes
    return -log_loss(y_va, pred)


def tune(cfg, X, y, splits):
    """Tune cheaply on a subsample of fold 0. Never tune on full CV."""
    tr_idx, va_idx = splits[0]
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    sub = rng.choice(tr_idx, size=int(len(tr_idx) * cfg.TUNE_FRAC), replace=False)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, cfg, X.iloc[sub], y[sub],
                                       X.iloc[va_idx], y[va_idx]),
                   n_trials=cfg.N_TRIALS, timeout=cfg.TUNE_TIMEOUT_S,
                   show_progress_bar=False)
    print(f"[tune] best score={study.best_value:.5f} params={study.best_params}")
    return study.best_params


def train_fold(cfg, params, X, y, tr_idx, va_idx, seed):
    params = dict(params, deterministic=True, seed=seed)
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx])
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=cfg.NUM_BOOST_ROUND,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False)])
    oof_pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    return model, oof_pred


def main():
    cfg = TabularConfig()
    run_dir = Path(cfg.OUT_DIR) / f"tabular_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y, X_test, test_ids, features = load_data(cfg)
    groups = pd.read_csv(Path(cfg.DATA_DIR) / "train.csv")[cfg.GROUP_COL].values \
        if cfg.GROUP_COL else None
    splits = make_splits(cfg, y, groups)

    best_params = tune(cfg, X, y, splits)
    base = dict(cfg.LGBM_PARAMS)
    base.update({
        "objective": {"binary": "binary", "regression": "regression",
                      "multiclass": "multiclass"}[cfg.TASK],
        "metric": {"binary": "auc", "regression": "rmse",
                   "multiclass": "multi_logloss"}[cfg.TASK],
    })
    base.update(best_params)

    n = len(y)
    n_class = len(np.unique(y)) if cfg.TASK == "multiclass" else 1
    oof = np.zeros((n, n_class)) if cfg.TASK == "multiclass" else np.zeros(n)
    test_pred = np.zeros((len(X_test), n_class)) if cfg.TASK == "multiclass" \
        else np.zeros(len(X_test))
    importances = pd.DataFrame(0.0, index=features, columns=["gain"])

    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_preds, fold_test = [], []
        for seed in cfg.SEEDS:  # seed averaging: cheap variance reduction
            model, op = train_fold(cfg, base, X, y, tr_idx, va_idx, seed)
            fold_preds.append(op)
            fold_test.append(model.predict(X_test, num_iteration=model.best_iteration))
            importances["gain"] += model.feature_importance(importance_type="gain") \
                / (len(splits) * len(cfg.SEEDS))
        if cfg.TASK == "multiclass":
            oof[va_idx] = np.mean(fold_preds, axis=0)
            test_pred += np.mean(fold_test, axis=0) / len(splits)
        else:
            oof[va_idx] = np.mean(fold_preds, axis=0)
            test_pred += np.mean(fold_test, axis=0) / len(splits)
        fold_score = roc_auc_score(y[va_idx], oof[va_idx]) if cfg.TASK == "binary" \
            else -mean_squared_error(y[va_idx], oof[va_idx])
        print(f"[fold {fold}] score={fold_score:.5f}")

    if cfg.TASK == "binary":
        print(f"[OOF] auc={roc_auc_score(y, oof):.5f}")
    elif cfg.TASK == "regression":
        print(f"[OOF] rmse={mean_squared_error(y, oof) ** 0.5:.5f}")
    else:
        print(f"[OOF] logloss={log_loss(y, oof):.5f}")

    # --- artifacts: everything downstream (blends, stacks) reads OOF, never refits ---
    if cfg.TASK == "multiclass":
        pd.DataFrame(oof).to_csv(run_dir / "oof.csv", index=False)
        pd.DataFrame(test_pred).to_csv(run_dir / "test_pred.csv", index=False)
    else:
        pd.DataFrame({"oof": oof}).to_csv(run_dir / "oof.csv", index=False)
        pd.DataFrame({"id": test_ids, "pred": test_pred}) \
            .to_csv(run_dir / "submission.csv", index=False)
    importances.sort_values("gain", ascending=False).to_csv(run_dir / "importances.csv")
    (run_dir / "params.json").write_text(json.dumps(base, indent=2, default=str))
    print(f"[done] artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
