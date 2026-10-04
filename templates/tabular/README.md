# Tabular: the grandmaster pipeline

**The lesson (Ensemble Era → still true today):** on tabular data, boosted trees win — 17 of 29 Kaggle-blogged 2015 wins used XGBoost, and LightGBM swept M5 (2020) against every classical forecaster. The model is the easy part. The win is in *honest validation* and *feature iteration speed*.

**When to reach for it:** any tabular competition or dataset. This is your week-1 baseline and usually your final backbone.

**The pipeline:**
1. **Tune cheap, train honest.** Optuna tunes hyperparameters on a single fold (or a data subsample) — never on the full CV, which wastes your experiment budget.
2. **Out-of-fold everything.** Stratified K-fold (classification) or K-fold (regression); GroupKFold when rows aren't independent. Every prediction that feeds a blend or a decision comes from OOF — never from in-fold predictions.
3. **Early stopping per fold**, with the best iteration averaged or re-set per fold.
4. **Seed averaging.** Train each fold with 2–3 seeds and average. Cheap variance reduction; grandmasters do it reflexively.

**Files:**
- `config.py` — all knobs in one dataclass.
- `train.py` — tune → cross-validate → OOF + test predictions → importances.

```bash
# TODO: point DATA_DIR at your competition files first
python templates/tabular/train.py
# artifacts land in ./runs/tabular_<timestamp>/
```
