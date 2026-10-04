"""Honest splits: match your validation to how the test set was generated.

- Temporal data (sales, clicks, prices) -> time-based split. Random K-fold
  leaks the future and lies to you.
- Non-independent rows (patients, stores, users) -> group split. Random K-fold
  puts the same entity in train and valid and lies to you.
- Everything else -> stratified/random K-fold.

Usage:
    python templates/validation/splits.py
TODO: set the columns for your competition.
"""
import numpy as np
import pandas as pd

# --- TODO: your setup ---
TIME_COL = None      # e.g. "date" -> enables time-based split
GROUP_COL = None     # e.g. "user_id" -> enables group split
TARGET_COL = "target"
N_SPLITS = 5
SEED = 42


def time_split(df, time_col, n_splits=N_SPLITS):
    """Expanding-window splits: train on the past, validate on the future."""
    df = df.sort_values(time_col).reset_index(drop=True)
    cuts = np.linspace(0, len(df), n_splits + 1, dtype=int)
    for i in range(1, n_splits + 1):
        yield (np.arange(cuts[i - 1]), np.arange(cuts[i - 1], cuts[i]))


def group_split(df, group_col, n_splits=N_SPLITS, seed=SEED):
    from sklearn.model_selection import GroupKFold
    groups = df[group_col].values
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None
    yield from GroupKFold(n_splits).split(df, y, groups)


def main():
    # TODO: load your train.csv here.
    df = pd.DataFrame({"target": np.random.randint(0, 2, 1000)})  # placeholder
    if TIME_COL:
        n = sum(1 for _ in time_split(df, TIME_COL))
        print(f"[splits] time-based expanding window: {n} splits on '{TIME_COL}'")
    elif GROUP_COL:
        n = sum(1 for _ in group_split(df, GROUP_COL))
        print(f"[splits] group K-fold: {n} splits on '{GROUP_COL}'")
    else:
        print(f"[splits] stratified/random {N_SPLITS}-fold "
              f"(set TIME_COL or GROUP_COL if rows aren't i.i.d.)")
    print("[splits] rule: if your split doesn't mirror test generation, "
          "your CV score is a rumor.")


if __name__ == "__main__":
    main()
