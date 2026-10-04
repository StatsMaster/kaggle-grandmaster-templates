"""Central knobs for the tabular pipeline. Edit this, not train.py."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TabularConfig:
    # --- TODO: your data ---
    DATA_DIR: str = "data"          # expects train.csv / test.csv here
    TARGET_COL: str = "target"      # TODO: your target column
    ID_COL: str = "id"              # TODO: your row-id column (dropped from features)
    TASK: str = "binary"            # "binary" | "regression" | "multiclass"

    # --- validation ---
    N_SPLITS: int = 5
    GROUP_COL: Optional[str] = None  # set for GroupKFold (e.g. patient/store id)
    RANDOM_STATE: int = 42

    # --- tuning (cheap: single fold, subsample) ---
    N_TRIALS: int = 40
    TUNE_FRAC: float = 0.25         # tune on this fraction of fold-0 train
    TUNE_TIMEOUT_S: int = 1800

    # --- training ---
    SEEDS: List[int] = field(default_factory=lambda: [42, 7, 123])
    EARLY_STOPPING_ROUNDS: int = 200
    NUM_BOOST_ROUND: int = 10000

    # --- base LightGBM params (Optuna refines these) ---
    LGBM_PARAMS: dict = field(default_factory=lambda: {
        "objective": "binary",      # train.py overrides for regression/multiclass
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": -1,
    })

    OUT_DIR: str = "runs"
