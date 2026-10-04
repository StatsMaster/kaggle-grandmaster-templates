"""Minimal experiment logger: params, metrics, artifacts -> local JSONL.

Zero required dependencies. If mlflow is installed, mirrors there too.

Usage:
    from tracker import Experiment
    with Experiment("my_run", {"lr": 0.05}) as exp:
        exp.log_metric("oof_auc", 0.91)
        exp.log_artifact("path/to/oof.csv")
"""
import json
import shutil
import time
from pathlib import Path

RUNS_DIR = Path("runs/experiments")


class Experiment:
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = RUNS_DIR / f"{name}_{self.ts}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.dir / "events.jsonl"
        self._mlflow = self._try_mlflow()
        self._write({"event": "start", "name": name, "params": self.params})

    def _try_mlflow(self):
        try:
            import mlflow
            mlflow.set_experiment(self.name)
            run = mlflow.start_run(run_name=f"{self.name}_{self.ts}")
            mlflow.log_params(_flatten(self.params))
            return mlflow
        except ImportError:
            return None

    def _write(self, record):
        record["t"] = time.time()
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_metric(self, key, value, step=None):
        self._write({"event": "metric", "key": key, "value": value, "step": step})
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step or 0)

    def log_artifact(self, path):
        dest = self.dir / "artifacts" / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
        self._write({"event": "artifact", "src": str(path), "dest": str(dest)})
        if self._mlflow:
            self._mlflow.log_artifact(path)

    def close(self):
        self._write({"event": "end"})
        if self._mlflow:
            self._mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
