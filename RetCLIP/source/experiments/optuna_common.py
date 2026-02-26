import json
import logging
from pathlib import Path

import optuna


def create_trial_split_exp_dir(
    base_dir: Path, trial: optuna.trial.Trial, split: int
) -> Path:
    trial_dir = base_dir / f"trial_{trial.number:03d}"
    split_dir = trial_dir / f"split_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def read_best_metric(exp_dir: Path) -> float:
    p = exp_dir / "val_metrics.json"
    if not p.is_file():
        logging.warning(f"Missing val_metrics.json at {p}")
        return float("nan")
    with open(p, "r") as f:
        j = json.load(f)
    return float(j.get("best_metric", float("nan")))
