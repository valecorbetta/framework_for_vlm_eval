import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import optuna
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig


def create_split_exp_dir(base_dir: Path, trial: optuna.trial.Trial, split: int) -> Path:
    """
    Each trial has its own subfolder, and each split has a sub-subfolder.
    e.g., BASE_DIR/trial_000/split_0
    """
    trial_dir = base_dir / f"trial_{trial.number:03d}"
    split_dir = trial_dir / f"split_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def _to_list(x):
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


def suggest_lora_hparams(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    """
    Mirrors your previous suggest_lora_hparams exactly.
    Writes into cfg.MODEL._class.* and cfg.TRAIN.*.
    """
    opt_cfg = cfg.OPTUNA

    r_choices = _to_list(opt_cfg.lora_r)
    r = trial.suggest_categorical("lora_r", r_choices)

    alpha_mult = float(getattr(opt_cfg, "lora_alpha_mult", 2.0))
    alpha = alpha_mult * int(r)

    dropout_choices = _to_list(opt_cfg.lora_dropout)
    dropout = trial.suggest_categorical("lora_dropout", dropout_choices)

    lr_cls_choices = _to_list(opt_cfg.lr_classifier)
    lr_lora_choices = _to_list(opt_cfg.lr_lora)

    lr_classifier = trial.suggest_categorical("lr_classifier", lr_cls_choices)
    lr_lora = trial.suggest_categorical("lr_lora", lr_lora_choices)

    # write into cfg (match your old keys; you used cfg.MODEL._class.* elsewhere)
    cfg.MODEL._class.lora_r = int(r)
    cfg.MODEL._class.lora_alpha = float(alpha)
    cfg.MODEL._class.lora_dropout = float(dropout)

    cfg.TRAIN.lr_classifier = float(lr_classifier)
    cfg.TRAIN.lr_lora = float(lr_lora)

    return cfg


def apply_best_params(best_trial: optuna.trial.FrozenTrial, cfg: DictConfig):
    params = best_trial.params
    if "lora_r" in params:
        cfg.MODEL._class.lora_r = int(params["lora_r"])
        alpha_mult = float(getattr(cfg.OPTUNA, "lora_alpha_mult", 2.0))
        cfg.MODEL._class.lora_alpha = float(alpha_mult * cfg.MODEL._class.lora_r)
    if "lora_dropout" in params:
        cfg.MODEL._class.lora_dropout = float(params["lora_dropout"])
    if "lr_classifier" in params:
        cfg.TRAIN.lr_classifier = float(params["lr_classifier"])
    if "lr_lora" in params:
        cfg.TRAIN.lr_lora = float(params["lr_lora"])


def objective_trainer(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    num_splits: int,
    device,
    exp_dir: Path,
    train_csvs: List[Path],
    val_csvs: List[Path],
    path_to_images: Path,
    overlay_cfg_train: DictConfig,
    overlay_cfg_test: Optional[DictConfig],
    build_trainer_fn: Callable[..., object],
) -> float:
    """
    Replicates your previous objective():
      - deep-copy cfg and suggest LoRA hparams
      - run across splits, log + read best metric
      - report running mean each split
      - prune after split >= 1
    """

    # Important: detach from Hydra / interpolation
    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_local = suggest_lora_hparams(trial, cfg_local)

    epochs_search = int(
        getattr(cfg_local.OPTUNA, "epochs_search", cfg_local.TRAIN.epochs)
    )
    cfg_local.TRAIN.epochs = int(epochs_search)

    metrics_per_split: List[float] = []

    for split in range(num_splits):
        logging.info(f"---------- Split {split} ----------")
        csv_train = train_csvs[split]
        csv_val = val_csvs[split]

        split_exp_dir = create_split_exp_dir(exp_dir, trial, split)
        seed = int(cfg_local.TRAIN.seed) + int(split)

        # Build trainer for this split
        trainer = build_trainer_fn(
            cfg=cfg_local,
            device=device,
            exp_dir=split_exp_dir,
            seed=seed,
            split_number=split,
            csv_train_path=csv_train,
            csv_val_path=csv_val,
            csv_test_path=None,
            path_to_images=path_to_images,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=(
                overlay_cfg_test if overlay_cfg_test is not None else overlay_cfg_train
            ),
        )

        # Run train/val
        # We support two patterns:
        # (A) trainer.run() returns a dict with best validation metric
        # (B) trainer.run() writes val_metrics.json with {"best_metric": ...}
        out = trainer.fit()
        metrics_path = split_exp_dir / "val_metrics.json"
        if metrics_path.is_file():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            best_metric = float(metrics.get("best_metric", 0.0))
        if best_metric is None:
            raise RuntimeError(
                f"Could not obtain best_metric for split={split}. "
                f"Return dict keys={list(out.keys()) if isinstance(out, dict) else type(out)} "
                f"and no val_metrics.json found."
            )

        metrics_per_split.append(best_metric)

        running_mean = float(sum(metrics_per_split) / len(metrics_per_split))
        trial.report(running_mean, step=split)

        if split >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    final_mean_metric = float(sum(metrics_per_split) / len(metrics_per_split))
    return final_mean_metric


# RetCLIP/source/utils/optuna_mica_utils.py

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna.trial import TrialState

from RetCLIP.source.utils.trainers.base_trainer import CheckpointPaths
from RetCLIP.source.utils.models import _run_single_fit
from RetCLIP.source.utils.misc import (
    _read_best_metric,
    _discover_best_paths,
    _save_json,
)


def create_split_exp_dir(base_dir: Path, trial: optuna.trial.Trial, split: int) -> Path:
    trial_dir = base_dir / f"trial_{trial.number:03d}"
    split_dir = trial_dir / f"split_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def _to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def apply_fundus_best_params(cfg: DictConfig, fundus_best_params: Dict) -> DictConfig:
    """
    Write the FundusClassifier best params into cfg (LoRA rank/dropout/lrs etc),
    so MICA stage1/2 share the same baseline setting.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    p = fundus_best_params

    if "lora_r" in p:
        cfg.MODEL._class.lora_r = int(p["lora_r"])
        alpha_mult = float(getattr(cfg.OPTUNA, "lora_alpha_mult", 2.0))
        cfg.MODEL._class.lora_alpha = float(alpha_mult * cfg.MODEL._class.lora_r)
    if "lora_dropout" in p:
        cfg.MODEL._class.lora_dropout = float(p["lora_dropout"])
    if "lr_classifier" in p:
        cfg.TRAIN.lr_classifier = float(p["lr_classifier"])
    if "lr_lora" in p:
        cfg.TRAIN.lr_lora = float(p["lr_lora"])

    return cfg


def suggest_mica_stage1_params(
    trial: optuna.trial.Trial, cfg: DictConfig
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    space = cfg.OPTUNA.mica.stage1

    cfg.MODEL.losses.temp1 = float(
        trial.suggest_categorical("temp1", _to_list(space.temp1))
    )
    cfg.MODEL.losses.temp2 = float(
        trial.suggest_categorical("temp2", _to_list(space.temp2))
    )
    cfg.MODEL.losses.temp3 = float(
        trial.suggest_categorical("temp3", _to_list(space.temp3))
    )

    cfg.MODEL.losses.local_loss_weight = float(
        trial.suggest_categorical(
            "local_loss_weight", _to_list(space.local_loss_weight)
        )
    )
    cfg.MODEL.losses.global_loss_weight = float(
        trial.suggest_categorical(
            "global_loss_weight", _to_list(space.global_loss_weight)
        )
    )
    cfg.MODEL.losses.concept_loss_weight = float(
        trial.suggest_categorical(
            "concept_loss_weight", _to_list(space.concept_loss_weight)
        )
    )

    return cfg


def suggest_mica_stage2_params(
    trial: optuna.trial.Trial, cfg: DictConfig
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    space = cfg.OPTUNA.mica.stage2
    cfg.MICA.stage2.concept_weight = float(
        trial.suggest_categorical("concept_weight", _to_list(space.concept_weight))
    )
    return cfg


def _extract_best_metric_from_val_json(val_metrics_path: Path) -> float:
    if not val_metrics_path.is_file():
        return float("nan")
    with open(val_metrics_path, "r") as f:
        d = json.load(f)
    # match your BaseTrainer’s output contract:
    # expected key "best_metric" or compute from "val_metrics"
    if "best_metric" in d:
        return float(d["best_metric"])
    if "metric" in d:
        return float(d["metric"])
    return float("nan")


def _run_stage_trainer_fit(
    *,
    cfg: DictConfig,
    device: torch.device,
    exp_dir: Path,
    seed: int,
    split_number: int,
    csv_train_path: Path,
    csv_val_path: Path,
    csv_test_path: Optional[Path],
    path_to_images: Path,
    overlay_cfg_train: DictConfig,
    overlay_cfg_test: DictConfig,
    stage: int,
) -> CheckpointPaths:
    if stage == 1:
        trainer_cfg = cfg.MICA.stage1.trainer
    elif stage == 2:
        trainer_cfg = cfg.MICA.stage2.trainer
    else:
        raise ValueError(stage)

    trainer = instantiate(
        trainer_cfg,
        cfg=cfg,
        device=device,
        exp_dir=exp_dir,
        seed=seed,
        split_number=split_number,
        csv_train_path=csv_train_path,
        csv_val_path=csv_val_path,
        csv_test_path=csv_test_path,
        path_to_images=path_to_images,
        overlay_cfg_train=overlay_cfg_train,
        overlay_cfg_test=overlay_cfg_test,
    )
    return trainer.fit()


def objective_mica_stage1(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    device: torch.device,
    base_dir: Path,
    num_splits: int,
    train_csvs: List[Path],
    val_csvs: List[Path],
    images_root: Path,
    overlay_cfg_train: DictConfig,
) -> float:
    cfg = suggest_mica_stage1_params(trial, cfg)
    cfg.TRAIN.epochs = int(getattr(cfg.OPTUNA.mica, "epochs_search", cfg.TRAIN.epochs))

    metrics = []
    for split in range(num_splits):
        split_dir = create_split_exp_dir(base_dir, trial, split)
        seed = int(cfg.TRAIN.seed) + split

        _ = _run_stage_trainer_fit(
            cfg=cfg,
            device=device,
            exp_dir=split_dir,
            seed=seed,
            split_number=split,
            csv_train_path=train_csvs[split],
            csv_val_path=val_csvs[split],
            csv_test_path=None,
            path_to_images=images_root,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=overlay_cfg_train,
            stage=1,
        )

        # Your stage1 trainer should write val_metrics.json in split_dir
        metric = _extract_best_metric_from_val_json(split_dir / "val_metrics.json")
        metrics.append(metric)

        running = float(np.nanmean(metrics))
        trial.report(running, step=split)
        if split >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(metrics))


def objective_mica_stage2(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    device: torch.device,
    base_dir: Path,
    num_splits: int,
    train_csvs: List[Path],
    val_csvs: List[Path],
    images_root: Path,
    overlay_cfg_train: DictConfig,
    stage1_best_artifacts: Dict[str, Dict[int, str]],
) -> float:
    """
    stage1_best_artifacts:
      {
        "stage1_ckpt": {split: "/path/to/best_stage1.ckpt", ...}  OR None
        "stage1_lora_dir": {split: "/path/to/best_lora_adapter", ...} OR None
      }
    """
    cfg = suggest_mica_stage2_params(trial, cfg)
    cfg.TRAIN.epochs = int(getattr(cfg.OPTUNA.mica, "epochs_search", cfg.TRAIN.epochs))

    metrics = []
    for split in range(num_splits):
        split_dir = create_split_exp_dir(base_dir, trial, split)
        seed = int(cfg.TRAIN.seed) + split

        # Inject stage1 artifact for this split into cfg.MICA.stage2
        cfg_split = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_split.MICA.stage2.stage1_ckpt = None
        cfg_split.MICA.stage2.stage1_lora_dir = None

        if (
            "stage1_lora_dir" in stage1_best_artifacts
            and split in stage1_best_artifacts["stage1_lora_dir"]
        ):
            cfg_split.MICA.stage2.stage1_lora_dir = stage1_best_artifacts[
                "stage1_lora_dir"
            ][split]
        elif (
            "stage1_ckpt" in stage1_best_artifacts
            and split in stage1_best_artifacts["stage1_ckpt"]
        ):
            cfg_split.MICA.stage2.stage1_ckpt = stage1_best_artifacts["stage1_ckpt"][
                split
            ]

        _ = _run_stage_trainer_fit(
            cfg=cfg_split,
            device=device,
            exp_dir=split_dir,
            seed=seed,
            split_number=split,
            csv_train_path=train_csvs[split],
            csv_val_path=val_csvs[split],
            csv_test_path=None,
            path_to_images=images_root,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=overlay_cfg_train,
            stage=2,
        )

        metric = _extract_best_metric_from_val_json(split_dir / "val_metrics.json")
        metrics.append(metric)

        running = float(np.nanmean(metrics))
        trial.report(running, step=split)
        if split >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(metrics))


# -----------------------------------------------------------------------------
# Optuna: generic single-trainer CV objective
# -----------------------------------------------------------------------------
def _optuna_objective_single_trainer(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    *,
    device: torch.device,
    exp_dir: Path,
    num_splits: int,
    train_csvs: List[Path],
    val_csvs: List[Path],
    images_root: Path,
    overlay_cfg_train: DictConfig,
    trainer_cfg: DictConfig,
    suggest_fn,
    epochs_search: int,
) -> float:
    """
    Runs K-split CV for one trial:
      - suggests params into a fresh cfg copy
      - fits trainer per split
      - reads val_metrics.json per split
      - reports running mean for pruning
      - returns mean metric
    """
    cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_t = suggest_fn(trial, cfg_t)
    cfg_t.TRAIN.epochs = int(epochs_search)

    split_metrics: List[float] = []
    for split in range(num_splits):
        split_dir = exp_dir / f"trial_{trial.number:03d}" / f"split_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)

        _ = _run_single_fit(
            trainer_cfg=trainer_cfg,
            cfg=cfg_t,
            device=device,
            exp_dir=split_dir,
            seed=int(cfg_t.TRAIN.seed) + split,
            split_number=split,
            csv_train_path=train_csvs[split],
            csv_val_path=val_csvs[split],
            csv_test_path=None,
            path_to_images=images_root,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=overlay_cfg_train,
        )

        m = _read_best_metric(split_dir)
        split_metrics.append(m)

        running = float(np.nanmean(split_metrics))
        trial.report(running, step=split)
        if split >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(split_metrics))


# -----------------------------------------------------------------------------
# Optuna: MICA two-stage optimization (Strategy B)
# -----------------------------------------------------------------------------
def _mica_suggest_stage1(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    space = cfg.OPTUNA.mica.stage1
    cfg.MODEL.losses.temp1 = float(
        trial.suggest_categorical("temp1", list(space.temp1))
    )
    cfg.MODEL.losses.temp2 = float(
        trial.suggest_categorical("temp2", list(space.temp2))
    )
    cfg.MODEL.losses.temp3 = float(
        trial.suggest_categorical("temp3", list(space.temp3))
    )

    cfg.MODEL.losses.local_loss_weight = float(
        trial.suggest_categorical("local_loss_weight", list(space.local_loss_weight))
    )
    cfg.MODEL.losses.global_loss_weight = float(
        trial.suggest_categorical("global_loss_weight", list(space.global_loss_weight))
    )
    cfg.MODEL.losses.concept_loss_weight = float(
        trial.suggest_categorical(
            "concept_loss_weight", list(space.concept_loss_weight)
        )
    )
    return cfg


def _mica_suggest_stage2(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    space = cfg.OPTUNA.mica.stage2
    cfg.MICA.stage2.concept_weight = float(
        trial.suggest_categorical("concept_weight", list(space.concept_weight))
    )
    return cfg


def _build_mica_stage1_best_artifacts(
    *,
    cfg_stage1_best: DictConfig,
    device: torch.device,
    out_dir: Path,
    num_splits: int,
    train_csvs: List[Path],
    val_csvs: List[Path],
    images_root: Path,
    overlay_cfg_train: DictConfig,
) -> Dict[str, Dict[int, str]]:
    """
    Run Stage-1 training once per split using the Stage-1 best params and
    save canonical artifacts into out_dir/split_{k}/...

    Returns:
      {"stage1_ckpt": {split: path_str}, "stage1_lora_dir": {split: path_str}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Dict[int, str]] = {"stage1_ckpt": {}, "stage1_lora_dir": {}}

    for split in range(num_splits):
        split_dir = out_dir / f"split_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)

        _ = _run_single_fit(
            trainer_cfg=cfg_stage1_best.MICA.stage1.trainer,
            cfg=cfg_stage1_best,
            device=device,
            exp_dir=split_dir,
            seed=int(cfg_stage1_best.TRAIN.seed) + split,
            split_number=split,
            csv_train_path=train_csvs[split],
            csv_val_path=val_csvs[split],
            csv_test_path=None,
            path_to_images=images_root,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=overlay_cfg_train,
        )

        ckpt_dir = split_dir / "checkpoints"
        cp = _discover_best_paths(ckpt_dir)
        if cp.best_lora_dir is not None:
            artifacts["stage1_lora_dir"][split] = str(cp.best_lora_dir)
        if cp.best_stage2_ckpt is not None:
            artifacts["stage1_ckpt"][split] = str(cp.best_stage2_ckpt)

    # remove empty maps to keep logs tidy
    if len(artifacts["stage1_lora_dir"]) == 0:
        artifacts.pop("stage1_lora_dir")
    if len(artifacts["stage1_ckpt"]) == 0:
        artifacts.pop("stage1_ckpt")

    _save_json(out_dir / "stage1_best_artifacts.json", artifacts)
    return artifacts


def _optuna_objective_mica_stage2(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    *,
    device: torch.device,
    exp_dir: Path,
    num_splits: int,
    train_csvs: List[Path],
    val_csvs: List[Path],
    images_root: Path,
    overlay_cfg_train: DictConfig,
    stage1_best_artifacts: Dict[str, Dict[int, str]],
    epochs_search: int,
) -> float:
    """
    Stage-2 objective: only tunes cfg.MICA.stage2.concept_weight and uses fixed
    Stage-1 artifacts per split (ckpt or LoRA adapter dir).
    """
    cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_t = _mica_suggest_stage2(trial, cfg_t)
    cfg_t.TRAIN.epochs = int(epochs_search)

    metrics: List[float] = []
    for split in range(num_splits):
        split_dir = exp_dir / f"trial_{trial.number:03d}" / f"split_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)

        cfg_split = OmegaConf.create(OmegaConf.to_container(cfg_t, resolve=True))
        cfg_split.MICA.stage2.stage1_ckpt = None
        cfg_split.MICA.stage2.stage1_lora_dir = None

        if (
            "stage1_lora_dir" in stage1_best_artifacts
            and split in stage1_best_artifacts["stage1_lora_dir"]
        ):
            cfg_split.MICA.stage2.stage1_lora_dir = stage1_best_artifacts[
                "stage1_lora_dir"
            ][split]
        elif (
            "stage1_ckpt" in stage1_best_artifacts
            and split in stage1_best_artifacts["stage1_ckpt"]
        ):
            cfg_split.MICA.stage2.stage1_ckpt = stage1_best_artifacts["stage1_ckpt"][
                split
            ]

        _ = _run_single_fit(
            trainer_cfg=cfg_split.MICA.stage2.trainer,
            cfg=cfg_split,
            device=device,
            exp_dir=split_dir,
            seed=int(cfg_split.TRAIN.seed) + split,
            split_number=split,
            csv_train_path=train_csvs[split],
            csv_val_path=val_csvs[split],
            csv_test_path=None,
            path_to_images=images_root,
            overlay_cfg_train=overlay_cfg_train,
            overlay_cfg_test=overlay_cfg_train,
        )

        m = _read_best_metric(split_dir)
        metrics.append(m)

        running = float(np.nanmean(metrics))
        trial.report(running, step=split)
        if split >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(metrics))
