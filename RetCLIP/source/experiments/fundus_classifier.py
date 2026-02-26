import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
from optuna.trial import TrialState

from RetCLIP.source.experiments.context import (
    RunContext,
    build_overlay_cfg,
    exp_dir_for,
)
from RetCLIP.source.experiments.optuna_common import (
    create_trial_split_exp_dir,
    read_best_metric,
)
from RetCLIP.source.experiments.base import BaseExperiment, OptunaResult
from RetCLIP.source.utils.checkpoints import CheckpointPaths
from RetCLIP.source.utils.misc import split_paths
from RetCLIP.source.trainers.fundus_classifier_trainer import FundusClassifierTrainer


class FundusExperiment(BaseExperiment):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    @staticmethod
    def _to_list(x):
        if isinstance(x, (list, tuple, ListConfig)):
            return list(x)
        else:
            return [x]

    def _make_trainer(
        self,
        ctx: RunContext,
        exp_dir: Path,
        split: int,
        csv_train: Path,
        csv_val: Path,
        overlay_train: DictConfig,
        overlay_test: DictConfig,
        csv_test: Optional[Path] = None,
        cfg_override: Optional[DictConfig] = None,
    ):
        cfg_use = cfg_override if cfg_override is not None else ctx.cfg
        return FundusClassifierTrainer(
            cfg=cfg_use,
            device=ctx.device,
            exp_dir=exp_dir,
            seed=int(cfg_use.TRAIN.seed) + int(split),
            split_number=split,
            csv_train_path=csv_train,
            csv_val_path=csv_val,
            csv_test_path=csv_test,
            path_to_images=ctx.images_root,
            overlay_cfg_train=overlay_train,
            overlay_cfg_test=overlay_test,
        )

    def _apply_best_params(
        self, cfg_local: DictConfig, best_params: Dict[str, Any]
    ) -> DictConfig:
        """
        Apply best params into cfg_local.
        Supports both:
        - Optuna raw keys: lora_r, lora_dropout, lr_classifier, lr_lora
        - Canonical keys: r, dropout, lr_head, lr_lora
        """

        # --- LoRA r ---
        r = None
        if "lora_r" in best_params:
            r = int(best_params["lora_r"])
        elif "r" in best_params:
            r = int(best_params["r"])

        if r is not None:
            cfg_local.TUNE.lora.r = r
        # --- LoRA dropout ---
        dropout = None
        if "lora_dropout" in best_params:
            dropout = float(best_params["lora_dropout"])
        elif "dropout" in best_params:
            dropout = float(best_params["dropout"])

        if dropout is not None:
            cfg_local.TUNE.lora.dropout = dropout

        # --- LR head ---
        lr_head = None
        if "lr_classifier" in best_params:
            lr_head = float(best_params["lr_classifier"])
        elif "lr_head" in best_params:
            lr_head = float(best_params["lr_head"])

        if lr_head is not None:
            cfg_local.TUNE.lr_head = lr_head

        # --- LR lora ---
        if "lr_lora" in best_params:
            cfg_local.TUNE.lr_lora = float(best_params["lr_lora"])

        return cfg_local

    # ---- Optuna suggestion for fundus ----
    def _suggest_optuna_params(
        self, trial: optuna.trial.Trial, cfg_local: DictConfig
    ) -> DictConfig:
        opt_cfg = cfg_local.OPTUNA

        r = trial.suggest_categorical("lora_r", self._to_list(opt_cfg.lora_r))
        dropout = trial.suggest_categorical(
            "lora_dropout", self._to_list(opt_cfg.lora_dropout)
        )

        lr_classifier = trial.suggest_categorical(
            "lr_classifier", self._to_list(opt_cfg.lr_classifier)
        )
        lr_lora = trial.suggest_categorical("lr_lora", self._to_list(opt_cfg.lr_lora))

        params = {
            "r": r,
            "dropout": dropout,
            "lr_head": lr_classifier,
            "lr_lora": lr_lora,
        }

        return self._apply_best_params(cfg_local, params)

    def run_train(
        self,
        ctx: RunContext,
        best: Optional[OptunaResult] = None,
        full_train: Optional[bool] = True,
    ):
        """
        If best is provided, patch a cfg copy with best params (coming from Optuna) and run training.
        """
        cfg_local = OmegaConf.create(OmegaConf.to_container(ctx.cfg, resolve=True))

        if best is not None:
            # Priority: explicit best_params dict, else load from best_params_path
            params = None
            if best.best_params is not None:
                params = best.best_params
            elif best.best_params_path is not None:
                params = self._load_best_params(Path(best.best_params_path))

            if params is not None:
                cfg_local = self._apply_best_params(cfg_local, params)
                logging.info(f"[Fundus][train] applied best params: {params}")
                logging.info(
                    "After patch, TUNE.lora.dropout=%s", cfg_local.TUNE.lora.dropout
                )

        logging.info(
            "[Train] Loaded TUNE configuration:\n%s",
            OmegaConf.to_yaml(cfg_local.TUNE),
        )

        for pct in ctx.overlay_percentages:
            overlay_train = build_overlay_cfg(cfg_local, "train", pct)
            overlay_test = build_overlay_cfg(cfg_local, "test", pct)

            for split in range(ctx.num_splits):
                exp_dir = exp_dir_for(ctx, pct, split)
                exp_dir.mkdir(parents=True, exist_ok=True)

                csv_train, csv_val, csv_test = split_paths(ctx.split_root, split)

                trainer = self._make_trainer(
                    ctx,
                    exp_dir,
                    split,
                    csv_train,
                    csv_val,
                    overlay_train,
                    overlay_test,
                    csv_test,
                    cfg_override=cfg_local,
                )

                best_paths = trainer.fit()

                if ctx.cfg.EVAL.run and csv_test is not None:
                    test_loader = trainer.build_test_loader()
                    res = trainer.test_from_ckpt(best_paths, test_loader)
                    logging.info(
                        f"[Fundus][pct={pct}][split={split}] test={res.get('metrics', res)}"
                    )

    def run_test_only(self, ctx: RunContext):
        """
        Uses the trainer's own load_for_test/test_from_ckpt logic.
        """

        root_dir = Path(ctx.cfg.EVAL.test_only.root_dir)
        logging.info(f"[Fundus] TEST-ONLY root_dir={root_dir}")

        for pct in ctx.overlay_percentages:
            overlay_test = build_overlay_cfg(ctx.cfg, "test", pct)
            overlay_train_dummy = build_overlay_cfg(ctx.cfg, "train", pct)

            for split in range(ctx.num_splits):
                prev_exp_dir = exp_dir_for(ctx, pct, split, root=root_dir)
                exp_dir = (
                    prev_exp_dir / "test_only" / ctx.cfg.DATASET.overlay_cfg_test.mode
                )
                exp_dir.mkdir(parents=True, exist_ok=True)

                _, _, csv_test = split_paths(ctx.split_root, split)

                trainer = self._make_trainer(
                    ctx,
                    exp_dir=exp_dir,
                    split=split,
                    csv_train=ctx.split_root,
                    csv_val=ctx.split_root,
                    csv_test=csv_test,
                    overlay_train=overlay_train_dummy,
                    overlay_test=overlay_test,
                )

                # Reconstruct checkpoint paths
                manifest_path = prev_exp_dir / "checkpoints" / "manifest.json"
                best = CheckpointPaths()
                if manifest_path.is_file():
                    m = json.load(open(manifest_path))
                    if "classifier_head" in m:
                        best.best_classifier_head = prev_exp_dir / m["classifier_head"]
                    if "lora_dir" in m:
                        best.best_lora_dir = prev_exp_dir / m["lora_dir"]
                else:
                    # backward compatibility
                    best.best_classifier_head = (
                        prev_exp_dir / "checkpoints" / "best_classifier_head.ckpt"
                    )
                    best.best_lora_dir = (
                        prev_exp_dir / "checkpoints" / "best_lora_adapter"
                    )

                test_loader = trainer.build_test_loader()
                res = trainer.test_from_ckpt(best, test_loader)
                logging.info(
                    f"[Fundus TEST-ONLY][pct={pct}][split={split}] test={res.get('metrics', res)}"
                )

                if ctx.cfg.EVAL.get("explainability", False):
                    from RetCLIP.source.utils.explainability import run_explainability_pipeline
                    run_explainability_pipeline(
                        model=res["model"],
                        test_dataset=test_loader.dataset,
                        device=ctx.device,
                        save_dir=exp_dir / "explainability",
                        split_number=split,
                        save_visualizations=(split == 0),
                        save_mask_overlays=(split == 0 and pct == 0),
                    )

    def run_optuna(self, ctx: RunContext) -> OptunaResult:
        # In Optuna we search at pct=0 only
        pct = 0.0
        overlay_train = build_overlay_cfg(ctx.cfg, "train", pct)
        overlay_test_dummy = build_overlay_cfg(ctx.cfg, "test", pct)

        opt_dir = ctx.hydra_subdir / "optuna"
        opt_dir.mkdir(parents=True, exist_ok=True)

        storage_name = opt_dir / f"{ctx.cfg.OPTUNA.study_name}.db"
        storage_uri = f"sqlite:///{storage_name}"
        storage = optuna.storages.RDBStorage(url=str(storage_uri))

        train_csvs, val_csvs = [], []
        for split in range(ctx.num_splits):
            tr, va, _ = split_paths(ctx.split_root, split)
            train_csvs.append(tr)
            val_csvs.append(va)

        def _objective(trial: optuna.trial.Trial) -> float:
            cfg_local = OmegaConf.create(OmegaConf.to_container(ctx.cfg, resolve=True))
            cfg_local = self._suggest_optuna_params(trial, cfg_local)

            # Short search schedule
            epochs_search = int(
                getattr(cfg_local.OPTUNA, "epochs_search", cfg_local.TRAIN.epochs)
            )
            cfg_local.TRAIN.epochs = epochs_search

            metrics = []
            for split in range(ctx.num_splits):
                split_exp_dir = create_trial_split_exp_dir(opt_dir, trial, split)

                trainer = self._make_trainer(
                    ctx=ctx,
                    exp_dir=split_exp_dir,
                    split=split,
                    csv_train=train_csvs[split],
                    csv_val=val_csvs[split],
                    csv_test=None,
                    overlay_train=overlay_train,
                    overlay_test=overlay_test_dummy,
                    cfg_override=cfg_local,
                )

                trainer.fit()
                m = read_best_metric(split_exp_dir)
                metrics.append(m)

                running_mean = float(np.nanmean(metrics))
                trial.report(running_mean, step=split)
                if split >= 1 and trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.nanmean(metrics))

        study = optuna.create_study(
            direction="maximize",
            study_name=ctx.cfg.OPTUNA.study_name,
            sampler=instantiate(ctx.cfg.OPTUNA.sampler),
            pruner=instantiate(ctx.cfg.OPTUNA.pruner),
            storage=storage,
        )
        study.optimize(_objective, n_trials=int(ctx.cfg.OPTUNA.n_trials))

        pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logging.info(
            f"[Fundus Optuna] finished={len(study.trials)} pruned={len(pruned)} complete={len(completed)}"
        )
        logging.info(
            f"[Fundus Optuna] best value={study.best_trial.value} params={study.best_trial.params}"
        )

        best_params = dict(study.best_trial.params)
        best_params_path = self._save_best_params(opt_dir, best_params)

        best_patch = {
            "TUNE": {
                "mode": "lora",
                "lr_head": study.best_trial.params.get("lr_classifier"),
                "lr_lora": study.best_trial.params.get("lr_lora"),
                "lora": {
                    "r": study.best_trial.params.get("lora_r"),
                    "dropout": study.best_trial.params.get("lora_dropout"),
                },
            }
        }

        logging.info("[Optuna] Best TUNE patch:\n%s", OmegaConf.to_yaml(best_patch))

        return OptunaResult(
            best_params=best_params,
            best_params_path=best_params_path,
            extra={
                "best_value": float(study.best_trial.value),
                "study_name": str(ctx.cfg.OPTUNA.study_name),
                "storage_db": str(storage_name),
            },
        )
