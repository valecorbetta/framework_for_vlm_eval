import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
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

from RetCLIP.source.utils.misc import split_paths
from RetCLIP.source.utils.checkpoints import CheckpointPaths
from RetCLIP.source.experiments.base import BaseExperiment, OptunaResult
from RetCLIP.source.experiments.fundus_classifier import FundusExperiment
from RetCLIP.source.trainers.multitask_trainer import RetCLIPMultiTaskTrainer


class MultiTaskExperiment(FundusExperiment):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _make_trainer(
        self,
        ctx: RunContext,
        exp_dir: Path,
        split: int,
        csv_train: Path,
        csv_val: Path,
        overlay_train: DictConfig,
        overlay_test: Optional[DictConfig] = None,
        csv_test: Optional[Path] = None,
        cfg_override: Optional[DictConfig] = None,
    ):
        cfg_use = ctx.cfg if cfg_override is None else cfg_override
        return RetCLIPMultiTaskTrainer(
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

    def _apply_best_params(self, cfg_local, best_params):
        """
        Apply best params into cfg_local.
        """
        # --- concept_weight ---
        concept_weight = None
        if "concept_weight" in best_params:
            concept_weight = float(best_params["concept_weight"])
        else:
            logging.info(
                f"concept_weight not present in best_params after Optuna optim: {concept_weight=}"
            )
        if concept_weight is not None:
            cfg_local.MODEL.losses.concept_weight = concept_weight

        return cfg_local

    def _suggest_optuna_params(self, trial, cfg_local):
        cw = cfg_local.OPTUNA.concept_weight
        if bool(getattr(cw, "log", True)):
            concept_weight = trial.suggest_float(
                "concept_weight", float(cw.low), float(cw.high), log=True
            )
        else:
            concept_weight = trial.suggest_float(
                "concept_weight", float(cw.low), float(cw.high), log=False
            )
        params = {"concept_weight": concept_weight}
        return self._apply_best_params(cfg_local, params)

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
