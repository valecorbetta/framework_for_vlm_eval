import logging
from pathlib import Path
from typing import Optional, Tuple

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
from RetCLIP.source.experiments.base import BaseExperiment, OptunaResult
from RetCLIP.source.utils.checkpoints import CheckpointPaths
from RetCLIP.source.utils.misc import split_paths
from RetCLIP.source.trainers.mica_trainer_stage1 import MICAStage1Trainer
from RetCLIP.source.trainers.mica_trainer_stage2 import MICAStage2CBMTrainer
from RetCLIP.source.data.dataset_FGADR import (
    FGADRConceptDataset,
    mica_collate_fulltokenizer,
)


class MICAExperiment(BaseExperiment):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _make_stage1_trainer(
        self,
        ctx: RunContext,
        exp_dir: Path,
        split: int,
        csv_train: Path,
        csv_val: Path,
        overlay_train: DictConfig,
        cfg_local: Optional[DictConfig] = None,
    ) -> MICAStage1Trainer:
        cfg_use = ctx.cfg if cfg_local is None else cfg_local
        return MICAStage1Trainer(
            cfg=cfg_use,
            device=ctx.device,
            exp_dir=exp_dir,
            seed=int(cfg_use.TRAIN.seed) + int(split),
            split_number=split,
            csv_train_path=csv_train,
            csv_val_path=csv_val,
            path_to_images=ctx.images_root,
            overlay_cfg_train=overlay_train,
        )

    def _make_stage2_trainer(
        self,
        ctx: RunContext,
        exp_dir: Path,
        split: int,
        csv_train: Path,
        csv_val: Path,
        csv_test: Optional[Path],
        overlay_train: DictConfig,
        overlay_test: DictConfig,
        cfg_local: Optional[DictConfig] = None,
    ):
        cfg_use = ctx.cfg if cfg_local is None else cfg_local
        return MICAStage2CBMTrainer(
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

    @staticmethod
    def get_cav_file_path(path_to_folder: Path, pct: float, split: int) -> Path:
        return path_to_folder / str(pct) / f"mica_concepts_split{split}.pkl"

    @staticmethod
    def _inject_stage1_artifact_into_stage2_cfg(
        cfg_local: DictConfig,
        stage1_paths: CheckpointPaths,
    ) -> DictConfig:
        """
        Stage2 trainer build_model expects:
          cfg_local.MICA.stage2.stage1_ckpt OR cfg_local.MICA.stage2.stage1_lora_dir
        We populate these from the stage1 fit outputs.
        """
        if (
            stage1_paths.best_lora_dir is not None
            and stage1_paths.best_lora_dir.exists()
        ):
            cfg_local.MODEL.stage_2.stage_1_lora_dir = str(stage1_paths.best_lora_dir)
            cfg_local.MODEL.stage_2.stage_1_ckpt = None
        elif (
            stage1_paths.best_stage1_ckpt is not None
            and stage1_paths.best_stage1_ckpt.exists()
        ):
            cfg_local.MODEL.stage_2.stage_1_ckpt = str(stage1_paths.best_stage1_ckpt)
            cfg_local.MODEL.stage_2.stage_1_lora_dir = None
        else:
            raise FileNotFoundError(
                f"Stage1 did not produce usable artifacts. "
                f"best_lora_dir={stage1_paths.best_lora_dir} best_stage1_ckpt={stage1_paths.best_stage1_ckpt}"
            )
        return cfg_local

    # -------------------- TRAIN -------------------- #
    def run_train(
        self,
        ctx: RunContext,
        best: Optional[OptunaResult] = None,
        full_train: Optional[bool] = True,
    ):
        # Get pre-trained stage 1 checkpoints from optuna (if any)
        # Dict[pct][split] -> CheckpointPaths
        logging.info("=" * 60)
        logging.info("[MICA TRAIN] Starting full training pipeline")
        if best is not None:
            logging.info("[MICA TRAIN] Using OptunaResult:")
            logging.info(f"    best_params: {best.best_params}")
            if best.extra and "stage1_checkpoints" in best.extra:
                logging.info(
                    f"    Pre-trained stage1 checkpoints available for pcts: {list(best.extra['stage1_checkpoints'].keys())}"
                )
        else:
            logging.info("[MICA TRAIN] No OptunaResult provided, using default config")
        pretrained_stage1 = {}
        if best is not None and best.extra and "stage1_checkpoints" in best.extra:
            pretrained_stage1 = best.extra["stage1_checkpoints"]

        for pct in ctx.overlay_percentages:
            overlay_train = build_overlay_cfg(ctx.cfg, "train", pct)
            overlay_test = build_overlay_cfg(ctx.cfg, "test", pct)

            for split in range(ctx.num_splits):
                logging.info("-" * 40)
                logging.info(f"[MICA TRAIN] pct={pct}, split={split}")
                exp_dir = exp_dir_for(ctx, pct, split)
                exp_dir.mkdir(parents=True, exist_ok=True)

                csv_train, csv_val, csv_test = split_paths(ctx.split_root, split)
                # Stage 1 params for this run (may be empty)
                stage1_params = (
                    best.best_params.get("stage1", {})
                    if best is not None and best.best_params
                    else {}
                )

                stage1_dir = exp_dir / "mica_stage1"
                stage1_dir.mkdir(parents=True, exist_ok=True)

                # Stage 1 - skip if already trained during optuna
                if pct in pretrained_stage1 and split in pretrained_stage1[pct]:
                    logging.info(
                        "[MICA TRAIN] SKIPPING stage 1 training - using pre-trained checkpoint"
                    )
                    logging.info(
                        f"[MICA][pct={pct}][split={split}] Using pre-trained stage 1 from optuna"
                    )
                    best_stage1 = pretrained_stage1[pct][split]
                    logging.info(f"    best_lora_dir={best_stage1.best_lora_dir}")
                    logging.info(f"    best_stage1_ckpt={best_stage1.best_stage1_ckpt}")
                else:
                    logging.info("[MICA TRAIN] Training stage 1")
                    if stage1_params:
                        logging.info(f"    Applying stage1 params: {stage1_params}")

                    # Apply best params if available
                    cfg_local = OmegaConf.create(
                        OmegaConf.to_container(ctx.cfg, resolve=True)
                    )
                    for k in [
                        "temp1",
                        "temp2",
                        "temp3",
                        "local_loss_weight",
                        "global_loss_weight",
                        "concept_loss_weight",
                    ]:
                        if k in stage1_params:
                            cfg_local.MODEL.stage_1.losses[k] = float(stage1_params[k])

                    t1: MICAStage1Trainer = self._make_stage1_trainer(
                        ctx,
                        exp_dir=stage1_dir,
                        split=split,
                        csv_train=csv_train,
                        csv_val=csv_val,
                        overlay_train=overlay_train,
                        cfg_local=cfg_local,
                    )
                    path_to_cav_file = self.get_cav_file_path(
                        ctx.cfg.PATHS.path_to_cav_folder, pct, split
                    )

                    best_stage1 = t1.fit(path_to_cav_file)

                    logging.info("[MICA TRAIN] Stage 1 complete")
                    logging.info(f"    best_lora_dir={best_stage1.best_lora_dir}")
                    logging.info(f"    best_stage1_ckpt={best_stage1.best_stage1_ckpt}")

                # Stage 2 (uses Stage1 artifact)
                logging.info(f"[MICA TRAIN] Training stage 2")
                stage2_dir = exp_dir / "mica_stage2"
                stage2_dir.mkdir(parents=True, exist_ok=True)

                # Create fresh config and apply stage2 params
                cfg_local_s2 = OmegaConf.create(
                    OmegaConf.to_container(ctx.cfg, resolve=True)
                )

                # Apply best stage 2 params if available
                stage2_params = (
                    best.best_params.get("stage2", {})
                    if best is not None and best.best_params
                    else {}
                )
                if stage2_params:
                    logging.info(f"    Applying stage2 params: {stage2_params}")
                    if "concept_weight" in stage2_params:
                        cfg_local_s2.MODEL.stage_2.concept_weight = float(
                            stage2_params["concept_weight"]
                        )

                # Inject stage1 artifact paths
                cfg_local_s2 = self._inject_stage1_artifact_into_stage2_cfg(
                    cfg_local_s2, best_stage1
                )
                logging.info(f"    Injecting stage1 artifact:")
                logging.info(
                    f"      stage_1_lora_dir={cfg_local_s2.MODEL.stage_2.stage_1_lora_dir}"
                )
                logging.info(
                    f"      stage_1_ckpt={cfg_local_s2.MODEL.stage_2.stage_1_ckpt}"
                )

                t2 = self._make_stage2_trainer(
                    ctx,
                    exp_dir=stage2_dir,
                    split=split,
                    csv_train=csv_train,
                    csv_val=csv_val,
                    csv_test=csv_test,
                    overlay_train=overlay_train,
                    overlay_test=overlay_test,
                    cfg_local=cfg_local_s2,
                )
                best_stage2 = t2.fit()

                if ctx.cfg.EVAL.run and csv_test is not None:
                    tokenizer = instantiate(self.cfg.MODEL.tokenizer)
                    max_len = int(self.cfg.DATASET.text.word_num)
                    collate = lambda b: mica_collate_fulltokenizer(
                        b, tokenizer, max_len
                    )
                    test_loader = t2.build_test_loader(collate_fn=collate)
                    res = t2.test_from_ckpt(
                        best_stage2,
                        test_loader,
                        t2.build_classification_loss(getattr(test_loader, "dataset")),
                    )
                    logging.info(
                        f"[MICA][pct={pct}][split={split}] stage2 test={res.get('metrics', res)}"
                    )
                    logging.info(f"[MICA TRAIN] Stage 2 complete")

        logging.info("=" * 60)
        logging.info("[MICA TRAIN] Full training pipeline complete")
        logging.info("=" * 60)

    # -------------------- TEST ONLY -------------------- #
    def run_test_only(self, ctx: RunContext):
        """
        For MICA, we test stage2.
        It requires:
          - stage2 classifier checkpoint
          - stage1 encoder artifact (ckpt or lora dir) inside the old run dir
        """

        root_dir = Path(ctx.cfg.EVAL.test_only.root_dir)
        logging.info(f"[MICA] TEST-ONLY root_dir={root_dir}")

        for pct in ctx.overlay_percentages:
            overlay_test = build_overlay_cfg(ctx.cfg, "test", pct)
            overlay_train_dummy = build_overlay_cfg(ctx.cfg, "train", pct)

            for split in range(ctx.num_splits):
                prev_exp_dir = exp_dir_for(ctx, pct, split, root=root_dir)

                stage1_prev = prev_exp_dir / "mica_stage1"
                stage2_prev = prev_exp_dir / "mica_stage2"

                stage2_ckpt = stage2_prev / "checkpoints" / "best_stage2_heads.pt"

                if not stage2_ckpt.is_file():
                    logging.warning(
                        f"[MICA TEST-ONLY] missing stage2 ckpt: {stage2_ckpt}"
                    )
                    continue

                # Try to detect stage1 artifact
                stage1_lora_dir = stage1_prev / "checkpoints" / "best_lora_adapter"
                stage1_full_ckpt = stage1_prev / "checkpoints" / "best_stage1_mica.pt"

                cfg_local = OmegaConf.create(
                    OmegaConf.to_container(ctx.cfg, resolve=True)
                )
                if stage1_lora_dir.is_dir():
                    cfg_local.MODEL.stage_2.stage_1_lora_dir = str(stage1_lora_dir)
                    cfg_local.MODEL.stage_2.stage_1_ckpt = None
                elif stage1_full_ckpt.is_file():
                    cfg_local.MODEL.stage_2.stage_1_ckpt = str(stage1_full_ckpt)
                    cfg_local.MODEL.stage_2.stage_1_lora_dir = None
                else:
                    logging.warning(
                        f"[MICA TEST-ONLY] missing stage1 artifact (lora dir or ckpt) "
                        f"under {stage1_prev}/checkpoints"
                    )
                    continue

                _, _, csv_test = split_paths(ctx.split_root, split)

                exp_dir = (
                    stage2_prev / "test_only" / ctx.cfg.DATASET.overlay_cfg_test.mode
                )
                exp_dir.mkdir(parents=True, exist_ok=True)

                t2 = self._make_stage2_trainer(
                    ctx,
                    exp_dir=exp_dir,
                    split=split,
                    csv_train=ctx.split_root,
                    csv_val=ctx.split_root,
                    csv_test=csv_test,
                    overlay_train=overlay_train_dummy,
                    overlay_test=overlay_test,
                    cfg_local=cfg_local,
                )

                best = CheckpointPaths(best_classifier_head=stage2_ckpt)

                tokenizer = instantiate(self.cfg.MODEL.tokenizer)
                max_len = int(self.cfg.DATASET.text.word_num)
                collate = lambda b: mica_collate_fulltokenizer(b, tokenizer, max_len)
                test_loader = t2.build_test_loader(collate_fn=collate)
                criterion = t2.build_classification_loss(
                    getattr(test_loader, "dataset", None)
                )
                res = t2.test_from_ckpt(best, test_loader, criterion)
                logging.info(
                    f"[MICA TEST-ONLY][pct={pct}][split={split}] test={res.get('metrics', res)}"
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

    # -------------------- OPTUNA -------------------- #
    def run_optuna(self, ctx: RunContext) -> OptunaResult:
        """
        Flow:
        1. Optimize stage 1 (pct=0.0)
        2. Full train stage 1 (pct=0.0 only)
        3. Optimize stage 2 (pct=0.0, using fully trained stage 1)
        4. Return best params for both stages
        """
        logging.info("=" * 60)
        logging.info("[MICA OPTUNA] Starting Optuna optimization pipeline")
        logging.info(f"[MICA OPTUNA] overlay_percentages: {ctx.overlay_percentages}")
        logging.info(f"[MICA OPTUNA] num_splits: {ctx.num_splits}")
        logging.info("=" * 60)
        # Step 1: Optimize stage 1
        logging.info(
            "[MICA OPTUNA] Step 1: Optimizing stage 1 hyperparameters (pct=0.0)"
        )
        best_stage1_params, _ = self._run_optuna_stage1(ctx)
        logging.info(f"[MICA] Best stage 1 params: {best_stage1_params}")
        self._save_best_params(
            ctx.hydra_subdir / "optuna_mica_stage1", best_stage1_params
        )

        # Step 2: Full train stage 1 for pct=0.0
        logging.info(
            "[MICA OPTUNA] Step 2: Full training stage 1 (pct=0.0, all splits)"
        )
        stage1_checkpoints_pct0 = self._train_stage1_with_params(
            ctx, best_stage1_params
        )
        logging.info(f"[MICA OPTUNA] Stage 1 checkpoints created for pct=0.0:")
        for split, ckpt_paths in stage1_checkpoints_pct0.items():
            logging.info(
                f"    split={split}: lora_dir={ckpt_paths.best_lora_dir}, ckpt={ckpt_paths.best_stage1_ckpt}"
            )

        # Step 3: Optimize stage 2 using fully trained stage 1
        logging.info(
            "[MICA OPTUNA] Step 3: Optimizing stage 2 hyperparameters (pct=0.0)"
        )
        best_stage2_params = self._run_optuna_stage2(
            ctx, stage1_paths=stage1_checkpoints_pct0
        )
        logging.info(f"[MICA] Best stage 2 params: {best_stage2_params}")
        self._save_best_params(
            ctx.hydra_subdir / "optuna_mica_stage2", best_stage1_params
        )

        logging.info("=" * 60)
        logging.info("[MICA OPTUNA] Optuna pipeline complete")
        logging.info("=" * 60)
        # Step 4: Return combined result
        return OptunaResult(
            best_params={
                "stage1": best_stage1_params,
                "stage2": best_stage2_params,
            },
            extra={
                "stage1_checkpoints": {
                    0.0: stage1_checkpoints_pct0
                },  # pct -> split -> CheckpointPaths
            },
        )

    def _suggest_stage_1_params(
        self, trial: optuna.trial.Trial, cfg_local: DictConfig
    ) -> DictConfig:
        opt_cfg = cfg_local.OPTUNA
        # temps + weights
        for k in ["temp1", "temp2", "temp3"]:
            r = getattr(opt_cfg, k)
            val = trial.suggest_float(
                k, float(r.low), float(r.high), log=bool(getattr(r, "log", True))
            )
            cfg_local.MODEL.stage_1.losses[k] = float(val)

        for k in ["local_loss_weight", "global_loss_weight", "concept_loss_weight"]:
            r = getattr(opt_cfg, k)
            val = trial.suggest_float(
                k, float(r.low), float(r.high), log=bool(getattr(r, "log", True))
            )
            cfg_local.MODEL.stage_1.losses[k] = float(val)

    def _suggest_stage_2_params(
        self, trial: optuna.trial.Trial, opt_cfg: DictConfig, cfg_local_base: DictConfig
    ) -> DictConfig:
        cw = opt_cfg.concept_weight
        if bool(getattr(cw, "log", True)):
            concept_weight = trial.suggest_float(
                "concept_weight", float(cw.low), float(cw.high), log=True
            )
        else:
            concept_weight = trial.suggest_float(
                "concept_weight", float(cw.low), float(cw.high), log=False
            )

        cfg_local_base.MODEL.stage_2.concept_weight = float(concept_weight)

    def _run_optuna_stage1(self, ctx: RunContext) -> Path:
        opt_cfg = OmegaConf.create(OmegaConf.to_container(ctx.cfg.OPTUNA, resolve=True))
        mode = opt_cfg.mode
        if mode is None or mode not in ["both", "stage1"]:
            raise ValueError(
                "Optuna for stage 1 is disabled but requested stage1 optuna."
            )

        pct = 0.0
        overlay_train = build_overlay_cfg(ctx.cfg, "train", pct)

        opt_dir = ctx.hydra_subdir / "optuna_mica_stage1"
        opt_dir.mkdir(parents=True, exist_ok=True)

        storage_name = opt_dir / f"{opt_cfg.study_name}.db"
        storage_uri = f"sqlite:///{storage_name}"
        storage = optuna.storages.RDBStorage(url=str(storage_uri))

        train_csvs, val_csvs = [], []
        for split in range(ctx.num_splits):
            tr, va, _ = split_paths(ctx.split_root, split)
            train_csvs.append(tr)
            val_csvs.append(va)

        def _objective(trial: optuna.trial.Trial) -> float:
            logging.info("-" * 40)
            logging.info(f"[OPTUNA S1] Trial {trial.number} starting")
            cfg_local = OmegaConf.create(OmegaConf.to_container(ctx.cfg, resolve=True))

            # temps + weights
            self._suggest_stage_1_params(trial, cfg_local)

            epochs_search = int(
                getattr(opt_cfg, "epochs_search", cfg_local.TRAIN.epochs)
            )
            cfg_local.TRAIN.epochs = epochs_search
            # Log suggested params
            logging.info(f"[OPTUNA S1] Trial {trial.number} params:")
            logging.info(f"    temp1={cfg_local.MODEL.stage_1.losses.temp1}")
            logging.info(f"    temp2={cfg_local.MODEL.stage_1.losses.temp2}")
            logging.info(f"    temp3={cfg_local.MODEL.stage_1.losses.temp3}")
            logging.info(
                f"    local_loss_weight={cfg_local.MODEL.stage_1.losses.local_loss_weight}"
            )
            logging.info(
                f"    global_loss_weight={cfg_local.MODEL.stage_1.losses.global_loss_weight}"
            )
            logging.info(
                f"    concept_loss_weight={cfg_local.MODEL.stage_1.losses.concept_loss_weight}"
            )
            logging.info(f"    epochs_search={cfg_local.TRAIN.epochs}")

            metrics = []
            for split in range(ctx.num_splits):
                logging.info(
                    f"[OPTUNA S1] Trial {trial.number}, split={split} starting"
                )
                split_exp_dir = create_trial_split_exp_dir(opt_dir, trial, split)

                t1: MICAStage1Trainer = self._make_stage1_trainer(
                    ctx,
                    exp_dir=split_exp_dir,
                    split=split,
                    csv_train=train_csvs[split],
                    csv_val=val_csvs[split],
                    overlay_train=overlay_train,
                    cfg_local=cfg_local,
                )
                path_to_cav_file = self.get_cav_file_path(
                    ctx.cfg.PATHS.path_to_cav_folder, pct, split
                )

                t1.fit(path_to_cav_file)

                m = read_best_metric(split_exp_dir)
                logging.info(
                    f"[OPTUNA S1] Trial {trial.number}, split={split} metric={m}"
                )
                metrics.append(m)

                running_mean = float(np.nanmean(metrics))
                trial.report(running_mean, step=split)
                if split >= 1 and trial.should_prune():
                    raise optuna.TrialPruned()
            logging.info(
                f"[OPTUNA S1] Trial {trial.number} complete, mean metric={float(np.nanmean(metrics))}"
            )
            return float(np.nanmean(metrics))

        study = optuna.create_study(
            direction="minimize",
            study_name=str(opt_cfg.study_name),
            sampler=instantiate(opt_cfg.sampler),
            pruner=instantiate(opt_cfg.pruner),
            storage=storage,
        )
        study.optimize(_objective, n_trials=int(opt_cfg.n_trials_stage_1))

        best_params = study.best_trial.params
        logging.info(
            f"[MICA Stage1 Optuna] best value={study.best_trial.value} params={best_params}"
        )

        # Return the best trial directory so stage2 can reuse those trained stage1 artifacts if desired.
        best_trial_dir = opt_dir / f"trial_{study.best_trial.number:03d}"
        return best_params, best_trial_dir

    def _train_stage1_with_params(
        self, ctx: RunContext, best_params: dict[str, float]
    ) -> dict[int, CheckpointPaths]:

        pct = 0.0
        overlay_train = build_overlay_cfg(ctx.cfg, "train", pct)
        """Train stage 1 with best params for all splits. Returns split -> CheckpointPaths."""
        logging.info("=" * 50)
        logging.info(f"[TRAIN S1] Full training stage 1 for pct={pct}")

        # Apply best params
        cfg_local = OmegaConf.create(OmegaConf.to_container(ctx.cfg, resolve=True))
        for k in [
            "temp1",
            "temp2",
            "temp3",
            "local_loss_weight",
            "global_loss_weight",
            "concept_loss_weight",
        ]:
            if k in best_params:
                cfg_local.MODEL.stage_1.losses[k] = float(best_params[k])

        # Log what was actually set
        logging.info(f"[TRAIN S1] Config after applying params:")
        logging.info(f"    temp1={cfg_local.MODEL.stage_1.losses.temp1}")
        logging.info(f"    temp2={cfg_local.MODEL.stage_1.losses.temp2}")
        logging.info(f"    temp3={cfg_local.MODEL.stage_1.losses.temp3}")
        logging.info(
            f"    local_loss_weight={cfg_local.MODEL.stage_1.losses.local_loss_weight}"
        )
        logging.info(
            f"    global_loss_weight={cfg_local.MODEL.stage_1.losses.global_loss_weight}"
        )
        logging.info(
            f"    concept_loss_weight={cfg_local.MODEL.stage_1.losses.concept_loss_weight}"
        )
        logging.info(f"    epochs={cfg_local.TRAIN.epochs}")

        results = {}
        full_train_dir = ctx.hydra_subdir / "full_stage1_training"

        for split in range(ctx.num_splits):
            csv_train, csv_val, _ = split_paths(ctx.split_root, split)
            split_dir = full_train_dir / f"split_{split}"
            split_dir.mkdir(parents=True, exist_ok=True)

            t1 = self._make_stage1_trainer(
                ctx,
                exp_dir=split_dir,
                split=split,
                csv_train=csv_train,
                csv_val=csv_val,
                overlay_train=overlay_train,
                cfg_local=cfg_local,
            )
            path_to_cav_file = self.get_cav_file_path(
                ctx.cfg.PATHS.path_to_cav_folder, pct, split
            )
            logging.info(f"[TRAIN S1] pct={pct}, split={split} starting")
            logging.info(f"    exp_dir={split_dir}")
            logging.info(f"    csv_train={csv_train}")
            logging.info(f"    csv_val={csv_val}")
            logging.info(f"    cav_file={path_to_cav_file}")
            results[split] = t1.fit(path_to_cav_file)
            logging.info(f"[TRAIN S1] pct={pct}, split={split} complete")
            logging.info(f"    best_lora_dir={results[split].best_lora_dir}")
            logging.info(f"    best_stage1_ckpt={results[split].best_stage1_ckpt}")

        logging.info(f"[TRAIN S1] Full training stage 1 for pct={pct} complete")
        logging.info("=" * 50)
        return results

    def _run_optuna_stage2(
        self, ctx: RunContext, stage1_paths: dict[int, CheckpointPaths]
    ) -> dict[str, float]:
        logging.info("[OPTUNA S2] Starting stage 2 optimization")
        logging.info("[OPTUNA S2] Using stage 1 checkpoints:")
        for split, ckpt_paths in stage1_paths.items():
            logging.info(
                f"    split={split}: lora_dir={ckpt_paths.best_lora_dir}, ckpt={ckpt_paths.best_stage1_ckpt}"
            )

        opt_cfg = OmegaConf.create(OmegaConf.to_container(ctx.cfg.OPTUNA, resolve=True))
        mode = opt_cfg.mode
        if mode is None or mode not in ["both", "stage2"]:
            raise ValueError(
                "Optuna for stage 2 is disabled but requested stage 2 optuna."
            )

        pct = 0.0
        overlay_train = build_overlay_cfg(ctx.cfg, "train", pct)
        overlay_test_dummy = build_overlay_cfg(ctx.cfg, "test", pct)

        opt_dir = ctx.hydra_subdir / "optuna_mica_stage2"
        opt_dir.mkdir(parents=True, exist_ok=True)

        storage_name = opt_dir / f"{opt_cfg.study_name}.db"
        storage_uri = f"sqlite:///{storage_name}"
        storage = optuna.storages.RDBStorage(url=str(storage_uri))

        train_csvs, val_csvs = [], []
        for split in range(ctx.num_splits):
            tr, va, _ = split_paths(ctx.split_root, split)
            train_csvs.append(tr)
            val_csvs.append(va)

        def _objective(trial: optuna.trial.Trial) -> float:
            logging.info("-" * 40)
            logging.info(f"[OPTUNA S2] Trial {trial.number} starting")

            cfg_local_base = OmegaConf.create(
                OmegaConf.to_container(ctx.cfg, resolve=True)
            )

            self._suggest_stage_2_params(trial, opt_cfg, cfg_local_base)

            epochs_search = int(
                getattr(opt_cfg, "epochs_search", cfg_local_base.TRAIN.epochs)
            )
            cfg_local_base.TRAIN.epochs = epochs_search

            # Log suggested params
            logging.info(f"[OPTUNA S2] Trial {trial.number} params:")
            logging.info(
                f"    concept_weight={cfg_local_base.MODEL.stage_2.concept_weight}"
            )
            logging.info(f"    epochs_search={cfg_local_base.TRAIN.epochs}")

            metrics = []
            for split in range(ctx.num_splits):
                logging.info(
                    f"[OPTUNA S2] Trial {trial.number}, split={split} starting"
                )
                split_exp_dir = create_trial_split_exp_dir(opt_dir, trial, split)

                # Inject stage1 artifact path for this split
                cfg_local = OmegaConf.create(
                    OmegaConf.to_container(cfg_local_base, resolve=True)
                )

                # Use stage1_paths directly instead of _resolve_stage1_artifacts_for_split
                ckpt_paths = stage1_paths[split]
                logging.info(f"[OPTUNA S2] Trial {trial.number}, split={split}")
                logging.info(
                    f"    Loading stage 1 from: lora_dir={ckpt_paths.best_lora_dir}, ckpt={ckpt_paths.best_stage1_ckpt}"
                )

                cfg_local = self._inject_stage1_artifact_into_stage2_cfg(
                    cfg_local, ckpt_paths
                )

                logging.info(f"    Config after injection:")
                logging.info(
                    f"      stage_1_lora_dir={cfg_local.MODEL.stage_2.stage_1_lora_dir}"
                )
                logging.info(
                    f"      stage_1_ckpt={cfg_local.MODEL.stage_2.stage_1_ckpt}"
                )

                t2 = self._make_stage2_trainer(
                    ctx,
                    exp_dir=split_exp_dir,
                    split=split,
                    csv_train=train_csvs[split],
                    csv_val=val_csvs[split],
                    csv_test=None,
                    overlay_train=overlay_train,
                    overlay_test=overlay_test_dummy,
                    cfg_local=cfg_local,
                )
                t2.fit()

                m = read_best_metric(split_exp_dir)
                logging.info(
                    f"[OPTUNA S2] Trial {trial.number}, split={split} metric={m}"
                )
                metrics.append(m)

                running_mean = float(np.nanmean(metrics))
                trial.report(running_mean, step=split)
                if split >= 1 and trial.should_prune():
                    raise optuna.TrialPruned()

            final_metric = float(np.nanmean(metrics))
            logging.info(
                f"[OPTUNA S2] Trial {trial.number} complete, mean metric={final_metric}"
            )
            return final_metric

        study = optuna.create_study(
            direction="maximize",
            study_name=str(opt_cfg.study_name),
            sampler=instantiate(opt_cfg.sampler),
            pruner=instantiate(opt_cfg.pruner),
            storage=storage,
        )
        study.optimize(_objective, n_trials=int(opt_cfg.n_trials_stage_2))

        best_params = study.best_trial.params
        logging.info(
            f"[MICA Stage2 Optuna] best value={study.best_trial.value} params={best_params}"
        )

        return best_params
