import logging
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from hydra.utils import instantiate

from RetCLIP.source.utils.text_utils import encode_batch_fulltokenizer
from RetCLIP.source.utils.checkpoints import CheckpointPaths
from RetCLIP.source.utils.misc import _discover_best_paths


def mica_collate_fn(fulltok, max_length: int):
    def _collate(batch):
        imgs = torch.stack([b["x"] for b in batch], dim=0)
        captions = [b["caption"] for b in batch]
        tok = encode_batch_fulltokenizer(fulltok, captions, max_length=max_length)

        concept_labels = torch.stack([b["concept_labels"] for b in batch], dim=0)

        return {
            "imgs": imgs,
            "caption_ids": tok["caption_ids"],
            "attention_mask": tok["attention_mask"],
            "token_type_ids": tok["token_type_ids"],
            "concept_labels": concept_labels,
            "filename": [b["filename"] for b in batch],
            "y": torch.stack([b["y"] for b in batch], dim=0),
            "spurious_type": [b["spurious_type"] for b in batch],
            "spurious_applied": torch.tensor([b["spurious_applied"] for b in batch]),
        }

    return _collate


# -------------------------
# Helper: trainer selection
# -------------------------
def _build_trainer(
    cfg: DictConfig,
    device: torch.device,
    exp_dir: Path,
    split_number: int,
    csv_train_path: Path,
    csv_val_path: Path,
    csv_test_path: Optional[Path],
    path_to_images: Path,
    overlay_cfg_train: DictConfig,
    overlay_cfg_test: DictConfig,
):
    """
    Instantiate the trainer specified by cfg.TRAIN.trainer (Hydra target).

    Expected in config, e.g.:
      TRAIN:
        trainer:
          _target_: RetCLIP.source.utils.trainers.fundus_trainer.FundusClassifierTrainer
    """
    if not hasattr(cfg.TRAIN, "trainer"):
        raise KeyError(
            "Missing cfg.TRAIN.trainer. Please set TRAIN.trainer._target_ to the desired Trainer class."
        )

    trainer = instantiate(
        cfg.TRAIN.trainer,
        cfg=cfg,
        device=device,
        exp_dir=exp_dir,
        seed=int(cfg.TRAIN.seed),
        split_number=int(split_number),
        csv_train_path=csv_train_path,
        csv_val_path=csv_val_path,
        csv_test_path=csv_test_path,
        path_to_images=path_to_images,
        overlay_cfg_train=overlay_cfg_train,
        overlay_cfg_test=overlay_cfg_test,
    )
    return trainer


# -------------------------
# Helper: run train/val
# -------------------------
def _run_train_val(
    trainer,
) -> Any:
    """
    Calls into the BaseTrainer training loop.

    IMPORTANT:
    I don't have your BaseTrainer API here. This function assumes BaseTrainer exposes
    a method named `run()` that returns CheckpointPaths OR a dict containing it.

    If your BaseTrainer uses a different entrypoint, change it here only.
    """
    if hasattr(trainer, "run") and callable(trainer.run):
        return trainer.run()

    # Common alternatives
    if hasattr(trainer, "fit") and callable(trainer.fit):
        return trainer.fit()

    raise AttributeError(
        f"{trainer.__class__.__name__} has no run()/fit() method. "
        "Please adapt _run_train_val() to your BaseTrainer API."
    )


# -------------------------
# Helper: run test
# -------------------------
def _run_test(
    trainer,
    best_paths,
    test_loader,
    train_dataset_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Some of your trainers require passing criterion to test_from_ckpt, others don't.
    We handle both patterns.
    """
    # criterion is often needed for binary vs multiclass behavior
    criterion = None
    if hasattr(trainer, "build_criterion"):
        try:
            criterion = trainer.build_criterion(train_dataset_df)
        except Exception:
            criterion = None

    # Try signature variants
    if criterion is not None:
        try:
            return trainer.test_from_ckpt(best_paths, test_loader, criterion)
        except TypeError:
            pass

    return trainer.test_from_ckpt(best_paths, test_loader)


# -----------------------------------------------------------------------------
# Trainer factory / run primitives
# -----------------------------------------------------------------------------
def _instantiate_trainer(
    *,
    trainer_cfg: DictConfig,
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
):
    return instantiate(
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


def _run_single_fit(
    *,
    trainer_cfg: DictConfig,
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
) -> CheckpointPaths:
    trainer = _instantiate_trainer(
        trainer_cfg=trainer_cfg,
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


# -----------------------------------------------------------------------------
# MICA pipeline (stage1 -> stage2) runner
# -----------------------------------------------------------------------------
def _run_mica_pipeline_fit(
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
) -> CheckpointPaths:
    """
    Runs MICA Stage-1 then Stage-2 sequentially.

    Contract:
      - Stage-1 trainer saves a checkpoint (full or LoRA adapter) that Stage-2 can load.
      - Stage-2 trainer reads cfg.MICA.stage2.stage1_ckpt OR cfg.MICA.stage2.stage1_lora_dir.

    This function:
      1) calls Stage-1 trainer.fit()
      2) sets cfg.MICA.stage2.stage1_* for Stage-2 based on Stage-1 outputs
      3) calls Stage-2 trainer.fit()
    """
    stage1_trainer_cfg = cfg.MICA.stage1.trainer
    stage2_trainer_cfg = cfg.MICA.stage2.trainer

    stage1_dir = exp_dir / "stage1"
    stage2_dir = exp_dir / "stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 ----
    paths_s1 = _run_single_fit(
        trainer_cfg=stage1_trainer_cfg,
        cfg=cfg,
        device=device,
        exp_dir=stage1_dir,
        seed=seed,
        split_number=split_number,
        csv_train_path=csv_train_path,
        csv_val_path=csv_val_path,
        csv_test_path=None,  # stage1 usually does not test
        path_to_images=path_to_images,
        overlay_cfg_train=overlay_cfg_train,
        overlay_cfg_test=overlay_cfg_train,
    )

    # ---- Prepare cfg for Stage 2 ----
    cfg_s2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Prefer LoRA adapter if present; otherwise use ckpt
    stage1_ckpt = getattr(
        paths_s1, "best_stage2_ckpt", None
    )  # some stage1 trainers may store here
    stage1_lora_dir = getattr(paths_s1, "best_lora_dir", None)

    # You may also have a dedicated field in your CheckpointPaths for stage1;
    # if so, add it here.
    if stage1_lora_dir is not None:
        cfg_s2.MICA.stage2.stage1_lora_dir = str(stage1_lora_dir)
        cfg_s2.MICA.stage2.stage1_ckpt = None
    elif stage1_ckpt is not None:
        cfg_s2.MICA.stage2.stage1_ckpt = str(stage1_ckpt)
        cfg_s2.MICA.stage2.stage1_lora_dir = None
    else:
        # fallback: look for common stage1 artifacts in stage1_dir/checkpoints
        ckpt_dir = stage1_dir / "checkpoints"
        discovered = _discover_best_paths(ckpt_dir)
        if discovered.best_lora_dir is not None:
            cfg_s2.MICA.stage2.stage1_lora_dir = str(discovered.best_lora_dir)
            cfg_s2.MICA.stage2.stage1_ckpt = None
        elif discovered.best_stage2_ckpt is not None:
            cfg_s2.MICA.stage2.stage1_ckpt = str(discovered.best_stage2_ckpt)
            cfg_s2.MICA.stage2.stage1_lora_dir = None
        else:
            logging.warning(
                "[MICA pipeline] Could not infer Stage-1 artifact (ckpt/lora). "
                "Stage-2 will run with randomly-initialized or default encoder weights."
            )

    # ---- Stage 2 ----
    paths_s2 = _run_single_fit(
        trainer_cfg=stage2_trainer_cfg,
        cfg=cfg_s2,
        device=device,
        exp_dir=stage2_dir,
        seed=seed,
        split_number=split_number,
        csv_train_path=csv_train_path,
        csv_val_path=csv_val_path,
        csv_test_path=(
            csv_test_path if bool(getattr(cfg.TEST, "enabled", False)) else None
        ),
        path_to_images=path_to_images,
        overlay_cfg_train=overlay_cfg_train,
        overlay_cfg_test=overlay_cfg_test,
    )
    return paths_s2
