from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from tqdm import tqdm

from source.utils.misc import set_seed
from source.trainers.base_trainer import BaseTrainer
from source.utils.checkpoints import CheckpointPaths
from source.models.mica_stage_1 import MICAStage1
from source.utils.data import mica_collate_hf_tokenizer


class MICAStage1Trainer(BaseTrainer):
    """
    Stage-1: train vision/text encoders + concept alignment losses.

    Assumptions:
      - model forward returns the tuple:
        (img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, predict_concepts, concept_labels)
      - predict_concepts are LOGITS with shape [B, Nc]
      - concept_labels are MULTI-HOT floats with shape [B, Nc]
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        exp_dir: Path,
        seed: int,
        split_number: int,
        csv_train_path: Path,
        csv_val_path: Path,
        path_to_images: Path,
        overlay_cfg_train: DictConfig,
        csv_test_path: Optional[Path] = None,
        overlay_cfg_test: Optional[DictConfig] = None,
    ):
        super().__init__(
            cfg,
            device,
            exp_dir,
            seed,
            split_number,
            csv_train_path,
            csv_val_path,
            csv_test_path,
            path_to_images,
            overlay_cfg_train,
            overlay_cfg_test,
        )

        self.sel_name = "loss"  # this is the validation loss
        self.worst = float("inf")

    def is_better(self, cur: float, best: float) -> bool:
        return cur <= best

    def build_model(self, path_to_cav_file: Path) -> MICAStage1:
        logging.info("=" * 50)
        logging.info("[MICAStage1Trainer] Building model")
        logging.info(f"[MICAStage1Trainer] CAV file: {path_to_cav_file}")

        # Log the config values being used
        logging.info("[MICAStage1Trainer] Config values:")
        logging.info(
            f"    MODEL.stage_1.losses: {OmegaConf.to_container(self.cfg.MODEL.stage_1.losses, resolve=True)}"
        )
        logging.info(
            f"    MODEL.text_encoder.embedding_dim: {self.cfg.MODEL.text_encoder.embedding_dim}"
        )
        logging.info(f"    DATASET.text.word_num: {self.cfg.DATASET.text.word_num}")
        cfg_for_model = OmegaConf.create(
            {
                "MODEL": {
                    "stage_1": {
                        "losses": OmegaConf.to_container(
                            self.cfg.MODEL.stage_1.losses, resolve=True
                        )
                    },
                    "text_encoder": {
                        "embedding_dim": int(self.cfg.MODEL.text_encoder.embedding_dim)
                    },
                },
                "DATASET": {"text": {"word_num": int(self.cfg.DATASET.text.word_num)}},
            }
        )
        logging.info(
            f"[MICAStage1Trainer] cfg_for_model created: {OmegaConf.to_yaml(cfg_for_model)}"
        )

        model = instantiate(
            self.cfg.MODEL.stage_1._class,
            cfg=cfg_for_model,
            path_to_cav_file=path_to_cav_file,
            device=self.device,
        )
        logging.info("[MICAStage1Trainer] Model built successfully")
        logging.info("=" * 50)
        return model

    def build_optimizer_and_scheduler(self, model: nn.Module, train_loader: DataLoader):
        lora_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora" in name:
                lora_params.append(p)
        if len(lora_params) == 0:
            raise RuntimeError(
                "Stage1 optimizer got 0 LoRA params. "
                "LoRA not attached, misnamed, or requires_grad=False."
            )
        optimizer = instantiate(
            self.cfg.OPTIMIZER, params=lora_params, lr=self.cfg.TUNE.lr_lora
        )
        scheduler = None
        if self.cfg.TRAIN.enable_scheduler:
            scheduler = instantiate(
                self.cfg.SCHEDULER,
                optimizer,
                total_epochs=self.cfg.TRAIN.epochs,
                warmup_steps=len(train_loader),
                total_steps=len(train_loader) * self.cfg.TRAIN.epochs,
            )
        logging.info("Trainable params sample:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logging.info(f"  {n}  {p.numel()}")

        return optimizer, scheduler

    def _mica_loss(
        self,
        model: MICAStage1,
        outputs: Tuple[torch.tensor],
    ) -> torch.Tensor:
        (
            img_emb_l,
            img_emb_g,
            text_emb_l,
            text_emb_g,
            sents,
            predict_concepts,
            concept_labels,
        ) = outputs

        # local + global loss from model methods
        l0, l1 = model._calc_local_loss(img_emb_l, text_emb_l, sents)
        g0, g1 = model._calc_global_loss(img_emb_g, text_emb_g)

        local_term = l0 + l1
        global_term = g0 + g1

        # concept labels: float multi-hot [B, Nc]
        # compute per-sample concept BCE then average per-sample, then average over batch
        # reduction="none": [B, Nc]
        bce_elem = model._calc_concept_loss(predict_concepts, concept_labels.float())
        per_sample_concept = bce_elem.sum(dim=1)
        concept_term = per_sample_concept.mean()

        wL = float(model.local_loss_weight)
        wG = float(model.global_loss_weight)
        wC = float(model.concept_loss_weight)

        total = (local_term * wL) + (global_term * wG) + (concept_term * wC)

        parts = {
            "local_loss": local_term.detach(),
            "global_loss": global_term.detach(),
            "concept_loss": concept_term.detach(),
            "loss": total.detach(),
        }
        return total, parts

    def train_one_epoch(
        self, model, train_loader, optimizer, scheduler, scaler
    ) -> Dict[str, float]:
        model.train()
        totals = {"loss": [], "local_loss": [], "global_loss": [], "concept_loss": []}

        for batch in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                outputs = model(batch)
                loss, parts = self._mica_loss(model, outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            for k in totals:
                totals[k].append(float(parts[k].item()))

        return {k: float(np.mean(v)) for k, v in totals.items()}

    def validate(self, model, val_loader) -> Dict[str, float]:
        model.eval()
        totals = {"loss": [], "local_loss": [], "global_loss": [], "concept_loss": []}
        with torch.no_grad():
            for batch in tqdm(val_loader):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                outputs = model(batch)
                _, parts = self._mica_loss(model, outputs)
                for k in totals:
                    totals[k].append(float(parts[k].item()))

        return {k: float(np.mean(v)) for k, v in totals.items()}

    def save_best(self, model: nn.Module) -> CheckpointPaths:
        logging.info("-" * 40)
        logging.info("[MICAStage1Trainer] Saving best model")
        # For stage1 you usually want to save: vision encoder (+ maybe text) weights
        paths = CheckpointPaths()
        if self.cfg.MODEL.vision_encoder.lora:
            paths.best_lora_dir = (
                self.ckpt_dir / "best_lora_adapter"
            )  # TODO: if we fine-tune the text encoder as well we should probably rename this
            paths.best_lora_dir.mkdir(exist_ok=True, parents=True)
            logging.info(
                f"[MICAStage1Trainer] Saving LoRA adapter to: {paths.best_lora_dir}"
            )
            model.vision_encoder.save_lora(paths.best_lora_dir)
            if paths.best_lora_dir.exists():
                files = list(paths.best_lora_dir.iterdir())
                logging.info(
                    f"[MICAStage1Trainer] LoRA adapter saved. Files: {[f.name for f in files]}"
                )
            else:
                logging.error(f"[MICAStage1Trainer] ERROR: LoRA dir not created!")
        else:
            # fallback: full model
            paths.best_stage1_ckpt = self.ckpt_dir / "best_stage1.pt"
            logging.info(
                f"[MICAStage1Trainer] Saving full checkpoint to: {paths.best_stage1_ckpt}"
            )
            torch.save(model.state_dict(), paths.best_stage1_ckpt)
            if paths.best_stage1_ckpt.exists():
                logging.info(
                    f"[MICAStage1Trainer] Checkpoint saved. Size: {paths.best_stage1_ckpt.stat().st_size / 1e6:.2f} MB"
                )
            else:
                logging.error(f"[MICAStage1Trainer] ERROR: Checkpoint not created!")

        logging.info("-" * 40)
        return paths

    def test_from_ckpt(
        self, best_paths: CheckpointPaths, test_loader: DataLoader
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Stage-1 is typically not evaluated like classification. Use stage-2 CBM trainer."
        )

    def fit(self, path_to_cav_file) -> CheckpointPaths:
        set_seed(self.seed)
        logging.info("=" * 60)
        logging.info("[MICAStage1Trainer] Starting Stage 1 training")
        logging.info(f"[MICAStage1Trainer] exp_dir: {self.exp_dir}")
        logging.info(f"[MICAStage1Trainer] split: {self.split_number}")
        logging.info(f"[MICAStage1Trainer] epochs: {self.cfg.TRAIN.epochs}")
        logging.info(f"[MICAStage1Trainer] CAV file: {path_to_cav_file}")
        logging.info("=" * 60)
        tokenizer = instantiate(self.cfg.MODEL.tokenizer)
        max_len = int(self.cfg.DATASET.text.word_num)
        collate = lambda b: mica_collate_hf_tokenizer(b, tokenizer, max_len)
        train_loader, val_loader = self.build_loaders(collate_fn=collate)
        model = self.build_model(path_to_cav_file).to(self.device)

        optimizer, scheduler = self.build_optimizer_and_scheduler(model, train_loader)
        scaler = torch.amp.GradScaler()

        best = self.worst
        best_paths = CheckpointPaths()

        history = []

        for epoch in range(int(self.cfg.TRAIN.epochs)):
            logging.info(
                f"\n======================> epoch: {epoch} <======================"
            )

            tr = self.train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
            va = self.validate(model, val_loader)

            sel = self.select_metric_name()
            cur = float(va.get(sel))
            self._log_fit_history(history, epoch, tr, va)

            best_paths, best = self.save_and_log_best(sel, cur, best, best_paths, model)

        self.finilize_fit(history)
        logging.info("=" * 60)
        logging.info("[MICAStage1Trainer] Stage 1 training complete")
        logging.info(
            f"[MICAStage1Trainer] Best paths: lora_dir={best_paths.best_lora_dir}, ckpt={best_paths.best_stage1_ckpt}"
        )
        logging.info("=" * 60)

        return best_paths
