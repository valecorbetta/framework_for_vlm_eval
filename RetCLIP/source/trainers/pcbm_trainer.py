import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from peft import PeftModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from RetCLIP.source.utils.compute_subgroup_metrics import (
    subgroup_metrics_binary_concepts,
    subgroup_metrics_multiclass_concepts,
)
from RetCLIP.source.utils.fairness import compute_multigroup_multiclass_fairness
from RetCLIP.source.utils.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
)

from RetCLIP.source.trainers.base_trainer import BaseTrainer
from RetCLIP.source.utils.checkpoints import CheckpointPaths
from RetCLIP.source.model.pcbm import PostHocCBM
from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.data.dataset_FGADR import (
    FGADRConceptDataset,
    mica_collate_fulltokenizer,
)
from RetCLIP.source.utils.misc import set_seed


class PostHocCAVCBMTrainer(BaseTrainer):
    """
    Post-hoc CAV-CBM:
      - vision encoder (RetCLIP) -> frozen by default
      - fixed concept layer (CAV margins)
      - train predictor only (concept->diag)
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
        csv_test_path: Optional[Path],
        path_to_images: Path,
        overlay_cfg_train: DictConfig,
        overlay_cfg_test: DictConfig,
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
        self.is_binary = cfg.TASK.name == "binary"
        self.sel_name = "auroc" if self.is_binary else "accuracy"
        self.worst = -float("inf")

    def is_better(self, cur: float, best: float) -> bool:
        return cur >= best

    def build_model(self, path_to_cav_file: Path) -> PostHocCBM:

        return instantiate(
            self.cfg.MODEL._class,
            device=self.device,
            cav_pkl=path_to_cav_file,
        ).to(self.device)

    def build_optimizer_and_scheduler(self, model: nn.Module, train_loader: DataLoader):
        """
        Post-hoc baseline typically trains predictor only. If freeze_encoder=True, this will be only predictor params.
        If you later enable LoRA/finetuning, parameters will show up as requires_grad=True.
        """
        # Mirror FundusClassifier split if you enable LoRA finetuning in this baseline
        classifier_params, lora_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if "lora" in name else classifier_params).append(p)

        # If you don't use LoRA, lora_params will be empty and it's fine.
        lr_lora = float(
            getattr(self.cfg.TRAIN, "lr_lora", getattr(self.cfg.TRAIN, "lr", 1e-4))
        )
        lr_cls = float(
            getattr(
                self.cfg.TRAIN, "lr_classifier", getattr(self.cfg.TRAIN, "lr", 1e-4)
            )
        )

        optimizer = instantiate(
            self.cfg.OPTIMIZER,
            [
                {"params": lora_params, "lr": lr_lora},
                {"params": classifier_params, "lr": lr_cls},
            ],
        )

        if self.cfg.TRAIN.enable_scheduler:
            scheduler = instantiate(
                self.cfg.SCHEDULER,
                optimizer,
                total_epochs=self.cfg.TRAIN.epochs,
                warmup_steps=len(train_loader),
                total_steps=len(train_loader) * self.cfg.TRAIN.epochs,
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def _step(self, model: PostHocCBM, batch: dict, diag_criterion: nn.Module):
        concept_scores, diag_logits = model(batch)
        diag_labels = batch["y"]

        if self.is_binary:
            # BCEWithLogitsLoss expects [B] or [B,1] logits and float labels
            diag_loss, probs_pos, preds = self._get_binary_output(
                diag_criterion, diag_logits, diag_labels
            )
        else:
            # e.g. CrossEntropyLoss
            diag_loss, probs_pos, preds = self._get_multiclass_output(
                diag_criterion, diag_logits, diag_labels
            )

        return diag_loss, preds, probs_pos, diag_labels

    def train_one_epoch(
        self, model, loader, optimizer, scheduler, scaler, criterion, epoch
    ) -> dict[str, float]:
        model.train()

        diag_losses = []
        y_true, y_pred, y_prob = [], [], []

        for batch in tqdm(loader):
            optimizer.zero_grad(set_to_none=True)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                d_loss, preds, probs, labels = self._step(model, batch, criterion)

            scaler.scale(d_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            diag_losses.append(float(d_loss.item()))

            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            if probs is not None:
                y_prob.append(probs.detach().cpu())

        y_true_np = torch.cat(y_true).numpy()
        y_pred_np = torch.cat(y_pred).numpy()

        out = {
            "diag_loss": float(np.mean(diag_losses)),
            "balanced_acc": float(balanced_accuracy_score(y_true_np, y_pred_np)),
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        }
        if self.is_binary:
            y_prob_np = torch.cat(y_prob).numpy()
            out["auroc"] = (
                float(roc_auc_score(y_true_np, y_prob_np))
                if len(np.unique(y_true_np)) == 2
                else float("nan")
            )
        return out

    def validate(self, model, loader, criterion, epoch) -> Dict[str, float]:
        model.eval()

        diag_losses = []
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                d_loss, preds, probs, labels = self._step(model, batch, criterion)

                diag_losses.append(float(d_loss.item()))

                y_true.append(labels.detach().cpu())
                y_pred.append(preds.detach().cpu())
                if probs is not None:
                    y_prob.append(probs.detach().cpu())

        y_true_np = torch.cat(y_true).numpy()
        y_pred_np = torch.cat(y_pred).numpy()

        out = {
            "diag_loss": float(np.mean(diag_losses)),
            "balanced_acc": float(balanced_accuracy_score(y_true_np, y_pred_np)),
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        }
        if self.is_binary:
            y_prob_np = torch.cat(y_prob).numpy()
            out["auroc"] = (
                float(roc_auc_score(y_true_np, y_prob_np))
                if len(np.unique(y_true_np)) == 2
                else float("nan")
            )
        return out

    def save_best(self, model: PostHocCBM) -> CheckpointPaths:
        paths = CheckpointPaths()

        if self.cfg.TUNE.mode == "linear_probing":
            paths.best_classifier_head = self.ckpt_dir / "best_classifier_head.ckpt"
            torch.save(model.predictor.state_dict(), paths.best_classifier_head)
            paths.best_lora_dir = None

        elif self.cfg.TUNE.mode == "lora":
            paths.best_classifier_head = self.ckpt_dir / "best_classifier_head.ckpt"
            torch.save(model.predictor.state_dict(), paths.best_classifier_head)

            paths.best_lora_dir = self.ckpt_dir / "best_lora_adapter"
            paths.best_lora_dir.mkdir(exist_ok=True, parents=True)
            model.vision_encoder.save_lora(paths.best_lora_dir)

        else:
            # fallback: full model
            paths.best_stage2_ckpt = self.ckpt_dir / "best_model.pt"
            torch.save(model.state_dict(), paths.best_stage2_ckpt)

        manifest = {}

        if paths.best_classifier_head is not None:
            manifest["classifier_head"] = "checkpoints/best_classifier_head.ckpt"

        if paths.best_lora_dir is not None:
            manifest["lora_dir"] = "checkpoints/best_lora_adapter"

        with open(self.exp_dir / "checkpoints" / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return paths

    def load_for_test(self, model: PostHocCBM, best_paths: CheckpointPaths) -> None:
        """
        Loads experiment-specific weights only.
        Base RetCLIP vision weights are loaded at model construction time
        via VisionViT.__init__.
        """
        # Always load classifier head if it exists
        if best_paths.best_classifier_head is not None:
            model.predictor.load_state_dict(
                torch.load(best_paths.best_classifier_head, map_location=self.device)
            )
            # Add debug logging
        logging.info(f"[DEBUG] best_lora_dir = {best_paths.best_lora_dir}")
        logging.info(f"[DEBUG] best_lora_dir type = {type(best_paths.best_lora_dir)}")
        logging.info(
            f"[DEBUG] best_lora_dir is None? {best_paths.best_lora_dir is None}"
        )
        if best_paths.best_lora_dir is not None:
            logging.info(
                f"[DEBUG] best_lora_dir.is_dir()? {best_paths.best_lora_dir.is_dir()}"
            )

        # Load LoRA adapter only if directory exists
        if best_paths.best_lora_dir is not None and best_paths.best_lora_dir.is_dir():
            model.vision_encoder.load_lora(best_paths.best_lora_dir)
        return model

    def test_from_ckpt(
        self,
        best_paths: CheckpointPaths,
        test_loader: DataLoader,
        path_to_cav_file: Path,
    ) -> Dict[str, Any]:
        logging.info("Instantiating test model...")
        cfg_model = OmegaConf.merge(
            self.cfg.MODEL._class, {"vision_encoder": {"lora": False}}
        )
        model: PostHocCBM = (
            instantiate(cfg_model, device=self.device, cav_pkl=path_to_cav_file)
            .float()
            .to(self.device)
        )
        model = self.load_for_test(model, best_paths)
        logging.info(
            "Test model instantiated, has_lora=%s", model.vision_encoder.has_lora()
        )
        model.eval()

        criterion = instantiate(self.cfg.TASK.loss).to(self.device)

        losses = []
        y_true, y_pred, y_prob = [], [], []
        meta_rows = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                _, logits = model({"x": x})

                if self.is_binary:
                    loss, probs, preds = self._get_binary_output(criterion, logits, y)
                    y_prob.append(probs.detach().cpu())
                else:
                    loss, probs, preds = self._get_multiclass_output(
                        criterion, logits, y
                    )
                    y_prob.append(probs.detach().cpu())

                losses.append(float(loss.item()))
                y_true.append(y.detach().cpu())
                y_pred.append(preds.detach().cpu())

                # Collect metadata for subgroup metrics
                meta_rows = self._collect_metadata(batch, meta_rows)

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        if self.is_binary:
            y_prob = torch.cat(y_prob).numpy().reshape(-1)
        else:
            y_prob = torch.cat(y_prob).numpy()

        # --- Compute metrics ---
        metrics = self._compute_shared_metrics(losses, y_true, y_pred)

        if self.is_binary:
            metrics = self._compute_binary_metrics(metrics, y_true, y_pred, y_prob)
        else:
            # Add macro F1 for multiclass
            metrics = self._compute_f1_score_multiclass(metrics, y_true, y_pred)

        # --- Confusion matrix ---
        cm = self._get_confusion_matrix(y_true, y_pred, y_prob)

        # --- Subgroup metrics ---
        meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()
        group_df = pd.DataFrame()
        sub_df = pd.DataFrame()

        if not meta_df.empty:
            if self.is_binary:
                sub_df = subgroup_metrics_binary_concepts(
                    meta_df=meta_df,
                    y_true=y_true.astype(int),
                    y_pred=y_pred.astype(int),
                    y_prob_pos=y_prob.astype(float),
                )
            else:
                sub_df = subgroup_metrics_multiclass_concepts(
                    meta_df=meta_df,
                    y_true_0based=y_true.astype(int),
                    y_pred_0based=y_pred.astype(int),
                )
        # --- Waterbirds metrics  ---
        mode = (
            str(self.overlay_cfg_test.get("mode", "")).lower()
            if self.overlay_cfg_test
            else ""
        )
        if (
            (not meta_df.empty)
            and ("spurious_applied" in meta_df.columns)
            and (mode == "waterbirds")
        ):
            a = meta_df["spurious_applied"].to_numpy(dtype=int)
            unique_groups = sorted(np.unique(a).tolist())
            num_classes = int(np.max(y_true)) + 1

            group_df, worst_group_acc, max_within_class_gap = (
                self._waterbirds_group_accuracy(y_true, y_pred, a)
            )
            metrics["worst_group_acc"] = float(worst_group_acc)
            metrics["max_within_class_acc_gap"] = float(max_within_class_gap)

            fairness_dict = compute_multigroup_multiclass_fairness(
                y_true=y_true.astype(int),
                y_pred=y_pred.astype(int),
                protected_groups=a,
                num_classes=num_classes,
                unique_groups=unique_groups,
            )
            metrics["fairness_groups"] = unique_groups
            metrics["fairness_eod_aod_by_class"] = fairness_dict

            # calibration
            if y_prob.ndim == 2:
                confs = np.max(y_prob, axis=1).astype(float)
            else:
                confs = np.maximum(y_prob, 1.0 - y_prob).astype(float)

            metrics["ece"] = float(
                expected_calibration_error(
                    confs, y_pred.astype(int), y_true.astype(int), num_bins=10
                )
            )
            metrics["mce"] = float(
                maximum_calibration_error(
                    confs, y_pred.astype(int), y_true.astype(int), num_bins=10
                )
            )

            for aa in [0, 1]:
                m = a == aa
                if m.any():
                    metrics[f"ece_a{aa}"] = float(
                        expected_calibration_error(
                            confs[m],
                            y_pred[m].astype(int),
                            y_true[m].astype(int),
                            num_bins=10,
                        )
                    )
                    metrics[f"mce_a{aa}"] = float(
                        maximum_calibration_error(
                            confs[m],
                            y_pred[m].astype(int),
                            y_true[m].astype(int),
                            num_bins=10,
                        )
                    )
                else:
                    metrics[f"ece_a{aa}"] = float("nan")
                    metrics[f"mce_a{aa}"] = float("nan")

        # --- Save results ---
        self._save_metrics(
            metrics, meta_df, y_true, y_pred, y_prob, cm, sub_df, group_df
        )

        return {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "subgroup_metrics": sub_df,
            "model": model,
        }

    def fit(self, path_to_cav_file: Path) -> CheckpointPaths:
        set_seed(self.seed)
        tokenizer = instantiate(self.cfg.MODEL.tokenizer)
        max_len = int(self.cfg.DATASET.text.word_num)
        collate = lambda b: mica_collate_fulltokenizer(b, tokenizer, max_len)
        train_loader, val_loader = self.build_loaders(collate_fn=collate)
        model = self.build_model(path_to_cav_file).to(self.device)

        criterion = self.build_classification_loss(self.train_dataset).to(self.device)

        optimizer, scheduler = self.build_optimizer_and_scheduler(model, train_loader)
        scaler = torch.amp.GradScaler()

        best = self.worst
        best_paths = CheckpointPaths()

        history = []

        for epoch in range(int(self.cfg.TRAIN.epochs)):
            logging.info(
                f"\n======================> epoch: {epoch} <======================"
            )
            tr = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, criterion, epoch
            )
            va = self.validate(model, val_loader, criterion, epoch)

            sel = self.select_metric_name()
            cur = float(va.get(sel))
            self._log_fit_history(history, epoch, tr, va)

            best_paths, best = self.save_and_log_best(sel, cur, best, best_paths, model)

        self.finilize_fit(history)

        return best_paths
