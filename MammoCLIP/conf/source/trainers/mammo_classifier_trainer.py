import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from hydra.utils import instantiate
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
)
from source.utils.misc import set_seed

from source.trainers.base_trainer import BaseTrainer
from source.utils.checkpoints import CheckpointPaths
from source.utils.compute_subgroup_metrics import (
    subgroup_metrics_binary_concepts,
    subgroup_metrics_multiclass_concepts,
)

from source.models.mammo_classifier import MammoClassifier
from source.data.embed_dataset import EMBEDDataset


class MammoClassifierTrainer(BaseTrainer):
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

        self.is_binary = cfg.TASK.name == "binary"
        self.sel_name = "balanced_acc" if self.is_binary else "balanced_acc"
        self.worst = -float("inf")

    def is_better(self, cur: float, best: float) -> bool:
        return cur >= best

    def build_model(self) -> MammoClassifier:
        model = instantiate(self.cfg.MODEL._class)
        return model

    def build_optimizer_and_scheduler(self, model: nn.Module, train_loader: DataLoader):
        classifier_params, lora_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if "lora" in name else classifier_params).append(p)

        optimizer = instantiate(
            self.cfg.OPTIMIZER,
            [
                {"params": lora_params, "lr": self.cfg.TUNE.lr_lora},
                {"params": classifier_params, "lr": self.cfg.TUNE.lr_head},
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

    def train_one_epoch(
        self, model, train_loader, optimizer, scheduler, scaler, criterion, epoch
    ) -> dict[str, float]:
        model.train()
        train_losses = []
        y_true, y_pred, y_prob = [], [], []

        for batch in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                logits = model(x)
                if self.is_binary:
                    loss, probs, preds = self._get_binary_output(criterion, logits, y)
                    y_prob.append(probs.detach().cpu())
                else:
                    loss, probs, preds = self._get_multiclass_output(
                        criterion, logits, y
                    )
                    preds = logits.argmax(dim=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            train_losses.append(float(loss.item()))
            y_true.append(y.detach().cpu())
            y_pred.append(preds.detach().cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        out = {
            "loss": float(np.mean(train_losses)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }
        if self.is_binary:
            y_prob = torch.cat(y_prob).numpy()
            out["auroc"] = (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) == 2
                else float("nan")
            )
        else:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        return out

    def validate(self, model, val_loader, criterion, epoch) -> Dict[str, float]:
        model.eval()
        losses = []
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                logits = model(x)

                if self.is_binary:
                    loss, probs, preds = self._get_binary_output(criterion, logits, y)
                    y_prob.append(probs.detach().cpu())
                else:
                    loss, probs, preds = self._get_multiclass_output(
                        criterion, logits, y
                    )
                    preds = logits.argmax(dim=1)

                losses.append(float(loss.item()))
                y_true.append(y.detach().cpu())
                y_pred.append(preds.detach().cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        out = {
            "loss": float(np.mean(losses)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }
        if self.is_binary:
            y_prob = torch.cat(y_prob).numpy()
            out["auroc"] = (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) == 2
                else float("nan")
            )
        else:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        return out

    def save_best(self, model: MammoClassifier) -> CheckpointPaths:
        paths = CheckpointPaths()

        if self.cfg.TUNE.mode == "linear_probing":
            paths.best_classifier_head = self.ckpt_dir / "best_classifier_head.ckpt"
            torch.save(model.classifier.state_dict(), paths.best_classifier_head)
            paths.best_lora_dir = None

        elif self.cfg.TUNE.mode == "lora":
            paths.best_classifier_head = self.ckpt_dir / "best_classifier_head.ckpt"
            torch.save(model.classifier.state_dict(), paths.best_classifier_head)

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

    def _format_float_token(self, x: float, ndigits: int = 4) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "nan"
        return f"{x:.{ndigits}f}".replace(".", "p")

    def _epoch_prefix(self, epoch: int, val_metrics: Dict[str, float]) -> str:
        sel_name = self.select_metric_name()  # "auroc" or "accuracy"
        vloss = float(val_metrics.get("loss", float("nan")))
        vsel = float(val_metrics.get(sel_name, float("nan")))

        loss_tok = self._format_float_token(vloss, ndigits=4)
        sel_tok = self._format_float_token(vsel, ndigits=4)

        return f"epoch_{epoch:03d}-valloss_{loss_tok}-{sel_name}_{sel_tok}"

    def save_epoch_checkpoint(
        self,
        model: "MammoClassifier",
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> CheckpointPaths:
        epoch_dir = self.ckpt_dir / "per_epoch"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        prefix = self._epoch_prefix(epoch, val_metrics)

        paths = CheckpointPaths()

        # Always save classifier head
        paths.best_classifier_head = epoch_dir / f"{prefix}-classifier_head.ckpt"
        torch.save(model.classifier.state_dict(), paths.best_classifier_head)

        # Save LoRA adapter only when relevant
        if self.cfg.TUNE.mode == "lora":
            paths.best_lora_dir = epoch_dir / f"{prefix}-lora_adapter"
            paths.best_lora_dir.mkdir(parents=True, exist_ok=True)
            model.vision_encoder.save_lora(paths.best_lora_dir)
        else:
            paths.best_lora_dir = None

        # Manifest JSON used to load this epoch later
        manifest = {
            "epoch": int(epoch),
            "tune_mode": str(self.cfg.TUNE.mode),
            "sel_name": self.select_metric_name(),
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val": {k: float(v) for k, v in val_metrics.items()},
            "artifacts": {
                "classifier_head": str(paths.best_classifier_head),
                "lora_dir": str(paths.best_lora_dir) if paths.best_lora_dir else None,
            },
        }
        with open(epoch_dir / f"{prefix}.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return paths

    def load_epoch_checkpoint_paths(self, epoch_manifest_json: Path) -> CheckpointPaths:
        with open(epoch_manifest_json, "r") as f:
            m = json.load(f)

        p = CheckpointPaths()
        art = m["artifacts"]

        p.best_classifier_head = (
            Path(art["classifier_head"]) if art.get("classifier_head") else None
        )
        p.best_lora_dir = Path(art["lora_dir"]) if art.get("lora_dir") else None
        return p

    def get_epoch_manifest(self, epoch: int) -> Path:
        epoch_dir = self.ckpt_dir / "per_epoch"
        if not epoch_dir.exists():
            raise FileNotFoundError(f"Per-epoch dir does not exist: {epoch_dir}")

        pattern = f"epoch_{epoch:03d}-*.json"
        matches = sorted(epoch_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No per-epoch manifest for epoch={epoch} with pattern '{pattern}' in {epoch_dir}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple manifests found for epoch={epoch}. "
                f"Got {len(matches)}: {[m.name for m in matches]}"
            )
        return matches[0]

    def resolve_eval_checkpoint_paths(
        self, best_paths: CheckpointPaths
    ) -> CheckpointPaths:
        """
        If cfg.EVAL.use_epoch_ckpt is True, load per-epoch checkpoint paths.
        Otherwise return best_paths (default behavior).
        """
        eval_cfg = self.cfg.get("EVAL", None)
        if eval_cfg is None:
            return best_paths  # no EVAL block -> keep current behavior

        use_epoch = bool(eval_cfg.get("use_epoch_ckpt", False))
        if not use_epoch:
            return best_paths

        # Option 1: explicit manifest path
        manifest_path = eval_cfg.get("epoch_manifest", None)
        if manifest_path not in (None, "null"):
            manifest_path = Path(manifest_path)
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"EVAL.epoch_manifest not found: {manifest_path}"
                )
            return self.load_epoch_checkpoint_paths(manifest_path)

        # Option 2: epoch number
        epoch = eval_cfg.get("epoch", None)
        if epoch is None:
            raise ValueError(
                "EVAL.use_epoch_ckpt=true but neither EVAL.epoch_manifest nor EVAL.epoch is set"
            )
        manifest = self.get_epoch_manifest(int(epoch))
        return self.load_epoch_checkpoint_paths(manifest)

    def load_for_test(
        self, model: MammoClassifier, best_paths: CheckpointPaths
    ) -> None:
        """
        Loads experiment-specific weights only.
        Base RetCLIP vision weights are loaded at model construction time
        via VisionViT.__init__.
        """
        # Always load classifier head if it exists
        if best_paths.best_classifier_head is not None:
            model.classifier.load_state_dict(
                torch.load(best_paths.best_classifier_head, map_location=self.device)
            )
            state_dict = torch.load(
                best_paths.best_classifier_head, map_location=self.device
            )
            logging.info(f"[DEBUG] classifier ckpt keys: {list(state_dict.keys())}")
            logging.info(
                f"[DEBUG] model.classifier keys: {list(model.classifier.state_dict().keys())}"
            )
            model.classifier.load_state_dict(state_dict)
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
        self, best_paths: CheckpointPaths, test_loader: DataLoader
    ) -> Dict[str, Any]:
        logging.info("Instantiating test model...")
        cfg_model = OmegaConf.merge(
            self.cfg.MODEL._class, {"vision_encoder": {"lora": False}}
        )
        model = instantiate(cfg_model).float().to(self.device)
        model = self.load_for_test(model, best_paths)
        logging.info(
            "Test model instantiated, has_lora=%s", model.vision_encoder.has_lora()
        )
        logging.info(
            f"[DEBUG] image_projection_global = {model.image_projection_global}"
        )
        logging.info(
            f"[DEBUG] ckpt_clip resolved = {cfg_model.get('ckpt_clip', 'NOT SET')}"
        )

        model.eval()
        # # After load_for_test, before model.eval()

        # with torch.no_grad():
        #     sample = next(iter(test_loader))["x"][:2].to(self.device)

        #     # Features BEFORE projection
        #     raw = model.vision_encoder(sample)
        #     logging.info(
        #         f"[DIAG] raw features: mean={raw.mean():.4f}, std={raw.std():.4f}, range=[{raw.min():.4f}, {raw.max():.4f}]"
        #     )

        #     # Features AFTER projection
        #     proj = model.image_projection_global(raw)
        #     logging.info(
        #         f"[DIAG] projected features: mean={proj.mean():.4f}, std={proj.std():.4f}, range=[{proj.min():.4f}, {proj.max():.4f}]"
        #     )

        #     # Logits
        #     logits = model.classifier(proj)
        #     logging.info(f"[DIAG] logits: {logits.squeeze().tolist()}")

        criterion = instantiate(self.cfg.TASK.loss).to(self.device)

        losses = []
        y_true, y_pred, y_prob = [], [], []
        meta_rows = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                logits = model(x)

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

        # --- Save results ---
        self._save_metrics(metrics, meta_df, y_true, y_pred, y_prob, cm, sub_df)

        return {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "subgroup_metrics": sub_df,
        }

    def fit(self) -> CheckpointPaths:
        set_seed(self.seed)
        train_loader, val_loader = self.build_loaders()
        model = self.build_model().to(self.device)

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

            if bool(self.cfg.TRAIN.get("save_per_epoch", True)):
                self.save_epoch_checkpoint(
                    model=model, epoch=epoch, train_metrics=tr, val_metrics=va
                )

            sel = self.select_metric_name()
            cur = float(va.get(sel))
            self._log_fit_history(history, epoch, tr, va)

            best_paths, best = self.save_and_log_best(sel, cur, best, best_paths, model)

        self.finilize_fit(history)

        return best_paths
