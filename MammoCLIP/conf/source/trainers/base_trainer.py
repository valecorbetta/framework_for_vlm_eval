import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from omegaconf import DictConfig
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from source.utils.data import prepare_transforms_embed
from source.utils.overlay_spurious import (
    get_applied_overlay_pct,
    log_overlay_stats_per_class,
)
from source.utils.checkpoints import CheckpointPaths


class BaseTrainer:
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
        overlay_cfg_test: Optional[DictConfig],
    ):
        self.cfg = cfg
        self.device = device
        self.exp_dir = exp_dir
        self.seed = seed
        self.split_number = split_number
        self.csv_train_path = csv_train_path
        self.csv_val_path = csv_val_path
        self.csv_test_path = csv_test_path
        self.path_to_images = path_to_images
        self.overlay_cfg_train = overlay_cfg_train
        self.overlay_cfg_test = overlay_cfg_test
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.sel_name = None

    def select_metric_name(self) -> str:
        return self.sel_name

    def build_loaders(self, collate_fn: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader]:  # type: ignore
        aug, prep = prepare_transforms_embed(self.seed)

        self.train_dataset = instantiate(
            self.cfg.DATASET._class,
            self.csv_train_path,
            self.path_to_images,
            augmentations=aug,
            preprocessing=prep,
            label_mode=self.cfg.TASK.label_mode,
            overlay_cfg=self.overlay_cfg_train,
        )

        log_overlay_stats_per_class(self.train_dataset, split_name="train")

        val_dataset = instantiate(
            self.cfg.DATASET._class,
            self.csv_val_path,
            self.path_to_images,
            augmentations=None,
            preprocessing=prep,
            label_mode=self.cfg.TASK.label_mode,
            overlay_cfg=self.overlay_cfg_train,
        )

        log_overlay_stats_per_class(val_dataset, split_name="val")

        g = torch.Generator()
        g.manual_seed(self.seed)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.batch_size,
            shuffle=True,
            num_workers=self.cfg.TRAIN.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.TRAIN.batch_size,
            shuffle=False,
            num_workers=self.cfg.TRAIN.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
        )

        return train_loader, val_loader

    def build_test_loader(self, collate_fn: Optional[Callable] = None) -> DataLoader:
        _, prep = prepare_transforms_embed(self.seed)
        if self.csv_test_path is not None:
            test_dataset = instantiate(
                self.cfg.DATASET._class,
                self.csv_test_path,
                self.path_to_images,
                augmentations=None,
                preprocessing=prep,
                label_mode=self.cfg.TASK.label_mode,
                overlay_cfg=self.overlay_cfg_test,
            )

            log_overlay_stats_per_class(test_dataset, split_name="test")

            g = torch.Generator()
            g.manual_seed(self.seed)
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.EVAL.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                generator=g,
                worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            )
        else:
            test_loader = None

        return test_loader

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def build_optimizer_and_scheduler(
        self, model: nn.Module, train_loader: DataLoader
    ) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        raise NotImplementedError

    def build_classification_loss(self, train_dataset: pd.DataFrame) -> nn.Module:
        use_class_weights = bool(getattr(self.cfg.TRAIN, "use_class_weights", True))
        if (not use_class_weights) or (train_dataset is None):
            return instantiate(self.cfg.TASK.loss)

        y_train = np.asarray(train_dataset.labels)
        if self.is_binary:
            # For BCEWithLogitsLoss: pos_weight multiplies the loss on positive examples
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            if n_pos > 0 and n_neg > 0 and not use_class_weights:
                pos_weight = torch.tensor(
                    [n_neg / n_pos], dtype=torch.float32, device=self.device
                )
                logging.info(
                    f"[CLASS-WEIGHTS][binary] pos_weight={pos_weight.item():.4f}"
                )
                return instantiate(self.cfg.TASK.loss, pos_weight=pos_weight)
            else:
                logging.warning("[CLASS-WEIGHTS][binary] missing class; unweighted.")
                return instantiate(self.cfg.TASK.loss)
        else:
            C = int(y_train.max()) + 1
            counts = np.bincount(y_train, minlength=C).astype(np.float64)
            w = counts.sum() / np.maximum(counts, 1.0)
            w = w / (w.mean() if w.mean() > 0 else 1.0)
            w = np.clip(w, 0.5, None)
            class_weights = torch.tensor(w, dtype=torch.float32, device=self.device)
            logging.info(
                f"[CLASS-WEIGHTS][multiclass] weights={class_weights.round(4).tolist()}"
            )
            return instantiate(self.cfg.TASK.loss, weight=class_weights)

    def build_concept_loss(self, train_dataset) -> nn.BCEWithLogitsLoss:
        """
        Build a BCE loss for the concept head with per-concept pos_weight
        derived from training label frequencies.
        """
        use_weights = bool(getattr(self.cfg.TRAIN, "use_concept_weights", True))
        if not use_weights or train_dataset is None:
            logging.info(
                "[CONCEPT-WEIGHTS] Disabled or no dataset; using unweighted BCE."
            )
            return nn.BCEWithLogitsLoss(reduction="mean")

        # Fast path: read concept flags directly from the DataFrame
        concept_keys = getattr(train_dataset, "concept_keys", None)
        if concept_keys is not None and hasattr(train_dataset, "df"):
            concept_labels = train_dataset.df[concept_keys].values.astype(np.float64)
        else:
            # Fallback: build from individual samples (slow but general)
            concept_labels = np.stack(
                [sample["concept_labels"].numpy() for sample in train_dataset]
            )

        n_samples, n_concepts = concept_labels.shape
        pos_counts = concept_labels.sum(axis=0)  # [Nc]
        neg_counts = n_samples - pos_counts  # [Nc]

        # pos_weight = n_neg / n_pos for each concept (clamp to avoid div-by-zero)
        pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        # pos_weight = np.clip(pos_weight, 1.0, 20.0)
        # pos_weight = np.clip(np.sqrt(pos_weight), 1.0, 30.0)

        logging.info(f"[CONCEPT-WEIGHTS] pos_weight per concept: {pos_weight.tolist()}")
        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self.device), reduction="mean"
        )

    def build_classification_loss(self, train_dataset: pd.DataFrame) -> nn.Module:
        use_class_weights = bool(getattr(self.cfg.TRAIN, "use_class_weights", True))
        if (not use_class_weights) or (train_dataset is None):
            return instantiate(self.cfg.TASK.loss)

        y_train = np.asarray(train_dataset.labels)
        if self.is_binary:
            # For BCEWithLogitsLoss: pos_weight multiplies the loss on positive examples
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor(
                    [n_neg / n_pos], dtype=torch.float32, device=self.device
                )
                logging.info(
                    f"[CLASS-WEIGHTS][binary] pos_weight={pos_weight.item():.4f}"
                )
                return instantiate(self.cfg.TASK.loss, pos_weight=pos_weight)
            else:
                logging.warning("[CLASS-WEIGHTS][binary] missing class; unweighted.")
                return instantiate(self.cfg.TASK.loss)
        else:
            C = int(y_train.max()) + 1
            counts = np.bincount(y_train, minlength=C).astype(np.float64)
            w = counts.sum() / np.maximum(counts, 1.0)
            w = w / (w.mean() if w.mean() > 0 else 1.0)
            class_weights = torch.tensor(w, dtype=torch.float32, device=self.device)
            logging.info(f"[CLASS-WEIGHTS][multiclass] weights={w.round(4).tolist()}")
            return instantiate(self.cfg.TASK.loss, weight=class_weights)

    @staticmethod
    def _get_binary_output(
        criterion, logits: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor]:
        loss = criterion(logits.squeeze(-1), y.float())
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= 0.5).long()

        return loss, probs, preds

    @staticmethod
    def _get_multiclass_output(
        criterion, logits: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor]:
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return loss, probs, preds

    @staticmethod
    def _collect_metadata(batch: torch.Tensor, meta_rows: list) -> list[dict]:
        concept_keys = [k for k in batch.keys() if k.startswith("has_")]
        batch_size = len(batch["filename"])
        for i in range(batch_size):
            row_dict = {"filename": batch["filename"][i]}
            for key in concept_keys:
                row_dict[key] = bool(batch[key][i].item())
            meta_rows.append(row_dict)

        return meta_rows

    @staticmethod
    def _compute_shared_metrics(
        losses: list[float], y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        # shared because they are for both the binary and multiclass case
        metrics = {}
        metrics["loss"] = float(np.mean(losses))
        metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        return metrics

    @staticmethod
    def _compute_binary_metrics(
        metrics: dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        metrics["auroc"] = (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )
        # Add F1 score
        try:
            metrics["f1"] = float(f1_score(y_true, y_pred))
        except Exception:
            metrics["f1"] = float("nan")
            logging.info(f"[TEST] Was not able to compute the f1 score.")
        metrics["threshold"] = 0.5

        return metrics

    @staticmethod
    def _compute_f1_score_multiclass(
        metrics: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        # Add macro F1 for multiclass
        try:
            metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        except Exception:
            metrics["macro_f1"] = float("nan")
            logging.info(f"[TEST] Was not able to compute the f1 score.")
        return metrics

    def _get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ):
        try:
            n_classes = 2 if self.is_binary else y_prob.shape[1]
            cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        except Exception:
            cm = None
            logging.info(f"[TEST] Was not able to compute the confusion matrix.")

        return cm

    def _save_metrics(
        self,
        metrics: dict[str, float],
        meta_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        cm: np.ndarray,
        sub_df: pd.DataFrame,
    ) -> None:
        test_dir = self.exp_dir / "test_only" / self.cfg.DATASET.overlay_cfg_test.mode
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"[TEST] Metrics saved to {metrics_path}")

        # 2) Predictions with metadata
        if not meta_df.empty:
            df = meta_df.copy()
            df["y_true"] = y_true.astype(int)
            df["y_pred"] = y_pred.astype(int)
            if self.is_binary:
                df["prob_pos"] = y_prob.astype(float)
            else:
                df["top1_prob"] = np.max(y_prob, axis=1).astype(float)
                df["top1_cls"] = np.argmax(y_prob, axis=1).astype(int)
            predictions_path = self.exp_dir / "predictions.csv"
            df.to_csv(predictions_path, index=False)
            logging.info(f"[TEST] Predictions saved to {predictions_path}")

        # 3) Raw arrays
        np.save(self.exp_dir / "preds.npy", y_pred)
        np.save(self.exp_dir / "probs.npy", y_prob)

        # 4) Confusion matrix
        if cm is not None:
            np.save(self.exp_dir / "confusion_matrix.npy", cm)
            logging.info(f"[TEST] Confusion matrix saved")

        # 5) Per-class accuracy (multiclass only)
        if not self.is_binary and y_prob.size > 0:
            per_class_acc = []
            for c in range(y_prob.shape[1]):
                m = y_true == c
                acc_c = float((y_pred[m] == c).mean()) if m.any() else float("nan")
                per_class_acc.append({"class": c, "acc": acc_c, "n": int(m.sum())})
            per_class_path = self.exp_dir / "per_class_accuracy.csv"
            pd.DataFrame(per_class_acc).to_csv(per_class_path, index=False)
            logging.info(f"[TEST] Per-class accuracy saved to {per_class_path}")

        # 6) Subgroup metrics
        if not sub_df.empty:
            subgroup_path = self.exp_dir / "subgroup_metrics.csv"
            sub_df.to_csv(subgroup_path, index=False)
            logging.info(f"[TEST] Subgroup metrics saved to {subgroup_path}")

    def train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        criterion: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def is_better(self, cur: float, best: float, mode="min") -> bool:
        raise ValueError("NOOO")

    def save_best(self, model: nn.Module) -> CheckpointPaths:
        raise NotImplementedError

    def load_for_test(
        self,
        model: nn.Module,
        best_paths: CheckpointPaths,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _plot_curves(
        self, history: List[Dict], keys: List[str], out_path: Path, title: str
    ) -> None:
        epochs = [h["epoch"] for h in history]
        fig, ax = plt.subplots()
        for k in keys:
            ys = [h.get(k, float("nan")) for h in history]
            ax.plot(epochs, ys, label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        fig.savefig(out_path)
        plt.close(fig)

    def finilize_fit(self, history: List[Dict]) -> None:
        """
        Docstring for finilize_fit

        :param self:
        :param history: dictionaries of losses and metrics throughout the epochs
        :type history: List[Dict]

        Generic post-fit hook:
            - saves plots for loss/metric keys found in history
            - saves val_metrics.json
        """
        with open(self.exp_dir / "fit_history.json", "w") as f:
            json.dump(history, f, indent=2)
        figs_dir = self.exp_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        if len(history) == 0:
            return
        else:
            all_keys = sorted({k for h in history for k in h.keys() if k != "epochs"})
            # ---- Plot losses ----
            # Any key containing "loss" gets plotted
            loss_keys = [k for k in all_keys if "loss" in k.lower()]
            if loss_keys:
                self._plot_curves(
                    history,
                    loss_keys,
                    figs_dir / f"loss_curves_fold_{self.split_number}.pdf",
                    title=f"Split {self.split_number} Loss Curves",
                )

            # ---- Plot metrics (optional) ----
            metric_keys = [k for k in all_keys if ("loss" not in k.lower())]
            if metric_keys:
                self._plot_curves(
                    history,
                    metric_keys,
                    figs_dir / f"metric_curves_fold_{self.split_number}.pdf",
                    title=f"Split {self.split_number} Metrics",
                )

            # ---- Save Optuna-friendly JSON ----
            sel = self.select_metric_name()
            val_sel_key = f"val_{sel}"
            best_metric = float("nan")
            if any(val_sel_key in h for h in history):
                vals = [h.get(val_sel_key, float("nan")) for h in history]
                if self.is_better(1.0, 0.0):  # heuristic: higher-is-better trainers
                    best_metric = float(np.nanmax(vals))
                else:
                    best_metric = float(np.nanmin(vals))

            val_metrics = {
                "best_metric": best_metric,
                "sel_name": sel,
                "epochs": int(self.cfg.TRAIN.epochs),
            }
            # store every curve as a list
            for k in all_keys:
                val_metrics[k] = [float(h.get(k, float("nan"))) for h in history]

            with open(self.exp_dir / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)

    def _log_fit_history(
        self,
        history: list[dict],
        epoch: int,
        tr: Dict[str, float],
        va: Dict[str, float],
    ) -> None:
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
            }
        )

        logging.info(f"[epoch={epoch}] train={tr} | val={va}")

    def save_and_log_best(
        self,
        sel: str,
        cur: float,
        best: float,
        best_paths: CheckpointPaths,
        model: nn.Module,
    ) -> Tuple[CheckpointPaths, float]:
        if np.isfinite(cur) and self.is_better(cur, best):
            best = cur
            best_paths = self.save_best(model)
            logging.info(f"New best {sel}={best:.6f}. Saved checkponts.")
        return best_paths, best

    # ---------- Generic fit loop ---------- #
    def fit(self) -> CheckpointPaths:
        raise NotImplementedError
