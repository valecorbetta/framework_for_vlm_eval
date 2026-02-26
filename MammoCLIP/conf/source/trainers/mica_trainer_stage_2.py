import numpy as np
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
from hydra.utils import instantiate
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from source.trainers.base_trainer import BaseTrainer
from source.utils.checkpoints import CheckpointPaths
from source.models.mica_stage_2 import MICAStage2CBM
from source.utils.compute_subgroup_metrics import (
    subgroup_metrics_binary_concepts,
    subgroup_metrics_multiclass_concepts,
)
from source.utils.data import mica_collate_hf_tokenizer
from source.utils.misc import set_seed


class MICAStage2CBMTrainer(BaseTrainer):
    """
    Replicates MICA Stage-2 CBM training:
      - concept prediction loss + diag prediction loss
      - diag predicted ONLY from concept logits
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
        self.sel_name = "balanced_acc" if self.is_binary else "balanced_acc"
        self.concept_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.worst = -float("inf")

    def is_better(self, cur: float, best: float) -> bool:
        return cur >= best

    def build_model(self) -> MICAStage2CBM | MICAStage2MultiTask:
        """
        Instantiate Stage-2 CBM via Hydra, then load Stage-1 vision encoder weights
        (either full checkpoint or LoRA adapter) into model.vision_encoder.
        """
        logging.info("=" * 50)
        logging.info("[MICAStage2Trainer] Building Stage 2 model")

        # Log the stage 2 config
        logging.info("[MICAStage2Trainer] Stage 2 config:")
        logging.info(
            f"    stage_1_ckpt: {getattr(self.cfg.MODEL.stage_2, 'stage_1_ckpt', None)}"
        )
        logging.info(
            f"    stage_1_lora_dir: {getattr(self.cfg.MODEL.stage_2, 'stage_1_lora_dir', None)}"
        )
        logging.info(
            f"    freeze_encoder: {getattr(self.cfg.MODEL.stage_2._class, 'freeze_encoder', True)}"
        )
        logging.info(
            f"    num_concepts: {getattr(self.cfg.DATASET.num_concepts, 'num_concepts', None)}"
        )
        logging.info(
            f"    concept_weight: {getattr(self.cfg.MODEL.stage_2, 'concept_weight', 1.0)}"
        )

        # Check if we should use LoRA in stage 2
        use_lora_stage2 = getattr(self.cfg.TUNE, "use_lora", False)
        freeze_encoder = getattr(self.cfg.MODEL.stage_2._class, "freeze_encoder", True)
        stage1_lora_dir = getattr(self.cfg.MODEL.stage_2, "stage_1_lora_dir", None)

        # Determine whether to instantiate with LoRA:
        # - If freeze_encoder=False and use_lora=True and no stage1_lora_dir: instantiate with lora=True
        # - If we have stage1_lora_dir: instantiate with lora=False, then load adapter (avoid double-wrapping)
        instantiate_with_lora = (
            use_lora_stage2 and not freeze_encoder and not stage1_lora_dir
        )

        logging.info(f"[MICAStage2Trainer] LoRA config:")
        logging.info(f"    use_lora_stage2: {use_lora_stage2}")
        logging.info(f"    freeze_encoder: {freeze_encoder}")
        logging.info(f"    stage1_lora_dir: {stage1_lora_dir}")
        logging.info(f"    instantiate_with_lora: {instantiate_with_lora}")

        cfg_stage2 = OmegaConf.merge(
            self.cfg.MODEL.stage_2._class,
            {
                "vision_encoder": {"lora": instantiate_with_lora},
                "freeze_encoder": freeze_encoder,
            },
        )
        model: MICAStage2CBM = instantiate(cfg_stage2).to(self.device)
        logging.info(
            f"[MICAStage2Trainer] MICAStage2CBM instantiated (with lora={instantiate_with_lora}, freeze_encoder={freeze_encoder})"
        )

        stage2_ve = getattr(model, "vision_encoder", None)
        if stage2_ve is None:
            raise AttributeError(
                "Stage-2 model must expose `vision_encoder` attribute to load Stage-1 weights."
            )

        # --- Resolve Stage-1 artifacts for THIS split ---
        stage2_cfg = self.cfg.MODEL.stage_2
        stage1_ckpt = getattr(stage2_cfg, "stage_1_ckpt", None)
        stage1_lora_dir = getattr(stage2_cfg, "stage_1_lora_dir", None)
        stage1_root_dir = getattr(stage2_cfg, "stage_1_root_dir", None)

        # If a root dir is provided, derive the per-split artifact paths automatically.
        # Tries multiple layout patterns:
        #   1. <root>/split{k}/mica_stage1/checkpoints/best_lora_adapter/  (run_train layout)
        #   2. <root>/split_{k}/checkpoints/best_lora_adapter/             (legacy layout)
        if stage1_root_dir:
            root = Path(str(stage1_root_dir))

            # Candidate paths (ordered by priority)
            candidates = [
                # Pattern 1: run_train layout (split0/mica_stage1/checkpoints/...)
                (
                    root
                    / f"split{self.split_number}"
                    / "mica_stage1"
                    / "checkpoints"
                    / "best_lora_adapter",
                    root
                    / f"split{self.split_number}"
                    / "mica_stage1"
                    / "checkpoints"
                    / "best_stage1.pt",
                ),
                # Pattern 2: legacy layout (split_0/checkpoints/...)
                (
                    root
                    / f"split_{self.split_number}"
                    / "checkpoints"
                    / "best_lora_adapter",
                    root
                    / f"split_{self.split_number}"
                    / "checkpoints"
                    / "best_stage1.pt",
                ),
            ]

            logging.info(
                "[MICAStage2Trainer] stage_1_root_dir is set; deriving Stage-1 paths:"
            )
            logging.info(f"    root: {root}")

            resolved = False
            for derived_lora_dir, derived_ckpt in candidates:
                logging.info(f"    Trying: {derived_lora_dir}")
                if derived_lora_dir.exists():
                    stage1_lora_dir = str(derived_lora_dir)
                    stage1_ckpt = None
                    resolved = True
                    logging.info(f"    Found LoRA adapter: {derived_lora_dir}")
                    break
                logging.info(f"    Trying: {derived_ckpt}")
                if derived_ckpt.exists():
                    stage1_ckpt = str(derived_ckpt)
                    stage1_lora_dir = None
                    resolved = True
                    logging.info(f"    Found checkpoint: {derived_ckpt}")
                    break

            if not resolved:
                raise FileNotFoundError(
                    f"[MICAStage2Trainer] stage_1_root_dir provided, but no Stage-1 artifacts found for "
                    f"split={self.split_number}. Checked:\n"
                    + "\n".join(f"  {lora} / {ckpt}" for lora, ckpt in candidates)
                )

        logging.info(f"[MICAStage2Trainer] Stage 1 artifact paths (resolved):")
        logging.info(f"    stage1_root_dir: {stage1_root_dir}")
        logging.info(f"    stage1_ckpt: {stage1_ckpt}")
        logging.info(f"    stage1_lora_dir: {stage1_lora_dir}")

        if stage1_ckpt and stage1_lora_dir:
            raise ValueError(
                "Provide only one of MODEL.stage_2.stage_1_ckpt or MODEL.stage_2.stage_1_lora_dir "
                "(or set MODEL.stage_2.stage_1_root_dir and let it resolve automatically)."
            )

        if stage1_lora_dir:
            # ---- LoRA case: adapter directory produced by .save_pretrained() ----
            adapter_dir = Path(stage1_lora_dir)
            if not adapter_dir.exists():
                logging.error(
                    f"[MICAStage2Trainer] ERROR: LoRA dir does not exist: {adapter_dir}"
                )
                raise FileNotFoundError(adapter_dir)

            logging.info(
                f"[MICA Stage2] Loading LoRA adapter into vision_encoder from: {adapter_dir}"
            )
            # List files in adapter dir
            files = list(adapter_dir.iterdir())
            logging.info(
                f"[MICAStage2Trainer] LoRA dir contents: {[f.name for f in files]}"
            )

            # Use VisionViT's load_lora method which correctly wraps the inner
            # vision_encoder (VisualTransformer) with the PEFT adapter
            model.vision_encoder.load_lora(adapter_dir)
            logging.info("[MICAStage2Trainer] LoRA adapter loaded successfully")

            # Configure encoder training state after LoRA is loaded
            model.configure_encoder_training(freeze_encoder=freeze_encoder)

            logging.info("=" * 50)
            return model

        elif stage1_ckpt:
            # ---- Full checkpoint case: extract vision_encoder.* keys and load ----
            ckpt_path = Path(stage1_ckpt)
            logging.info(
                f"[MICAStage2Trainer] Loading Stage-1 weights from full checkpoint: {ckpt_path}"
            )
            if not ckpt_path.exists():
                logging.error(
                    f"[MICAStage2Trainer] ERROR: Checkpoint does not exist: {ckpt_path}"
                )
                raise FileNotFoundError(ckpt_path)

            logging.info(
                f"[MICA Stage2] Loading Stage-1 vision_encoder weights from full ckpt: {ckpt_path}"
            )

            raw = torch.load(ckpt_path, map_location="cpu")
            sd = self._extract_state_dict(raw)
            logging.info(
                f"[MICAStage2Trainer] Checkpoint keys (first 10): {list(sd.keys())[:10]}"
            )

            prefix = getattr(
                self.cfg.MODEL.stage_2, "stage1_vision_prefix", "vision_encoder"
            )
            logging.info(f"[MICAStage2Trainer] Filtering keys with prefix: '{prefix}.'")
            ve_sd = self._filter_and_strip_prefix(sd, prefix + ".")
            logging.info(
                f"[MICAStage2Trainer] Filtered {len(ve_sd)} keys for vision_encoder"
            )
            logging.info(
                f"[MICAStage2Trainer] Vision encoder keys (first 10): {list(ve_sd.keys())[:10]}"
            )
            model.vision_encoder.load_state_dict(ve_sd, strict=True)
            logging.info(
                "[MICAStage2Trainer] Vision encoder weights loaded successfully"
            )

            # Configure encoder training state
            model.configure_encoder_training(freeze_encoder=freeze_encoder)

            logging.info("=" * 50)
            return model
        else:
            # If neither provided, we just return the instantiated model.
            logging.warning(
                "[MICAStage2Trainer] No Stage-1 weights provided (stage1_ckpt/stage1_lora_dir both empty). Using default init."
            )

            # Still configure encoder training state
            model.configure_encoder_training(freeze_encoder=freeze_encoder)

            logging.info("=" * 50)
            return model

    @staticmethod
    def _extract_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
        """
        Supports common checkpoint shapes:
          - raw is state_dict
          - raw has 'state_dict' key (Lightning / some wrappers)
          - raw has 'model' key etc.
        Also strips leading 'module.' from DDP.
        """
        if isinstance(raw, dict):
            if "state_dict" in raw and isinstance(raw["state_dict"], dict):
                sd = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], dict):
                sd = raw["model"]
            else:
                # assume it's already a state_dict-like dict
                # (but ensure values are tensors)
                sd = raw
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(raw)}")

        # strip possible DDP prefix
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        return sd

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

    @staticmethod
    def _filter_and_strip_prefix(
        sd: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, torch.Tensor]:
        """
        Keep only keys that start with `prefix` and strip that prefix.
        Example: prefix='vision_encoder.' turns 'vision_encoder.blocks.0...' into 'blocks.0...'
        """
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
        if not out:
            # Make failure mode explicit early
            raise KeyError(
                f"No keys found with prefix '{prefix}'. "
                f"Checkpoint keys sample: {list(sd.keys())[:20]}"
            )
        return out

    def build_optimizer_and_scheduler(
        self, model: torch.nn.Module, train_loader: DataLoader
    ):
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
        scheduler = None
        if self.cfg.TRAIN.enable_scheduler:
            scheduler = instantiate(
                self.cfg.SCHEDULER,
                optimizer,
                total_epochs=self.cfg.TRAIN.epochs,
                warmup_steps=len(train_loader),
                total_steps=len(train_loader) * self.cfg.TRAIN.epochs,
            )
        return optimizer, scheduler

    def _step(self, model: MICAStage2CBM, batch: dict, criterion: nn.Module):
        concept_logits, diag_logits, concept_labels, diag_labels = model(batch)

        # concept loss (multi-label)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_logits, concept_labels.float(), reduction="mean"
        )

        concept_weight = float(getattr(self.cfg.MODEL.stage_2, "concept_weight", 1.0))

        if self.is_binary:
            # BCEWithLogitsLoss expects [B] or [B,1] logits and float labels
            diag_loss, probs_pos, preds = self._get_binary_output(
                criterion, diag_logits, diag_labels
            )
        else:
            # e.g. CrossEntropyLoss
            diag_loss, probs_pos, preds = self._get_multiclass_output(
                criterion, diag_logits, diag_labels
            )

        total = concept_weight * concept_loss + diag_loss
        return total, concept_loss, diag_loss, preds, probs_pos, diag_labels

    def train_one_epoch(
        self, model, loader, optimizer, scheduler, scaler, criterion, epoch
    ) -> dict[str, float]:
        model.train()

        total_losses, concept_losses, diag_losses = [], [], []
        y_true, y_pred, y_prob = [], [], []

        for batch in tqdm(loader):
            optimizer.zero_grad(set_to_none=True)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                loss, c_loss, d_loss, preds, probs_pos, labels = self._step(
                    model, batch, criterion
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            total_losses.append(float(loss.item()))
            concept_losses.append(float(c_loss.item()))
            diag_losses.append(float(d_loss.item()))

            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            if probs_pos is not None:
                y_prob.append(probs_pos.detach().cpu())

        y_true_np = torch.cat(y_true).numpy()
        y_pred_np = torch.cat(y_pred).numpy()

        out = {
            "loss": float(np.mean(total_losses)),
            "concept_loss": float(np.mean(concept_losses)),
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

        total_losses, concept_losses, diag_losses = [], [], []
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                loss, c_loss, d_loss, preds, probs_pos, labels = self._step(
                    model, batch, criterion
                )

                total_losses.append(float(loss.item()))
                concept_losses.append(float(c_loss.item()))
                diag_losses.append(float(d_loss.item()))

                y_true.append(labels.detach().cpu())
                y_pred.append(preds.detach().cpu())
                if probs_pos is not None:
                    y_prob.append(probs_pos.detach().cpu())

        y_true_np = torch.cat(y_true).numpy()
        y_pred_np = torch.cat(y_pred).numpy()

        out = {
            "loss": float(np.mean(total_losses)),
            "concept_loss": float(np.mean(concept_losses)),
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

    def save_best(self, model: MICAStage2CBM) -> CheckpointPaths:
        logging.info("-" * 40)
        logging.info("[MICAStage2Trainer] Saving best model heads")
        paths = CheckpointPaths()

        # Save only trainable heads (encoder is frozen and loaded from stage1 config)
        paths.best_classifier_head = self.ckpt_dir / "best_stage2_heads.pt"

        payload = {
            "concept_head": model.concept_head.state_dict(),
            "predictor": model.predictor.state_dict(),
            "meta": {
                "is_binary": bool(self.is_binary),
                "concept_weight": float(
                    getattr(self.cfg.MODEL.stage_2, "concept_weight", 1.0)
                ),
            },
        }
        torch.save(payload, paths.best_classifier_head)
        logging.info(f"[MICA Stage2] Saved heads to: {paths.best_classifier_head}")
        logging.info(f"[MICAStage2Trainer] Payload meta: {payload['meta']}")
        if paths.best_classifier_head.exists():
            logging.info(
                f"[MICAStage2Trainer] Heads saved. Size: {paths.best_classifier_head.stat().st_size / 1e6:.2f} MB"
            )
        else:
            logging.error(f"[MICAStage2Trainer] ERROR: Heads checkpoint not created!")

        logging.info("-" * 40)
        return paths

    def load_for_test(
        self, model: MICAStage2CBM, best_paths: CheckpointPaths
    ) -> MICAStage2CBM:
        """
        Assumes `model` was created via `build_model()`, which already loads the Stage-1
        vision encoder (full ckpt or LoRA adapter). Here we only load the saved heads.
        """
        logging.info("-" * 40)
        logging.info("[MICAStage2Trainer] Loading model for test")
        logging.info(
            f"[MICAStage2Trainer] best_paths.best_classifier_head: {best_paths.best_classifier_head}"
        )
        if best_paths.best_classifier_head is None:
            logging.warning(
                "[MICA Stage2] No heads checkpoint found in best_paths.best_classifier_head; using current weights."
            )
            return model

        ckpt_path = best_paths.best_classifier_head
        if not ckpt_path.exists():
            logging.error(
                f"[MICAStage2Trainer] ERROR: Heads checkpoint does not exist: {ckpt_path}"
            )
            raise FileNotFoundError(ckpt_path)

        logging.info(f"[MICAStage2Trainer] Loading heads from: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")

        if (
            not isinstance(payload, dict)
            or "concept_head" not in payload
            or "predictor" not in payload
        ):
            raise ValueError(
                f"[MICA Stage2] Invalid heads checkpoint format at {ckpt_path}. "
                "Expected dict with keys: 'concept_head', 'predictor'."
            )

        logging.info(f"[MICAStage2Trainer] Payload keys: {list(payload.keys())}")
        if "meta" in payload:
            logging.info(f"[MICAStage2Trainer] Payload meta: {payload['meta']}")

        model.concept_head.load_state_dict(payload["concept_head"], strict=True)
        model.predictor.load_state_dict(payload["predictor"], strict=True)

        logging.info(f"[MICA Stage2] Loaded heads from: {ckpt_path}")
        logging.info("-" * 40)
        return model

    def test_from_ckpt(
        self, best_paths: CheckpointPaths, test_loader: DataLoader, criterion
    ) -> Dict[str, Any]:
        model = self.build_model().to(
            self.device
        )  # loads stage1 encoder (full or LoRA)
        model = self.load_for_test(model, best_paths)  # loads heads only
        model.eval()

        losses = []
        y_true, y_pred, y_prob = [], [], []
        meta_rows = []

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device, non_blocking=True)

                loss, _, _, preds, probs_pos, labels = self._step(
                    model, batch, criterion
                )

                losses.append(float(loss.item()))
                y_true.append(labels.detach().cpu())
                y_pred.append(preds.detach().cpu())
                y_prob.append(probs_pos.detach().cpu())

                # Collect metadata for subgroup metrics
                meta_rows = self._collect_metadata(batch, meta_rows)

        y_true_np = torch.cat(y_true).numpy()
        y_pred_np = torch.cat(y_pred).numpy()

        if self.is_binary:
            y_prob = torch.cat(y_prob).numpy().reshape(-1)
        else:
            y_prob = torch.cat(y_prob).numpy()

        # --- Compute metrics ---
        metrics = self._compute_shared_metrics(losses, y_true_np, y_pred_np)

        if self.is_binary:
            metrics = self._compute_binary_metrics(
                metrics, y_true_np, y_pred_np, y_prob
            )
        else:
            # Add macro F1 for multiclass
            metrics = self._compute_f1_score_multiclass(metrics, y_true_np, y_pred_np)

        # --- Confusion matrix ---
        cm = self._get_confusion_matrix(y_true_np, y_pred_np, y_prob)

        # --- Subgroup metrics ---
        meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()
        sub_df = pd.DataFrame()  # Initialize to empty DataFrame

        if not meta_df.empty:
            if self.is_binary:
                sub_df = subgroup_metrics_binary_concepts(
                    meta_df=meta_df,
                    y_true=y_true_np.astype(int),
                    y_pred=y_pred_np.astype(int),
                    y_prob_pos=y_prob.astype(float),
                )
            else:
                sub_df = subgroup_metrics_multiclass_concepts(
                    meta_df=meta_df,
                    y_true_0based=y_true_np.astype(int),
                    y_pred_0based=y_pred_np.astype(int),
                )

        # --- Save results ---
        self._save_metrics(metrics, meta_df, y_true_np, y_pred_np, y_prob, cm, sub_df)

        return {
            "metrics": metrics,
            "y_true": y_true_np,
            "y_pred": y_pred_np,
            "y_prob": y_prob,
            "subgroup_metrics": sub_df,
        }

    def fit(self) -> CheckpointPaths:
        set_seed(self.seed)
        logging.info("=" * 60)
        logging.info("[MICAStage2Trainer] Starting Stage 2 training")
        logging.info(f"[MICAStage2Trainer] exp_dir: {self.exp_dir}")
        logging.info(f"[MICAStage2Trainer] split: {self.split_number}")
        logging.info(f"[MICAStage2Trainer] epochs: {self.cfg.TRAIN.epochs}")
        logging.info(f"[MICAStage2Trainer] is_binary: {self.is_binary}")
        logging.info(
            f"[MICAStage2Trainer] concept_weight: {getattr(self.cfg.MODEL.stage_2, 'concept_weight', 1.0)}"
        )
        logging.info("=" * 60)
        tokenizer = instantiate(self.cfg.MODEL.tokenizer)
        max_len = int(self.cfg.DATASET.text.word_num)
        collate = lambda b: mica_collate_hf_tokenizer(b, tokenizer, max_len)
        train_loader, val_loader = self.build_loaders(collate_fn=collate)
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

            sel = self.select_metric_name()
            cur = float(va.get(sel))
            self._log_fit_history(history, epoch, tr, va)

            best_paths, best = self.save_and_log_best(sel, cur, best, best_paths, model)

        self.finilize_fit(history)
        logging.info("=" * 60)
        logging.info("[MICAStage2Trainer] Stage 2 training complete")
        logging.info(
            f"[MICAStage2Trainer] Best paths: classifier_head={best_paths.best_classifier_head}"
        )
        logging.info("=" * 60)

        return best_paths
