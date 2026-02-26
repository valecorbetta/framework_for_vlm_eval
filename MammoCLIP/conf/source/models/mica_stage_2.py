import logging
from typing import Optional
import torch
import torch.nn as nn

from source.models.vision_efficientnet import VisionEfficientNet
from source.models.linear_classifier import LinearClassifier


class MICAStage2CBM(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEfficientNet,
        concept_head: LinearClassifier,
        predictor: LinearClassifier,
        freeze_encoder: bool = True,
        ckpt_clip: Optional[str] = None,
        image_global_feat_dim: int = 2048,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.freeze_encoder = freeze_encoder

        # Projection layer: only used when a MammoCLIP checkpoint is provided
        if ckpt_clip is not None:
            self.image_projection_global = nn.Linear(image_global_feat_dim, embed_dim)
            self._load_image_projection_from_checkpoint(ckpt_clip)
            # Always frozen in MICA — part of the pretrained encoder pipeline
            for p in self.image_projection_global.parameters():
                p.requires_grad = False
            logging.info("[MICAStage2CBM] image_projection_global is FROZEN")
        else:
            self.image_projection_global = None
            logging.info(
                "[MICAStage2CBM] No projection head — raw encoder features fed to concept head."
            )

        # Log initial encoder state (before LoRA loading)
        total_params = sum(p.numel() for p in vision_encoder.parameters())
        trainable_params = sum(
            p.numel() for p in vision_encoder.parameters() if p.requires_grad
        )
        lora_params = sum(
            p.numel()
            for n, p in vision_encoder.named_parameters()
            if "lora" in n.lower()
        )
        logging.info(
            f"[MICAStage2CBM] Vision encoder (initial): {total_params:,} total, "
            f"{trainable_params:,} trainable, {lora_params:,} LoRA"
        )

        self.concept_head = concept_head
        self.predictor = predictor

    def _load_image_projection_from_checkpoint(self, ckpt_path: str) -> None:
        """
        Load MammoCLIP image projection if shapes match.
        Expects keys:
          image_projection.projection.weight  (out, in)
          image_projection.projection.bias    (out,)
        """
        logging.info(f"[MICAStage2CBM] Loading image projection from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        w_key = "image_projection.projection.weight"
        b_key = "image_projection.projection.bias"
        if w_key not in ckpt or b_key not in ckpt:
            logging.warning(
                "[MICAStage2CBM] No MammoCLIP image projection keys found; "
                "keeping random initialisation."
            )
            return

        w = ckpt[w_key]
        b = ckpt[b_key]
        out_dim, in_dim = w.shape

        if (
            in_dim != self.image_projection_global.in_features
            or out_dim != self.image_projection_global.out_features
        ):
            logging.warning(
                f"[MICAStage2CBM] Skipping projection load: ckpt ({out_dim},{in_dim}) "
                f"!= expected ({self.image_projection_global.out_features},"
                f"{self.image_projection_global.in_features})"
            )
            return

        self.image_projection_global.weight.data.copy_(w)
        self.image_projection_global.bias.data.copy_(b)
        logging.info(f"[MICAStage2CBM] Loaded image projection weights: {w.shape}")

    def configure_encoder_training(self, freeze_encoder: bool = None):
        """
        Configure encoder trainability. Call after LoRA adapter is loaded.

        Args:
            freeze_encoder: If True, freeze all encoder params. If False, unfreeze LoRA params.
                            If None, uses the value from __init__.
        """
        if freeze_encoder is None:
            freeze_encoder = self.freeze_encoder

        # Always keep projection frozen — it is part of the pretrained encoder pipeline
        if self.image_projection_global is not None:
            for p in self.image_projection_global.parameters():
                p.requires_grad = False

        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            logging.info(
                "[MICAStage2CBM] Froze all encoder parameters (incl. image_projection_global)"
            )
        else:
            # Unfreeze LoRA params only (base model stays frozen)
            lora_count = 0
            for name, p in self.vision_encoder.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
                    lora_count += 1
            logging.info(
                f"[MICAStage2CBM] Unfroze {lora_count} LoRA parameter tensors "
                f"(image_projection_global remains frozen)"
            )

        # Log final state
        total = sum(p.numel() for p in self.vision_encoder.parameters())
        trainable = sum(
            p.numel() for p in self.vision_encoder.parameters() if p.requires_grad
        )
        lora_trainable = sum(
            p.numel()
            for n, p in self.vision_encoder.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        )
        logging.info(
            f"[MICAStage2CBM] Final state: {total:,} total, {trainable:,} trainable, "
            f"{lora_trainable:,} LoRA trainable"
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(x, get_local=False)  # [B, image_global_feat_dim]
        if self.image_projection_global is not None:
            feats = self.image_projection_global(feats)  # [B, embed_dim]
        return feats

    def forward(self, batch: dict):
        x = batch["x"]
        concept_labels = batch["concept_labels"]
        diag_labels = batch["y"]

        feats = self._encode(x)  # [B, D]
        concept_logits = self.concept_head(feats)  # [B, Nc]
        diag_logits = self.predictor(concept_logits)  # [B, C]

        return concept_logits, diag_logits, concept_labels, diag_labels
