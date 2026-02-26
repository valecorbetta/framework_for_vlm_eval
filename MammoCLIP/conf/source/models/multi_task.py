import logging
from typing import Optional

import torch
import torch.nn as nn

from source.models.vision_efficientnet import VisionEfficientNet
from source.models.linear_classifier import LinearClassifier


class MammoCLIPMultiTask(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEfficientNet,
        concept_head: LinearClassifier,
        diag_head: LinearClassifier,
        ckpt_clip: Optional[str] = None,
        image_global_feat_dim: int = 2048,
        embed_dim: int = 512,
        freeze_projection: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.concept_head = concept_head
        self.diag_head = diag_head
        # Projection layer: only used when a MammoCLIP checkpoint is provided
        if ckpt_clip is not None:
            self.image_projection_global = nn.Linear(image_global_feat_dim, embed_dim)
            self._load_image_projection_from_checkpoint(ckpt_clip)
            if freeze_projection:
                for p in self.image_projection_global.parameters():
                    p.requires_grad = False
                logging.info("[MammoClassifier] image_projection_global is FROZEN")
            else:
                logging.info("[MammoClassifier] image_projection_global is TRAINABLE")
        else:
            self.image_projection_global = None
            logging.info(
                "[MammoClassifier] No projection head — raw encoder features fed to classifier."
            )

    def _load_image_projection_from_checkpoint(self, ckpt_path: str) -> None:
        """
        Load MammoCLIP image projection weights if shapes match.
        Expects keys:
          image_projection.projection.weight  (out, in)
          image_projection.projection.bias    (out,)
        """
        logging.info(f"[MammoClassifier] Loading image projection from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        w_key = "image_projection.projection.weight"
        b_key = "image_projection.projection.bias"
        if w_key not in ckpt or b_key not in ckpt:
            logging.warning(
                "[MammoClassifier] No MammoCLIP image projection keys found; "
                "keeping random initialisation."
            )
            return

        w = ckpt[w_key]  # [out, in]
        b = ckpt[b_key]  # [out]
        out_dim, in_dim = w.shape

        if (
            in_dim != self.image_projection_global.in_features
            or out_dim != self.image_projection_global.out_features
        ):
            logging.warning(
                f"[MammoClassifier] Skipping projection load: ckpt shape ({out_dim},{in_dim}) "
                f"!= expected ({self.image_projection_global.out_features},"
                f"{self.image_projection_global.in_features})"
            )
            return

        self.image_projection_global.weight.data.copy_(w)
        self.image_projection_global.bias.data.copy_(b)
        logging.info(f"[MammoClassifier] Loaded image projection weights: {w.shape}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(x, get_local=False)
        if self.image_projection_global is not None:
            print("THE PROJECTION IS NOT NONE")
            feats = self.image_projection_global(feats)
        return feats

    def forward(self, batch: dict):
        x = batch["x"]
        feats = self._encode(x)

        concept_logits = self.concept_head(feats)  # [B, Nc]
        diag_logits = self.diag_head(feats)  # [B] or [B,1] if binary; else [B,C]

        return concept_logits, diag_logits
