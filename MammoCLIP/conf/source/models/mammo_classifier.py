import logging
from typing import Optional
from torch import nn
import torch

from source.models.vision_efficientnet import VisionEfficientNet
from source.models.linear_classifier import LinearClassifier


class MammoClassifier(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEfficientNet,
        classifier: LinearClassifier,
        ckpt_clip: Optional[str] = None,
        image_global_feat_dim: int = 2048,
        embed_dim: int = 512,
        freeze_projection: bool = True,
    ):
        """
        Args:
            ckpt_clip: Path to MammoCLIP checkpoint. If provided, the image projection
                       layer is added and its weights are loaded from the checkpoint.
                       If None (e.g. ImageNet baseline), no projection is applied and
                       the raw encoder features go directly into the classifier.
            image_global_feat_dim: EfficientNet output dim (used only when ckpt_clip set).
            embed_dim: CLIP joint embedding dim (used only when ckpt_clip set).
            freeze_projection: Whether to freeze the projection layer (used only when
                               ckpt_clip set).
        """
        super(MammoClassifier, self).__init__()
        self.vision_encoder = vision_encoder

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

        self.classifier = classifier

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

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(image)  # [B, image_global_feat_dim]
        if self.image_projection_global is not None:
            feats = self.image_projection_global(feats)  # [B, embed_dim]
        return feats

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(image)
        logits = self.classifier(image_features)
        return logits
