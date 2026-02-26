import torch
import torch.nn as nn

from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.model.linear_classifier import LinearClassifier


class RetCLIPMultiTask(nn.Module):

    def __init__(
        self,
        vision_encoder: VisionViT,
        concept_head: LinearClassifier,
        diag_head: LinearClassifier,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.concept_head = concept_head
        self.diag_head = diag_head

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(x, get_local=False)

    def forward(self, batch: dict):
        x = batch["x"]
        feats = self._encode(x)

        concept_logits = self.concept_head(feats)  # [B, Nc]
        diag_logits = self.diag_head(feats)  # [B] or [B,1] if binary; else [B,C]

        return concept_logits, diag_logits
