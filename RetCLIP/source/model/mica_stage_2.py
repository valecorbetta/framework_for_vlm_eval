import logging
import torch
import torch.nn as nn

from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.model.linear_classifier import LinearClassifier


class MICAStage2CBM(nn.Module):
    """
    Replicates MICA Stage-2 CBM graph:

      x -> vision_encoder -> feats -> concept_head -> concept_logits
      concept_logits -> predictor -> diag_logits
    """

    def __init__(
        self,
        vision_encoder: VisionViT,
        concept_head: LinearClassifier,
        predictor: LinearClassifier,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.freeze_encoder = freeze_encoder
        # Log initial state (before LoRA loading)
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
            f"[MICAStage2CBM] Vision encoder (initial): {total_params:,} total, {trainable_params:,} trainable, {lora_params:,} LoRA"
        )

        self.concept_head = concept_head
        self.predictor = predictor

    def configure_encoder_training(self, freeze_encoder: bool = None):
        """
        Configure encoder trainability. Call after LoRA adapter is loaded.

        Args:
            freeze_encoder: If True, freeze all encoder params. If False, unfreeze LoRA params.
                        If None, uses the value from __init__.
        """
        if freeze_encoder is None:
            freeze_encoder = self.freeze_encoder

        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            logging.info("[MICAStage2CBM] Froze all encoder parameters")
        else:
            # Unfreeze LoRA params only (base model stays frozen)
            lora_count = 0
            for name, p in self.vision_encoder.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
                    lora_count += 1
            logging.info(f"[MICAStage2CBM] Unfroze {lora_count} LoRA parameter tensors")

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
            f"[MICAStage2CBM] Final state: {total:,} total, {trainable:,} trainable, {lora_trainable:,} LoRA trainable"
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(x, get_local=False)

    def forward(self, batch: dict):
        x = batch["x"]
        concept_labels = batch["concept_labels"]
        diag_labels = batch["y"]

        feats = self._encode(x)  # [B, D]
        concept_logits = self.concept_head(feats)  # [B, Nc]
        diag_logits = self.predictor(concept_logits)  # [B, C]

        return concept_logits, diag_logits, concept_labels, diag_labels
