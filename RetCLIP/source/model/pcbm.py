from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pickle
import torch
import torch.nn as nn

from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.model.linear_classifier import LinearClassifier
from RetCLIP.source.utils.misc import ConceptBank


class PostHocCBM(nn.Module):

    def __init__(
        self,
        vision_encoder: VisionViT,
        predictor: LinearClassifier,  # (num_concepts, num_diag_classes)
        feat_dim: int,
        device: torch.device,
        cav_pkl: Path,
        use_normalized_margin: bool = True,
    ):
        super().__init__()
        logging.info("=" * 50)
        logging.info("[PostHocCBM] Initializing model")
        self.vision_encoder = vision_encoder
        self.use_normalized_margin = use_normalized_margin
        all_concepts = pickle.load(open(cav_pkl, "rb"))
        self.concept_names = list(all_concepts.keys())
        self.num_concepts = len(self.concept_names)
        logging.info(
            f"Bank path: {cav_pkl}. {len(self.concept_names)} concepts will be used."
        )
        self.concept_bank = ConceptBank(all_concepts, device)
        self.cav_tensors = self.concept_bank.concept_info.vectors
        self.intercepts = self.concept_bank.concept_info.intercepts
        self.norms = self.concept_bank.concept_info.norms
        self.predictor = predictor

        # sanity check
        if int(feat_dim) != int(self.cav_tensors.shape[1]):
            raise ValueError(
                f"feat_dim={feat_dim} but CAV dimension={self.cav_tensors.shape[1]}. "
                f"Ensure CAVs were trained on the same encoder embedding dimension."
            )

        # Check vision encoder training status
        total_params = sum(p.numel() for p in vision_encoder.parameters())
        trainable_params = sum(
            p.numel() for p in vision_encoder.parameters() if p.requires_grad
        )
        lora_params = sum(
            p.numel()
            for n, p in vision_encoder.named_parameters()
            if "lora" in n.lower()
        )
        frozen = trainable_params == 0
        logging.info(
            f"[PostHocCBM] Vision encoder: {total_params:,} total params, {trainable_params:,} trainable, {lora_params:,} LoRA params"
        )
        if frozen:
            logging.info("[PostHocCBM] Vision encoder is FROZEN (no trainable params)")
        elif lora_params > 0:
            logging.info(
                f"[PostHocCBM] Vision encoder is using LoRA ({lora_params:,} LoRA params trainable)"
            )
        else:
            logging.info(
                "[PostHocCBM] Vision encoder is FULLY TRAINABLE (no LoRA detected)"
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(x, get_local=False)  # [B, D]

    def concept_scores(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B,D]
        returns: [B,Nc] concept logits/scores
        """
        # [B,Nc] = [B,D] @ [D,Nc]
        raw = feats @ self.cav_tensors.t() + self.intercepts.view(1, -1)
        if self.use_normalized_margin:
            raw = raw / self.norms.view(1, -1)

        return raw

    def forward(self, batch: dict):
        x = batch["x"]
        feats = self._encode(x)  # [B,D]
        c = self.concept_scores(feats)  # [B,Nc]
        diag_logits = self.predictor(c)  # [B,1] or [B,C]
        return c, diag_logits
