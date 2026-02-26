from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pickle
import torch
import torch.nn as nn

from source.models.vision_efficientnet import VisionEfficientNet
from source.models.linear_classifier import LinearClassifier
from source.utils.misc import ConceptBank


class PostHocCBM(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEfficientNet,
        predictor: LinearClassifier,  # (num_concepts, num_diag_classes)
        device: torch.device,
        cav_pkl: Path,
        use_normalized_margin: bool = True,
        ckpt_clip: Optional[str] = None,
        image_global_feat_dim: int = 2048,
        embed_dim: int = 512,
        freeze_projection: bool = True,
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
        effective_feat_dim = (
            embed_dim if ckpt_clip is not None else image_global_feat_dim
        )
        if int(effective_feat_dim) != int(self.cav_tensors.shape[1]):
            raise ValueError(
                f"effective_feat_dim={effective_feat_dim} but CAV dimension={self.cav_tensors.shape[1]}. "
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
        feats = self.vision_encoder(x, get_local=False)  # [B, D]
        if self.image_projection_global is not None:
            feats = self.image_projection_global(feats)  # [B, embed_dim]
        return feats

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
