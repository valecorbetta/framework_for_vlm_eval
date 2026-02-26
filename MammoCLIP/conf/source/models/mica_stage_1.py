import logging
import pickle
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

from source.models.vision_efficientnet import VisionEfficientNet
from source.models.text_encoder import TextEncoderMICA
from source.utils.losses import attention_fn, local_loss, global_loss
from source.utils.misc import ConceptBank, update_state_dict


class MICAStage1(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEfficientNet,
        text_encoder: TextEncoderMICA,
        cfg: DictConfig,
        *,
        path_to_cav_file: Path,
        device: torch.device,
        ckpt_clip: Optional[str] = None,
        image_feat_dim: int = 120,  # 176,
        image_global_feat_dim: int = 2048,
        text_feat_dim: int = 768,
        embed_dim: int = 512,
    ):
        super(MICAStage1, self).__init__()
        logging.info("=" * 50)
        logging.info("[MICAStage1] Initializing model")
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim

        # Projection layers to map encoders to shared embedding space
        # Image local: [B, C, H, W] -> [B, embed_dim, H, W] using 1x1 conv
        self.image_projection = nn.Conv2d(image_feat_dim, embed_dim, kernel_size=1)
        # For global image features: [B, C_global] -> [B, embed_dim]
        self.image_projection_global = nn.Linear(image_global_feat_dim, embed_dim)
        # Text: [B, text_feat_dim, T] -> [B, embed_dim, T]
        self.text_projection = nn.Linear(text_feat_dim, embed_dim)

        # Load projection weights from checkpoint if provided
        if ckpt_clip is not None:
            self._load_projections_from_checkpoint(ckpt_clip)

        logging.info(f"[MICAStage1] Projection dimensions:")
        logging.info(f"    image local: {image_feat_dim} -> {embed_dim}")
        logging.info(f"    image global: {image_global_feat_dim} -> {embed_dim}")
        logging.info(f"    text: {text_feat_dim} -> {embed_dim}")

        self.temp1 = cfg.MODEL.stage_1.losses.temp1
        self.temp2 = cfg.MODEL.stage_1.losses.temp2
        self.temp3 = cfg.MODEL.stage_1.losses.temp3
        self.local_loss = local_loss
        self.global_loss = global_loss
        # self.concept_loss = nn.CrossEntropyLoss()
        self.concept_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.local_loss_weight = cfg.MODEL.stage_1.losses.local_loss_weight
        self.global_loss_weight = cfg.MODEL.stage_1.losses.global_loss_weight
        self.concept_loss_weight = cfg.MODEL.stage_1.losses.concept_loss_weight
        logging.info("[MICAStage1] Loss parameters:")
        logging.info(f"    temp1={self.temp1}")
        logging.info(f"    temp2={self.temp2}")
        logging.info(f"    temp3={self.temp3}")
        logging.info(f"    local_loss_weight={self.local_loss_weight}")
        logging.info(f"    global_loss_weight={self.global_loss_weight}")
        logging.info(f"    concept_loss_weight={self.concept_loss_weight}")
        all_concepts = pickle.load(open(path_to_cav_file, "rb"))
        all_concept_names = list(all_concepts.keys())
        logging.info(
            f"Bank path: {path_to_cav_file}. {len(all_concept_names)} concepts will be used."
        )
        self.concept_bank = ConceptBank(all_concepts, device)  # set device
        self.cavs = self.concept_bank.concept_info.vectors
        self.intercepts = self.concept_bank.concept_info.intercepts
        self.norms = self.concept_bank.concept_info.norms
        # CAV dimension must match the concept encoder output
        cav_dim = self.cavs.shape[1]
        self.concept_encoder = nn.Sequential(
            nn.Linear(embed_dim * cfg.DATASET.text.word_num, cav_dim),
            nn.Tanh(),
        )
        logging.info(
            f"[MICAStage1] Concept encoder: {embed_dim * cfg.DATASET.text.word_num} -> {cav_dim} (CAV dim)"
        )

    def _load_projections_from_checkpoint(self, ckpt_path: str) -> None:
        """
        Load projection layer weights from MammoCLIP checkpoint.

        The checkpoint has:
            image_projection.projection.weight/bias
            text_projection.projection.weight/bias

        Note: Only loads projections if dimensions match. For EfficientNet local features,
        the checkpoint's image projection (trained for global features) won't match,
        so the local projection remains randomly initialized.
        """
        logging.info(f"[MICAStage1] Loading projections from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle wrapped checkpoints
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # Load image projection (only if dimensions match)
        if "image_projection.projection.weight" in ckpt:
            img_weight = ckpt["image_projection.projection.weight"]  # [out, in]
            img_bias = ckpt["image_projection.projection.bias"]  # [out]
            ckpt_out_dim, ckpt_in_dim = img_weight.shape

            # Check if dimensions match for local projection (Conv2d)
            local_in_dim = self.image_projection.in_channels
            local_out_dim = self.image_projection.out_channels

            if ckpt_in_dim == local_in_dim and ckpt_out_dim == local_out_dim:
                # Convert Linear weight to Conv2d weight: [out, in] -> [out, in, 1, 1]
                self.image_projection.weight.data = img_weight.unsqueeze(-1).unsqueeze(
                    -1
                )
                self.image_projection.bias.data = img_bias
                logging.info(
                    f"[MICAStage1] Loaded local image projection: {img_weight.shape}"
                )
            else:
                logging.info(
                    f"[MICAStage1] Skipping local image projection load: "
                    f"checkpoint shape {img_weight.shape} != expected ({local_out_dim}, {local_in_dim})"
                )

            # Check if dimensions match for global projection (Linear)
            global_in_dim = self.image_projection_global.in_features
            global_out_dim = self.image_projection_global.out_features

            if ckpt_in_dim == global_in_dim and ckpt_out_dim == global_out_dim:
                self.image_projection_global.weight.data = img_weight
                self.image_projection_global.bias.data = img_bias
                logging.info(
                    f"[MICAStage1] Loaded global image projection: {img_weight.shape}"
                )
            else:
                logging.info(
                    f"[MICAStage1] Skipping global image projection load: "
                    f"checkpoint shape {img_weight.shape} != expected ({global_out_dim}, {global_in_dim})"
                )

        # Load text projection (only if dimensions match)
        if "text_projection.projection.weight" in ckpt:
            text_weight = ckpt["text_projection.projection.weight"]  # [out, in]
            text_bias = ckpt["text_projection.projection.bias"]  # [out]
            ckpt_out_dim, ckpt_in_dim = text_weight.shape

            text_in_dim = self.text_projection.in_features
            text_out_dim = self.text_projection.out_features

            if ckpt_in_dim == text_in_dim and ckpt_out_dim == text_out_dim:
                self.text_projection.weight.data = text_weight
                self.text_projection.bias.data = text_bias
                logging.info(
                    f"[MICAStage1] Loaded text projection: {text_weight.shape}"
                )
            else:
                logging.info(
                    f"[MICAStage1] Skipping text projection load: "
                    f"checkpoint shape {text_weight.shape} != expected ({text_out_dim}, {text_in_dim})"
                )

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        # Apply text projection: [B, D_bert, T] -> [B, embed_dim, T]
        B, D, T = text_emb_l.shape
        # Reshape for linear: [B, D, T] -> [B, T, D] -> [B*T, D]
        text_emb_l_flat = text_emb_l.permute(0, 2, 1).reshape(B * T, D)
        text_emb_l_proj = self.text_projection(text_emb_l_flat)
        # Reshape back: [B*T, embed_dim] -> [B, T, embed_dim] -> [B, embed_dim, T]
        text_emb_l_proj = text_emb_l_proj.reshape(B, T, self.embed_dim).permute(0, 2, 1)

        # Project global text: [B, D_bert] -> [B, embed_dim]
        text_emb_g_proj = self.text_projection(text_emb_g)

        return text_emb_l_proj, text_emb_g_proj, sents

    def image_encoder_forward(self, imgs):
        img_emb_g, img_emb_l = self.vision_encoder(imgs, get_local=True)
        # Apply image projection
        # Local: [B, C, H, W] -> [B, embed_dim, H, W]
        img_emb_l_proj = self.image_projection(img_emb_l)
        # Global: [B, C] -> [B, embed_dim]
        img_emb_g_proj = self.image_projection_global(img_emb_g)

        return img_emb_g_proj, img_emb_l_proj

    def concept_encoder_forward(self, img_emb_l, text_emb_l):
        weighted_represent = self.get_weighted_representation(img_emb_l, text_emb_l)
        predict_concepts = self.concept_encoder(weighted_represent)
        predict_concepts = (
            torch.matmul(self.cavs, predict_concepts.T) + self.intercepts
        ) / self.norms
        return predict_concepts.T

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, _ = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def _calc_concept_loss(self, predict_concepts, concept_labels):
        return self.concept_loss(predict_concepts, concept_labels.float())

    def calc_loss(
        self,
        img_emb_l,
        img_emb_g,
        text_emb_l,
        text_emb_g,
        sents,
        predict_concepts,
        concept_labels,
    ):

        l_loss0, l_loss1 = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        concept_loss = self._calc_concept_loss(predict_concepts, concept_labels)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight
        loss += concept_loss * self.concept_loss_weight

        return loss

    def get_weighted_representation(self, img_emb_l, text_emb_l):
        batch_size = img_emb_l.shape[0]
        weighted_contexts = []

        for i in range(text_emb_l.shape[0]):
            word = text_emb_l[i].unsqueeze(0).contiguous()
            word = word.repeat(batch_size, 1, 1)
            context = img_emb_l

            weiContext, attn = attention_fn(
                word, context, temp1=4.0
            )  # (batch_size, 768, words_num)
            weighted_contexts.append(weiContext)

        # average the weighted contexts and get the weighted representation
        weighted_represent = torch.stack(weighted_contexts, dim=0).mean(dim=0)
        return weighted_represent.view(
            batch_size, text_emb_l.shape[1] * text_emb_l.shape[2]
        )

    def forward(self, x):

        # img encoder branch
        img_emb_g, img_emb_l = self.image_encoder_forward(x["x"])

        # text encoder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        predict_concepts = self.concept_encoder_forward(img_emb_l, text_emb_l)
        concept_labels = x["concept_labels"]

        return (
            img_emb_l,
            img_emb_g,
            text_emb_l,
            text_emb_g,
            sents,
            predict_concepts,
            concept_labels,
        )
