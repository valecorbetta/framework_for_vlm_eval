import logging
import pickle
from typing import List
from omegaconf import DictConfig, ListConfig
from torch import nn
import torch
from peft import LoraConfig, get_peft_model
from pathlib import Path

from RetCLIP.source.model.vision_vit import VisionViT
from RetCLIP.source.model.text_bert import TextBert
from RetCLIP.source.utils.losses import attention_fn, local_loss, global_loss
from RetCLIP.source.utils.misc import ConceptBank


class MICAStage1(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionViT,
        text_encoder: TextBert,
        cfg: DictConfig,
        *,
        path_to_cav_file: Path,
        device: torch.device,
    ):
        super(MICAStage1, self).__init__()
        logging.info("=" * 50)
        logging.info("[MICAStage1] Initializing model")
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
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
        # ckpt_clip = torch.load(cfg.MODEL.path_to_ckpt)
        embedding_dim = cfg.MODEL.text_encoder.embed_dim
        self.concept_encoder = nn.Sequential(
            nn.Linear(embedding_dim * cfg.DATASET.text.word_num, 512),
            nn.Tanh(),
        )
        logging.info(
            f"[MICAStage1] Concept encoder input dim: {embedding_dim} * {cfg.DATASET.text.word_num} = {embedding_dim * cfg.DATASET.text.word_num}"
        )

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_emb_g, img_emb_l = self.vision_encoder(imgs, get_local=True)
        return img_emb_g, img_emb_l

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
