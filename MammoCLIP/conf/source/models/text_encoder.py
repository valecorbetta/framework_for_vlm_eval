from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import ListConfig
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

from source.utils.misc import freeze_params, update_state_dict


class TextEncoderMICA(nn.Module):
    """
    MICA-style BERT text encoder with:
    - Last N layers aggregation (sum/mean)
    - Token aggregation (wordpiece -> word)
    - Local (word-level) and global (sentence-level) embeddings
    - Optional projection layers
    - Optional L2 normalization
    - Optional LoRA fine-tuning
    """

    def __init__(
        self,
        bert_type: str = "emilyalsentzer/Bio_ClinicalBERT",
        last_n_layers: int = 4,
        aggregate_method: str = "sum",
        norm: bool = False,
        embedding_dim: int = 768,
        agg_tokens: bool = True,
        ckpt_clip: Optional[str] = None,
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha_mult: int = 2,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super(TextEncoderMICA, self).__init__()

        self.bert_type = bert_type
        self.last_n_layers = last_n_layers
        self.aggregate_method = aggregate_method
        self.norm = norm
        self.embedding_dim = embedding_dim
        self.agg_tokens = agg_tokens

        # Load BERT model with hidden states output enabled
        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        # Load tokenizer and create inverse vocab mapping
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # Load weights from MammoCLIP checkpoint if provided
        if ckpt_clip is not None:
            self._load_from_checkpoint(ckpt_clip)

        # Get hidden size from model config
        self.text_hidden_size = self.model.config.hidden_size

        # Projection layers (only if hidden size != embedding dim)
        if self.text_hidden_size == self.embedding_dim:
            self.emb_global = None
            self.emb_local = None
        else:
            self.emb_global = nn.Linear(self.text_hidden_size, self.embedding_dim)
            self.emb_local = nn.Linear(self.text_hidden_size, self.embedding_dim)

        # LoRA or freeze
        if lora:
            if lora_target_modules is None:
                lora_target_modules = ["query", "key", "value"]
            elif isinstance(lora_target_modules, ListConfig):
                lora_target_modules = list(lora_target_modules)

            lora_alpha = lora_alpha_mult * lora_r
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            freeze_params(self.model)

    def _load_from_checkpoint(self, ckpt_path: str) -> None:
        """
        Load text encoder weights from MammoCLIP checkpoint.

        The checkpoint structure has keys like:
            text_encoder.text_encoder.embeddings.word_embeddings.weight
            text_encoder.text_encoder.encoder.layer.0.attention.self.query.weight
            ...

        These map to self.model (HuggingFace BERT):
            embeddings.word_embeddings.weight
            encoder.layer.0.attention.self.query.weight
            ...
        """
        print(f"[TextEncoderMICA] Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle wrapped checkpoints
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # Extract text encoder weights: text_encoder.text_encoder.X -> X
        text_encoder_state = update_state_dict(ckpt, "text_encoder.text_encoder.")

        # Load into self.model
        missing, unexpected = self.model.load_state_dict(text_encoder_state, strict=False)
        if missing:
            print(f"[TextEncoderMICA] Missing keys: {missing}")
        if unexpected:
            print(f"[TextEncoderMICA] Unexpected keys: {unexpected}")
        print(f"[TextEncoderMICA] Loaded {len(text_encoder_state)} parameters from checkpoint")

    def aggregate_tokens(self, embeddings, caption_ids):
        """
        Aggregate wordpiece tokens into words.

        Args:
            embeddings: [B, num_layers, T, H]
            caption_ids: [B, T]

        Returns:
            agg_embs_batch: [B, num_layers, T, H]
            sentences: List[List[str]] with length B
        """
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)  # [B, T, num_layers, H]
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    if len(token_bank) > 0:
                        new_emb = torch.stack(token_bank).sum(dim=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank).sum(dim=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])

            # Handle case where sequence ended without [SEP]
            if len(token_bank) > 0 and (len(words) == 0 or words[-1] != "[SEP]"):
                new_emb = torch.stack(token_bank).sum(dim=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim, device=embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)  # [B, num_layers, T, H]
        return agg_embs_batch, sentences

    def has_lora(self) -> bool:
        return isinstance(self.model, PeftModel)

    def save_lora(self, path: str | Path):
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("No LoRA adapter to save.")
        self.model.save_pretrained(path)

    def load_lora(self, path: str | Path):
        self.model = PeftModel.from_pretrained(self.model, path)

    def forward(
        self,
        ids: torch.Tensor,  # [B, T]
        attn_mask: torch.Tensor,  # [B, T]
        token_type_ids: torch.Tensor,  # [B, T]
    ):
        """
        Returns:
            text_emb_l: [B, embedding_dim, T] (local / word-level)
            text_emb_g: [B, embedding_dim] (global / sentence-level)
            sents: List[List[str]]
        """
        outputs = self.model(ids, attn_mask, token_type_ids)

        # Aggregate intermediate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]  # tuple of hidden states
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # [N, B, T, H]
            embeddings = embeddings.permute(1, 0, 2, 3)  # [B, N, T, H]

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            # Sentence embeddings as mean over tokens: [B, N, H]
            sent_embeddings = embeddings.mean(dim=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(dim=1)  # [B, T, H]
                sent_embeddings = sent_embeddings.sum(dim=1)  # [B, H]
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(dim=1)  # [B, T, H]
                sent_embeddings = sent_embeddings.mean(dim=1)  # [B, H]
            else:
                raise ValueError(
                    f"Aggregation method '{self.aggregate_method}' not implemented"
                )

        # Use last layer only
        else:
            word_embeddings = outputs[0]  # [B, T, H]
            sent_embeddings = outputs[1]  # [B, H] (pooler output / CLS)
            sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

        # Local projection to embedding_dim
        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)  # [B, D, T]

        # Global projection
        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        # Optional L2 normalization
        if self.norm:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents
