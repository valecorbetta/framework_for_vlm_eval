from typing import List
from omegaconf import ListConfig
from torch import nn
import torch
from peft import LoraConfig, get_peft_model

from RetCLIP.source.model.model import BertModel, FullTokenizer
from RetCLIP.source.utils.misc import update_state_dict, freeze_params


class TextBert(nn.Module):
    def __init__(
        self,
        ckpt_clip,
        text_encoder: BertModel,
        tokenizer: FullTokenizer,
        text_hidden_size: int,
        embed_dim: int,
        last_n_layers: int,
        aggregate_method: str,
        norm: bool,
        agg_tokens: bool,
        lora: bool,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_alpha_mult: int = 2,
        lora_target_modules: List[str] = None,
    ):
        super(TextBert, self).__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.idxtoword = self.tokenizer.inv_vocab

        self.last_n_layers = last_n_layers
        self.aggregate_method = aggregate_method
        self.norm = norm
        self.agg_tokens = agg_tokens

        self.text_hidden_size = text_hidden_size
        self.embedding_dim = embed_dim

        # Projections from BERT hidden size -> MICA/CLIP embedding_dim.
        if self.text_hidden_size == self.embedding_dim:
            self.emb_local = None
            self.emb_global = None
        else:
            self.emb_local = nn.Linear(self.text_hidden_size, self.embedding_dim)
            self.emb_global = nn.Linear(self.text_hidden_size, self.embedding_dim)

        # Load BERT weights from CLIP checkpoint
        ckpt = torch.load(ckpt_clip, map_location="cpu")
        bert_state = update_state_dict(ckpt, "bert.")
        self.text_encoder.load_state_dict(bert_state, strict=False)

        # Enable hidden states if we need last_N_layers aggregation
        if self.last_n_layers > 1:
            self.text_encoder.config.output_hidden_states = True
            self.text_encoder.encoder.output_hidden_states = True

        # Optionally wrap with LoRA
        if lora:
            if lora_target_modules is None:
                # attention projections
                lora_target_modules = ["query", "key", "value"]
            elif isinstance(lora_target_modules, ListConfig):
                lora_target_modules = list(lora_target_modules)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            self.text_encoder.print_trainable_parameters()
        else:
            freeze_params(self.text_encoder)

    # ------------------------------------------------------------------
    # Token aggregation (wordpiece -> word), same logic as MICA BertEncoder
    # ------------------------------------------------------------------
    def aggregate_tokens(self, embeddings: torch.Tensor, caption_ids: torch.Tensor):
        """
        embeddings: [B, num_layers, T, H]
        caption_ids: [B, T]

        Returns:
            agg_embs_batch: [B, num_layers, T, H]
            sentences:      List[List[str]] with length B
        """
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)  # [B, T, num_layers, H]
        agg_embs_batch = []
        sentences = []

        device = embeddings.device

        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword[int(word_id.item())]

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

            # If sequence ended without seeing [SEP] but we have a word in progress
            if len(token_bank) > 0 and (len(words) == 0 or words[-1] != "[SEP]"):
                new_emb = torch.stack(token_bank).sum(dim=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))

            agg_embs = torch.stack(agg_embs)  # [T_eff, num_layers, H]
            padding_size = num_words - agg_embs.size(0)
            paddings = torch.zeros(
                padding_size, num_layers, dim, device=device
            )  # [pad, num_layers, H]
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings], dim=0))
            sentences.append(words)

        # [B, T, num_layers, H] -> [B, num_layers, T, H]
        agg_embs_batch = torch.stack(agg_embs_batch).permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(
        self,
        ids: torch.Tensor,  # [B, T]
        attn_mask: torch.Tensor,  # [B, T] (1=keep, 0=pad)
        token_type_ids: torch.Tensor,  # [B, T]
    ):
        """
        Returns:
            text_emb_l: [B, embedding_dim, T]  (local / word-level)
            text_emb_g: [B, embedding_dim]     (global / sentence-level)
            sents:      List[List[str]]
        """

        outputs = self.text_encoder(
            input_ids=ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        # outputs = (sequence_output, pooled_output, hidden_states?, attentions?)
        sequence_output = outputs[0]  # [B, T, H]

        # ---- Case 1: aggregate last N layers (MICA-style) ----
        if self.last_n_layers > 1:
            # outputs[2] is all_hidden_states (tuple) when output_hidden_states=True
            all_hidden_states = outputs[2]  # (embeddings, layer1, ..., layerL)
            # stack last N layers: [N, B, T, H]
            embeddings = torch.stack(all_hidden_states[-self.last_n_layers :])
            embeddings = embeddings.permute(1, 0, 2, 3)  # [B, N, T, H]

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[int(w.item())] for w in sent] for sent in ids]

            # sentence embeddings as mean over tokens: [B, N, H]
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

        # ---- Case 2: last layer only ----
        else:
            word_embeddings = sequence_output  # [B, T, H]
            sent_embeddings = sequence_output[:, 0, :]  # CLS token [B, H]
            sents = [[self.idxtoword[int(w.item())] for w in sent] for sent in ids]

        # ---- Local projection to embedding_dim ----
        B, T, H = word_embeddings.shape
        word_embeddings = word_embeddings.view(B * T, H)  # [B*T, H]

        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)  # [B*T, D]
            local_dim = self.embedding_dim
        else:
            local_dim = H

        word_embeddings = word_embeddings.view(B, T, local_dim)  # [B, T, D]
        word_embeddings = word_embeddings.permute(0, 2, 1)  # [B, D, T]

        # ---- Global projection ----
        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)  # [B, D]

        # ---- Optional L2 normalization ----
        if self.norm:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, p=2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, p=2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents
