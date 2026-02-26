from pathlib import Path
from typing import List, Optional
from omegaconf import ListConfig
from torch import nn
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from RetCLIP.source.model.model import VisualTransformer
from RetCLIP.source.utils.misc import update_state_dict, freeze_params


class VisionViT(nn.Module):
    def __init__(
        self,
        lora,
        ckpt_clip,
        vision_encoder: VisualTransformer,
        lora_r: Optional[int] = None,
        lora_alpha_mult: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        lora_target_modules: Optional[List[str]] = None,
    ):
        """
        forward(x, get_local=True) -> (global_ft, local_ft)
        generate_embeddings(global_ft, local_ft) -> (img_emb_g, img_emb_l)
        """
        super(VisionViT, self).__init__()
        self.vision_encoder = vision_encoder
        ckpt_clip = torch.load(ckpt_clip, map_location="cpu")
        vision_encoder_weights = update_state_dict(ckpt_clip, "module.visual.")
        self.vision_encoder.load_state_dict(vision_encoder_weights, strict=True)
        print(f"{lora=}")
        if lora:
            lora_alpha = lora_alpha_mult * lora_r
            if lora_target_modules is None:
                lora_target_modules = ["out_proj", "c_fc", "c_proj"]
            elif isinstance(lora_target_modules, ListConfig):
                lora_target_modules = list(lora_target_modules)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )

            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            self.vision_encoder.print_trainable_parameters()

        else:
            freeze_params(self.vision_encoder)

        self.vit = self._get_vit()
        self.feature_dim = self.vit.ln_post.normalized_shape[0]
        if isinstance(self.feature_dim, (list, tuple)):
            self.feature_dim = self.feature_dim[0]

        # Natural ViT spatial grid size (e.g. 14 x 14 for 224x224 / 16)
        self.grid_h, self.grid_w = self.vit.grid_size

        # Embedding dimension in CLIP joint space (output_dim of proj, e.g. 512)
        self.embed_dim = (
            self.vit.proj.shape[1] if self.vit.proj is not None else self.feature_dim
        )

    def _get_vit(self) -> VisualTransformer:
        """
        Helper: get underlying VisualTransformer wrapped in PEFT
        """
        m = self.vision_encoder
        if hasattr(m, "base_model"):
            return m.base_model
        return m

    def has_lora(self) -> bool:
        return isinstance(self.vision_encoder, PeftModel)

    def save_lora(self, path: str | Path):
        if not isinstance(self.vision_encoder, PeftModel):
            raise RuntimeError("No LoRA adapter to save.")
        self.vision_encoder.save_pretrained(path)

    def load_lora(self, path: str | Path):
        self.vision_encoder = PeftModel.from_pretrained(self.vision_encoder, path)

    # ------------------------------------------------------------------
    # ViT feature extraction: CLS global + patch locals in CLIP space
    # ------------------------------------------------------------------
    def vit_forward(self, x: torch.Tensor):
        """
        Returns:
            global_ft: [B, embed_dim]
            local_ft:  [B, embed_dim, H, W]  (H,W from vit.grid_size)
        """
        B = x.size(0)

        x = x.to(self.vit.conv1.weight.dtype)

        # Patch embedding
        x = self.vit.conv1(x)  # [B, width, gh, gw]
        x = x.reshape(B, x.shape[1], -1)  # [B, width, L]
        x = x.permute(0, 2, 1)  # [B, L, width]

        # Prepend CLS token
        cls_token = self.vit.class_embedding.to(x.dtype)
        cls_tokens = cls_token + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, width]

        # Add positional embeddings
        x = x + self.vit.positional_embedding.to(x.dtype)

        # Pre-transformer LN
        x = self.vit.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)  # [L+1, B, width]
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L+1, width]

        # Separate CLS and patch tokens
        cls = x[:, 0, :]  # [B, width]
        patches = x[:, 1:, :]  # [B, L, width], L = grid_h * grid_w

        # Global feature: CLIP-style
        cls_ln = self.vit.ln_post(cls)  # [B, width]
        if self.vit.proj is not None:
            global_ft = cls_ln @ self.vit.proj  # [B, embed_dim]
        else:
            global_ft = cls_ln  # [B, width]

        # Local features: project patch tokens to embed_dim and reshape to grid
        if self.vit.proj is not None:
            patch_emb = patches @ self.vit.proj  # [B, L, embed_dim]
        else:
            patch_emb = patches  # [B, L, width]

        B, L, D = patch_emb.shape
        assert (
            L == self.grid_h * self.grid_w
        ), f"Patch tokens ({L}) != grid_h*grid_w ({self.grid_h * self.grid_w})"

        # [B, D, H, W]
        local_ft = patch_emb.permute(0, 2, 1).reshape(B, D, self.grid_h, self.grid_w)

        return global_ft, local_ft

    def forward(self, x: torch.Tensor, get_local: bool = False):
        if get_local:
            return self.vit_forward(x)
        else:
            global_ft, _ = self.vit_forward(x)
            return global_ft
