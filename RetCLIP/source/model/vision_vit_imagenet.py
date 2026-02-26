"""
VisionViT variant that initialises the VisualTransformer with
ImageNet-pretrained ViT-B/16 weights (from timm) instead of the
RetCLIP checkpoint.

The CLIP-specific projection layer (`proj`, 768 → 512) has no
ImageNet counterpart and is left at its random initialisation.
"""

from pathlib import Path
from typing import List, Optional
from omegaconf import ListConfig
from torch import nn
import torch
import timm
from peft import LoraConfig, get_peft_model, PeftModel

from RetCLIP.source.model.model import VisualTransformer
from RetCLIP.source.utils.misc import freeze_params


# ── weight-mapping helpers ──────────────────────────────────────────


def _map_timm_to_visual_transformer(timm_sd: dict) -> dict:
    """
    Convert a timm `vit_base_patch16_224` state-dict to the key
    convention used by `VisualTransformer`.

    Keys that have no counterpart (e.g. `head.*`, `patch_embed.proj.bias`
    when conv1 has no bias, or `pre_logits.*`) are silently dropped.
    """
    out = {}

    # ── patch embedding ──
    if "patch_embed.proj.weight" in timm_sd:
        out["conv1.weight"] = timm_sd["patch_embed.proj.weight"]
    # conv1 in VisualTransformer has bias=False, so skip proj.bias

    # ── cls token  (timm [1, 1, D] → custom [D]) ──
    if "cls_token" in timm_sd:
        out["class_embedding"] = timm_sd["cls_token"].squeeze(0).squeeze(0)

    # ── positional embedding  (timm [1, N, D] → custom [N, D]) ──
    if "pos_embed" in timm_sd:
        out["positional_embedding"] = timm_sd["pos_embed"].squeeze(0)

    # ── pre-transformer layer norm ──
    # timm ViT (augreg) does NOT have a pre-LN by default,
    # but some variants do.  Fall back gracefully.
    for src_prefix, dst_prefix in [
        ("norm_pre", "ln_pre"),
        ("patch_embed.norm", "ln_pre"),  # some timm variants
    ]:
        w_key = f"{src_prefix}.weight"
        b_key = f"{src_prefix}.bias"
        if w_key in timm_sd:
            out[f"{dst_prefix}.weight"] = timm_sd[w_key]
            out[f"{dst_prefix}.bias"] = timm_sd[b_key]
            break

    # ── transformer blocks ──
    i = 0
    while f"blocks.{i}.norm1.weight" in timm_sd:
        src = f"blocks.{i}"
        dst = f"transformer.resblocks.{i}"

        # layer-norms
        out[f"{dst}.ln_1.weight"] = timm_sd[f"{src}.norm1.weight"]
        out[f"{dst}.ln_1.bias"] = timm_sd[f"{src}.norm1.bias"]
        out[f"{dst}.ln_2.weight"] = timm_sd[f"{src}.norm2.weight"]
        out[f"{dst}.ln_2.bias"] = timm_sd[f"{src}.norm2.bias"]

        # self-attention  (timm fuses q/k/v into one matrix)
        out[f"{dst}.attn.in_proj_weight"] = timm_sd[f"{src}.attn.qkv.weight"]
        out[f"{dst}.attn.in_proj_bias"] = timm_sd[f"{src}.attn.qkv.bias"]
        out[f"{dst}.attn.out_proj.weight"] = timm_sd[f"{src}.attn.proj.weight"]
        out[f"{dst}.attn.out_proj.bias"] = timm_sd[f"{src}.attn.proj.bias"]

        # MLP
        out[f"{dst}.mlp.c_fc.weight"] = timm_sd[f"{src}.mlp.fc1.weight"]
        out[f"{dst}.mlp.c_fc.bias"] = timm_sd[f"{src}.mlp.fc1.bias"]
        out[f"{dst}.mlp.c_proj.weight"] = timm_sd[f"{src}.mlp.fc2.weight"]
        out[f"{dst}.mlp.c_proj.bias"] = timm_sd[f"{src}.mlp.fc2.bias"]

        i += 1

    # ── final layer norm ──
    if "norm.weight" in timm_sd:
        out["ln_post.weight"] = timm_sd["norm.weight"]
        out["ln_post.bias"] = timm_sd["norm.bias"]
    # timm may also use fc_norm in some variants
    elif "fc_norm.weight" in timm_sd:
        out["ln_post.weight"] = timm_sd["fc_norm.weight"]
        out["ln_post.bias"] = timm_sd["fc_norm.bias"]

    # NOTE: `proj` (width → output_dim, e.g. 768 → 512) is CLIP-specific
    # and has no ImageNet counterpart.  It stays randomly initialised.

    return out


def _load_imagenet_weights_into(
    vision_encoder: VisualTransformer, timm_model_name: str = "vit_base_patch16_224"
) -> None:
    """
    Download ImageNet-pretrained weights via timm and load
    the mapped subset into the custom VisualTransformer.
    """
    timm_model = timm.create_model(timm_model_name, pretrained=True)
    timm_sd = timm_model.state_dict()

    mapped_sd = _map_timm_to_visual_transformer(timm_sd)

    missing, unexpected = vision_encoder.load_state_dict(mapped_sd, strict=False)

    # `proj` and possibly `ln_pre` are expected to be missing
    print(f"[VisionViTImageNet] Loaded ImageNet weights from timm '{timm_model_name}'")
    if missing:
        print(f"  Keys kept at random init (no ImageNet counterpart): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")


# ── main class ──────────────────────────────────────────────────────


class VisionViTImageNet(nn.Module):
    """
    Drop-in replacement for VisionViT that initialises the
    VisualTransformer backbone with ImageNet-pretrained weights
    (ViT-B/16 via timm) instead of the RetCLIP checkpoint.

    The API (forward, has_lora, save_lora, load_lora, etc.) is
    identical to VisionViT so that FundusClassifier and the
    trainer work without changes.
    """

    def __init__(
        self,
        lora: bool,
        vision_encoder: VisualTransformer,
        timm_model_name: str = "vit_base_patch16_224",
        lora_r: Optional[int] = None,
        lora_alpha_mult: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder

        # ── Load ImageNet-pretrained weights ──
        _load_imagenet_weights_into(self.vision_encoder, timm_model_name)

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

        self.grid_h, self.grid_w = self.vit.grid_size
        # Always use the raw transformer output dim (768 for ViT-B/16).
        # The CLIP proj matrix has no ImageNet counterpart and is randomly
        # initialised, so we bypass it entirely — using it would corrupt the
        # features with a random linear transformation.
        self.embed_dim = self.feature_dim

    # ── helpers (identical to VisionViT) ──

    def _get_vit(self) -> VisualTransformer:
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

    # ── forward (identical to VisionViT) ──

    def vit_forward(self, x: torch.Tensor):
        B = x.size(0)
        x = x.to(self.vit.conv1.weight.dtype)

        x = self.vit.conv1(x)
        x = x.reshape(B, x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = self.vit.class_embedding.to(x.dtype)
        cls_tokens = cls_token + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)

        cls = x[:, 0, :]
        patches = x[:, 1:, :]

        # Bypass proj entirely — it is randomly initialised for ImageNet weights
        # and would corrupt the features. The 768-dim ln_post output is the best
        # representation this backbone can produce.
        global_ft = self.vit.ln_post(cls)
        patch_emb = patches

        B, L, D = patch_emb.shape
        assert L == self.grid_h * self.grid_w
        local_ft = patch_emb.permute(0, 2, 1).reshape(B, D, self.grid_h, self.grid_w)

        return global_ft, local_ft

    def forward(self, x: torch.Tensor, get_local: bool = False):
        if get_local:
            return self.vit_forward(x)
        else:
            global_ft, _ = self.vit_forward(x)
            return global_ft
