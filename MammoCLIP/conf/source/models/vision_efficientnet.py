from pathlib import Path
from typing import List, Optional
from omegaconf import ListConfig
from torch import nn
import torch
from peft import LoraConfig, get_peft_model, PeftModel

from source.models.efficientnet import EfficientNet
from source.utils.misc import update_state_dict, freeze_params


class VisionEfficientNet(nn.Module):
    def __init__(
        self,
        lora,
        vision_encoder: EfficientNet,
        ckpt_clip: Optional[str] = None,
        lora_r: Optional[int] = None,
        lora_alpha_mult: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        lora_target_modules: Optional[List[str]] = None,
    ):
        """
        forward(x, get_local=True) -> (global_ft, local_ft)
        generate_embeddings(global_ft, local_ft) -> (img_emb_g, img_emb_l)

        If ckpt_clip is None, the vision encoder keeps its ImageNet pretrained
        weights (loaded by EfficientNet.from_pretrained) without any override.
        """
        super(VisionEfficientNet, self).__init__()
        self.vision_encoder = vision_encoder
        if ckpt_clip is not None:
            ckpt = torch.load(ckpt_clip, map_location="cpu", weights_only=False)
            # Handle wrapped checkpoints
            if "model" in ckpt:
                ckpt = ckpt["model"]
            elif "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            vision_encoder_weights = update_state_dict(ckpt, "image_encoder.")
            self.vision_encoder.load_state_dict(vision_encoder_weights, strict=True)
            print("Loaded MammoCLIP vision encoder weights from checkpoint.")
        else:
            print(
                "No MammoCLIP checkpoint provided — using ImageNet pretrained weights."
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        print(f"{lora=}")
        if lora:
            lora_alpha = lora_alpha_mult * lora_r
            if lora_target_modules is None:
                lora_target_modules = ["out_proj", "c_fc", "c_proj"]
            elif isinstance(lora_target_modules, ListConfig):
                lora_target_modules = list(lora_target_modules)

            # Collect all BatchNorm module names so their running statistics
            # (running_mean, running_var) are saved/loaded with the adapter.
            # Without this, PEFT only saves LoRA matrices and the BN stats
            # reset to pretrained values at test time, causing a train/test
            # mismatch.  See: https://huggingface.co/docs/peft/developer_guides/troubleshooting
            bn_modules_to_save = [
                name
                for name, module in self.vision_encoder.named_modules()
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))
            ]
            print(
                f"[VisionEfficientNet] Adding {len(bn_modules_to_save)} BatchNorm "
                f"layers to modules_to_save for PEFT checkpoint consistency."
            )

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                modules_to_save=bn_modules_to_save,
            )

            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            self.vision_encoder.print_trainable_parameters()

        else:
            freeze_params(self.vision_encoder)

        self.vit = self._get_vision_backbone()

    def _get_vision_backbone(self) -> EfficientNet:
        """
        Helper: get underlying EfficientNet wrapped in PEFT
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

    def efficientnet_forward(self, x):
        # Get multi-scale maps at each reduction
        endpoints = self.vision_encoder.extract_endpoints(x)

        local_ft = endpoints["reduction_4"]
        final_map = endpoints[list(endpoints.keys())[-1]]  # last is head

        # Global vector from final_map
        global_ft = self.pool(final_map)  # [B, C, 1, 1]
        global_ft = global_ft.flatten(1)  # [B, C]

        return global_ft, local_ft

    def forward(self, x: torch.Tensor, get_local: bool = False):
        if get_local:
            return self.efficientnet_forward(x)
        else:
            global_ft, _ = self.efficientnet_forward(x)
            return global_ft
