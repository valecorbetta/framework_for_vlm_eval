from typing import List, Tuple
import albumentations as A
import cv2
import torch
import numpy as np


def prepare_fgadr_augmentations(
    seed: int,
    mean: Tuple[float] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float] = (0.26862954, 0.26130258, 0.27577711),
    resolution: int = 224,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.0,
    hue: float = 0.0,
):
    # Use ReplayCompose for augmentations so we can replay spatial transforms on masks
    augmentations = A.Compose(
        [
            A.Rotate(
                limit=30,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
        ],
        seed=seed,
    )

    preprocessing = A.Compose(
        [
            A.Resize(resolution, resolution),
            A.Normalize(mean=mean, std=std),
            A.pytorch.ToTensorV2(),
        ],
        p=1.0,
        seed=seed,
    )

    return augmentations, preprocessing


def unnormalize_to_uint8_rgb(
    x: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]
) -> np.ndarray:
    """
    x: normalized tensor of shape (1,3,H,W) or (3,H,W)
       normalized using (img - mean) / std for each channel.
    mean, std: 3-element tuples (one per RGB channel)

    Returns: uint8 RGB image (H,W,3)
    """
    # Remove batch if present
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]  # -> (3,H,W)

    assert x.ndim == 3 and x.shape[0] == 3, "Expected CHW RGB tensor"

    # Move to CPU numpy
    img = x.detach().cpu().clone()

    # Unnormalize channel-wise:
    # x_c = x_c * std[c] + mean[c]
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    # Clamp into [0,1]
    img = img.clamp(0.0, 1.0)

    # Convert to HWC uint8
    img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    return img
