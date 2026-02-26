from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class RunContext:
    cfg: DictConfig
    device: torch.device
    hydra_subdir: Path
    split_root: Path
    images_root: list[Path] | Path
    overlay_percentages: list[float]
    num_splits: int


def build_overlay_cfg(cfg: DictConfig, which: str, pct: float) -> DictConfig:
    if which == "train":
        base = OmegaConf.to_container(cfg.DATASET.overlay_cfg_train, resolve=True)
    elif which == "test":
        base = OmegaConf.to_container(cfg.DATASET.overlay_cfg_test, resolve=True)
    else:
        raise ValueError(f"which must be 'train' or 'test', got {which}")

    out = deepcopy(base)
    out["percent"] = float(pct)
    out["enabled"] = bool(pct > 0) and bool(out.get("mode", "same"))
    return OmegaConf.create(out)


def exp_dir_for(
    ctx: RunContext, pct: float, split: int, root: Optional[Path] = None
) -> Path:
    base = ctx.hydra_subdir if root is None else root
    return base / f"pct_{int(pct * 100):03d}" / f"split{split:01d}"
