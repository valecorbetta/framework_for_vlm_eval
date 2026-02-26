from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class CheckpointPaths:
    best_classifier_head: Optional[Path] = None
    best_lora_dir: Optional[Path] = None
    best_stage1_ckpt: Optional[Path] = None
    best_stage2_ckpt: Optional[Path] = None
