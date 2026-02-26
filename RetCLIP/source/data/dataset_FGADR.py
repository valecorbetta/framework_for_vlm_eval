from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

from RetCLIP.source.utils.overlay_spurious import (
    overlay_spurious,
    precompute_balanced_overlay_mask,
    precompute_waterbirds_overlay_mask,
)
from RetCLIP.source.model.model import FullTokenizer

# Mapping from concept column names to mask folder names
CONCEPT_MASK_FOLDERS = {
    "has_HardExudate": "HardExudate_Masks",
    "has_Hemohedge": "Hemohedge_Masks",
    "has_IRMA": "IRMA_Masks",
    "has_Microaneurysms": "Microaneurysms_Masks",
    "has_Neovascularization": "Neovascularization_Masks",
    "has_SoftExudate": "SoftExudate_Masks",
}


class FGADRDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        overlay_cfg: DictConfig,
        augmentations: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        label_mode: str = "retinopathy_grade",
        mask_root: Optional[Path] = None,
        return_masks: bool = False,
    ):
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        self.image_root = image_root
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.label_mode = label_mode
        self.overlay_cfg = overlay_cfg
        self.mask_root = mask_root
        self.return_masks = return_masks
        self.labels = [
            self.build_label(self.df.iloc[idx]) for idx in range(len(self.df))
        ]
        # Get concept keys from dataframe columns
        self.concept_keys = sorted([c for c in self.df.columns if c.startswith("has_")])

        # Precompute balanced overlay mask (ensures equal % per class)
        self._overlay_mask = self._build_overlay_mask()

    def _build_overlay_mask(self) -> set[str] | None:
        """Precompute which images should receive overlays (balanced per class)."""
        if not self.overlay_cfg:
            return None
        if not self.overlay_cfg.get("enabled", False):
            return None
        percent = float(self.overlay_cfg.get("percent", 0))
        if percent <= 0:
            return None

        uids = [str(row["image"]) for _, row in self.df.iterrows()]
        seed = int(self.overlay_cfg.get("seed", 0))

        return precompute_balanced_overlay_mask(
            uids=uids,
            labels=self.labels,
            percent=percent,
            seed=seed,
            skip_class=0,  # Class 0 never receives overlays
        )

    def build_label(self, row) -> int:
        if self.label_mode == "referable_dr":
            # grades >= 2 positive class; used in Kaggle EyePACS, Messidor benchmark and screening
            out = int(row["dr_grade"] >= 2)

        elif self.label_mode == "referable_degrees":
            # non referable vs. referable-moderate vs. referable-severe
            out = 0 if row["dr_grade"] <= 1 else 1 if row["dr_grade"] == 2 else 2

        elif self.label_mode == "retinopathy_grade":
            out = int(row["dr_grade"])
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        return out

    def get_concepts_meta(self, row) -> dict:
        concept_cols = [c for c in self.df.columns if c.startswith("has_")]
        return {col: int(row[col]) for col in concept_cols}

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _imread(path: Path) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
        if im is None:
            raise FileNotFoundError(f"cv2 failed to read {path}")
        return im

    @staticmethod
    def _imread_mask(path: Path) -> np.ndarray:
        """Read a mask as grayscale, return as HxW uint8."""
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"cv2 failed to read mask {path}")
        return im

    def _load_concept_masks(self, row) -> Dict[str, np.ndarray]:
        """
        Load concept segmentation masks for concepts that are present (has_* == 1).
        Returns a dict mapping concept key (e.g., 'has_HardExudate') to mask array.
        """
        masks = {}
        if not self.return_masks or self.mask_root is None:
            return masks

        image_name = row["image"]
        for concept_key in self.concept_keys:
            if int(row[concept_key]) == 1 and concept_key in CONCEPT_MASK_FOLDERS:
                mask_folder = CONCEPT_MASK_FOLDERS[concept_key]
                mask_path = self.mask_root / mask_folder / image_name
                if mask_path.exists():
                    masks[concept_key] = self._imread_mask(mask_path)
        return masks

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = self.image_root / row["image"]
        img = self._imread(path)
        label = self.build_label(row)
        H_orig, W_orig = img.shape[:2]

        # Load concept masks before any transforms
        concept_masks_dict = self._load_concept_masks(row)

        # Prepare masks list for albumentations (in consistent order)
        mask_keys = list(self.concept_keys)  # consistent ordering
        masks = []
        if self.return_masks:
            concept_masks_dict = self._load_concept_masks(row)
            for concept_key in mask_keys:
                if concept_key in concept_masks_dict:
                    masks.append(concept_masks_dict[concept_key])
                else:
                    masks.append(np.zeros((H_orig, W_orig), dtype=np.uint8))

        enabled = bool(self.overlay_cfg.get("enabled", False))
        percent = float(self.overlay_cfg.get("percent", 0.0))

        # Waterbirds-style configs define per-class probabilities instead of percent
        has_waterbirds_schedule = (
            self.overlay_cfg.get("p_by_class", None) is not None
            or self.overlay_cfg.get("rho", None) is not None
        )

        apply_overlay = enabled and (percent > 0 or has_waterbirds_schedule)

        if self.augmentations is not None:
            if self.return_masks and masks:
                aug_result = self.augmentations(image=img, masks=masks)
                img = aug_result["image"]
                masks = aug_result["masks"]
            else:
                aug_result = self.augmentations(image=img)
                img = aug_result["image"]

        artifact_mask = None
        if apply_overlay:
            uid = str(row["image"])
            img, spurious_applied, spurious_type, _, artifact_mask = overlay_spurious(
                img, label, uid, self.overlay_cfg, overlay_mask=self._overlay_mask
            )
            if spurious_type is None:
                spurious_type = "none"
        else:
            spurious_applied = False
            spurious_type = "none"

        # Apply preprocessing to image and masks
        if self.preprocessing is not None:
            if self.return_masks:
                # Create artifact_mask if it doesn't exist (all zeros)
                if artifact_mask is None:
                    H, W = img.shape[:2]
                    artifact_mask = np.zeros((H, W), dtype=np.uint8)
                # Add artifact_mask to masks list for resizing
                all_masks = masks + [artifact_mask]
                preprocess_result = self.preprocessing(image=img, masks=all_masks)
                img = preprocess_result["image"]
                processed_masks = preprocess_result["masks"]
                # Split back: concept masks and artifact_mask
                masks = processed_masks[:-1]
                artifact_mask = processed_masks[-1]
            else:
                preprocess_result = self.preprocessing(image=img)
                img = preprocess_result["image"]

        # build sample

        sample = {
            "filename": row["image"],
            "x": img,
            "y": torch.as_tensor(label, dtype=torch.long),
            "spurious_applied": spurious_applied,
            "spurious_type": spurious_type,
            **self.get_concepts_meta(row),
        }

        # Add masks to sample only if return_masks is True
        if self.return_masks:
            # Determine final mask size
            if isinstance(img, torch.Tensor):
                mask_H, mask_W = img.shape[1], img.shape[2]  # CHW format
            else:
                mask_H, mask_W = img.shape[:2]

            # Add artifact mask
            if artifact_mask is None:
                artifact_mask = np.zeros((mask_H, mask_W), dtype=np.uint8)
            sample["artifact_mask"] = torch.as_tensor(artifact_mask, dtype=torch.uint8)

            # Add concept masks
            for i, concept_key in enumerate(mask_keys):
                # Binarize mask (in case it's not already binary)
                m = masks[i]
                if isinstance(m, torch.Tensor):
                    mask = (m > 0).byte()
                else:
                    mask = (m > 0).astype(np.uint8)
                sample[f"{concept_key}_mask"] = torch.as_tensor(mask, dtype=torch.uint8)

        return sample


class FGADRConceptDataset(FGADRDataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        overlay_cfg: DictConfig,
        concept_map: DictConfig,
        augmentations: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        label_mode: str = "retinopathy_grade",
        mask_root: Optional[Path] = None,
        return_masks: bool = False,
    ):
        super().__init__(
            csv_path=csv_path,
            image_root=image_root,
            overlay_cfg=overlay_cfg,
            augmentations=augmentations,
            preprocessing=preprocessing,
            label_mode=label_mode,
            mask_root=mask_root,
            return_masks=return_masks,
        )
        self.concept_map = concept_map
        self.concept_keys = [k for k in self.concept_map.keys() if k.startswith("has_")]
        self.concept_keys = sorted(self.concept_keys)

    def build_caption(self, row: pd.DataFrame) -> str:
        present_concepts = [
            self.concept_map[k] for k in self.concept_keys if int(row[k]) == 1
        ]

        if len(present_concepts) == 0:
            return self.concept_map["no_concepts"]
        else:
            return self.concept_map["concept_present"] + " ".join(present_concepts)

    def build_concept_labels(self, row: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(
            [int(row[k]) for k in self.concept_keys], dtype=torch.float32
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        row = self.df.iloc[idx]

        caption = self.build_caption(row)
        concept_labels = self.build_concept_labels(row)

        sample.update({"caption": caption, "concept_labels": concept_labels})
        return sample


def mica_collate_fulltokenizer(
    batch: list[dict[str, Any]], tokenizer: FullTokenizer, max_length: int
):
    pad_id = tokenizer.vocab.get("[PAD]", 0)
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    unk_id = tokenizer.vocab.get("[UNK]", 100)

    # images
    imgs = torch.stack([b["x"] for b in batch])  # [B, C, H, W]

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for b in batch:
        text = b["caption"]
        toks = tokenizer.tokenize(text)
        # map unknown tokens safely
        toks = [t if t in tokenizer.vocab else "[UNK]" for t in toks]
        ids = tokenizer.convert_tokens_to_ids(toks)

        # truncate for special tokens
        ids = ids[: max_length - 2]
        ids = [cls_id] + ids + [sep_id]

        mask = [1] * len(ids)
        types = [0] * len(ids)

        # pad
        pad_n = max_length - len(ids)
        if pad_n > 0:
            ids += [pad_id] * pad_n
            mask += [0] * pad_n
            types += [0] * pad_n
        else:
            ids = ids[:max_length]
            mask = mask[:max_length]
            types = types[:max_length]

        input_ids.append(ids)
        attention_mask.append(mask)
        token_type_ids.append(types)

    concept_labels = torch.stack([b["concept_labels"] for b in batch]).float()

    return {
        "x": imgs,
        "caption_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "concept_labels": concept_labels,
        "filename": [b["filename"] for b in batch],
        "y": torch.stack([b["y"] for b in batch]),
    }

