import ast
import os, re, cv2
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import difflib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from source.utils.overlay_spurious import (
    overlay_spurious,
    precompute_balanced_overlay_mask,
)


class EmbedAnnotatedDataset(Dataset):
    # Only concepts with sufficient samples for CAV training (>=10 per class)
    CONCEPT_COLUMNS = [
        "has_benign_mass",
        "has_suspicious_mass",
        "has_focal_asymmetry",
        "has_suspicious_calcifications",
        "has_benign_calcifications",
    ]

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        *,
        augmentations: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        overlay_cfg: Optional[Dict[str, Any]] = None,
        label_mode: str = "diagnosis",
        image_id_col: str = "image_id",
        birads_col: str = "birads_image",
        laterality_col: str = "laterality",
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.image_root = Path(image_root)
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.overlay_cfg = overlay_cfg
        self.label_mode = label_mode
        self.image_id_col = image_id_col
        self.birads_col = birads_col
        self.laterality_col = laterality_col

        # Build concept columns
        self._build_concept_columns()

        # Build labels
        self.labels = self._build_labels()

        # Precompute balanced overlay mask (ensures equal % per class)
        self._overlay_mask = self._build_overlay_mask()

        # Validate image paths exist
        self._build_path_index()

    def _build_concept_columns(self):
        """Build concept boolean columns from the raw annotation columns."""
        df = self.df

        # has_benign_mass / has_suspicious_mass: based on mass_margin values
        _BENIGN_MARGINS = {"circumscribed"}
        _SUSPICIOUS_MARGINS = {"obscured", "microlobulated", "indistinct", "spiculated"}

        def _classify_mass_margins(margin_val):
            if pd.isna(margin_val) or margin_val == "":
                return False, False
            try:
                margins = ast.literal_eval(str(margin_val))
            except (ValueError, SyntaxError):
                return False, False
            vals = {v.strip().lower() for v in margins.values()}
            return bool(vals & _BENIGN_MARGINS), bool(vals & _SUSPICIOUS_MARGINS)

        margin_flags = df["mass_margin"].apply(_classify_mass_margins)
        df["has_benign_mass"] = margin_flags.apply(lambda t: t[0])
        df["has_suspicious_mass"] = margin_flags.apply(lambda t: t[1])

        # has_asymmetry: true if asymmetry == "Yes" AND asymmetry_type == "Asymmetry"
        df["has_asymmetry"] = (
            (df["asymmetry"].astype(str).str.lower() == "yes")
            & (
                df["asymmetry_type"]
                .astype(str)
                .str.lower()
                .str.contains("asymmetry", na=False)
            )
            & (
                ~df["asymmetry_type"]
                .astype(str)
                .str.lower()
                .str.contains("focal", na=False)
            )
        )

        # has_focal_asymmetry: true if asymmetry == "Yes" AND asymmetry_type == "Focal"
        df["has_focal_asymmetry"] = (
            df["asymmetry"].astype(str).str.lower() == "yes"
        ) & (
            df["asymmetry_type"].astype(str).str.lower().str.contains("focal", na=False)
        )

        # has_architectural_distortion: true if architectural_distortion == "Yes"
        df["has_architectural_distortion"] = (
            df["architectural_distortion"].astype(str).str.lower() == "yes"
        )

        # Helper function to check if a value is in associated_features list
        def check_associated_feature(val, feature_name):
            if pd.isna(val) or val == "":
                return False
            return feature_name.lower() in str(val).lower()

        # has_skin_thickening: true if "Skin thickening" in associated_features
        df["has_skin_thickening"] = df["associated_features"].apply(
            lambda x: check_associated_feature(x, "skin thickening")
        )

        # has_nipple_retraction: true if "Nipple retraction" in associated_features
        df["has_nipple_retraction"] = df["associated_features"].apply(
            lambda x: check_associated_feature(x, "nipple retraction")
        )

        # has_trabecular_thickening: true if "Trabecular thickening" in associated_features
        df["has_trabecular_thickening"] = df["associated_features"].apply(
            lambda x: check_associated_feature(x, "trabecular thickening")
        )

        # has_suspicious_calcifications:
        # - "Calcifications" in associated_features, OR
        # - calcifications == "Yes" AND calcification_morphology == "Suspicious"
        def check_suspicious_calc(row):
            assoc = row.get("associated_features", "")
            if pd.notna(assoc) and "calcifications" in str(assoc).lower():
                return True
            if (
                str(row.get("calcifications", "")).lower() == "yes"
                and "suspicious" in str(row.get("calcification_morphology", "")).lower()
            ):
                return True
            return False

        df["has_suspicious_calcifications"] = df.apply(check_suspicious_calc, axis=1)

        # has_benign_calcifications:
        # calcifications == "Yes" AND calcification_morphology contains "Typically benign"
        def check_benign_calc(row):
            if str(row.get("calcifications", "")).lower() == "yes":
                morph = str(row.get("calcification_morphology", "")).lower()
                if "typically benign" in morph:
                    return True
            return False

        df["has_benign_calcifications"] = df.apply(check_benign_calc, axis=1)

    def _build_labels(self) -> List[int]:
        """Build labels based on label_mode."""
        df = self.df
        birads = pd.to_numeric(df[self.birads_col], errors="coerce")

        if self.label_mode == "image_birads":
            return (birads - 1).astype(int).tolist()  # Convert 1-5 to 0-4

        elif self.label_mode == "followup":
            return (birads >= 3).astype(int).tolist()

        elif self.label_mode == "diagnosis":
            # BI-RADS 1 -> 0, BI-RADS 2,3 -> 1, BI-RADS 4,5 -> 2
            return [
                0 if v == 1 else 1 if v in (2, 3) else 2 for v in birads.astype(int)
            ]

        elif self.label_mode == "one_vs_all":
            return (birads >= 2).astype(int).tolist()

        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def _build_overlay_mask(self) -> set[str] | None:
        """Precompute which images should receive overlays (balanced per class)."""
        if self.overlay_cfg is None:
            return None
        if not self.overlay_cfg.get("enabled", False):
            return None
        percent = float(self.overlay_cfg.get("percent", 0))
        if percent <= 0:
            return None

        uids = [str(row[self.image_id_col]) for _, row in self.df.iterrows()]
        seed = int(self.overlay_cfg.get("seed", 0))

        return precompute_balanced_overlay_mask(
            uids=uids,
            labels=self.labels,
            percent=percent,
            seed=seed,
            skip_class=0,  # Class 0 never receives overlays
        )

    def _build_path_index(self) -> None:
        """Validate that processed images exist for all rows."""
        missing = []
        for _, row in self.df.iterrows():
            path = self._transform_dicom_path(row.get("anon_dicom_path", ""))
            if not path.exists():
                missing.append(str(path))

        if missing:
            print(
                f"[EmbedAnnotatedDataset] WARNING: {len(missing)} images not found. Examples: {missing[:5]}"
            )
        else:
            print(f"[EmbedAnnotatedDataset] All {len(self.df)} images found.")

    def _transform_dicom_path(self, dicom_path: str) -> Path:
        """Transform the original DICOM path to the processed PNG path."""
        old_prefix = "/mnt/NAS2/mammo/anon_dicom/"
        new_prefix = "PATH_TO_PROCESSED_DICOMS"

        # Replace prefix and extension
        png_path = str(dicom_path).replace(old_prefix, new_prefix)
        png_path = png_path.replace(".dcm", ".png")

        return Path(png_path)

    def _get_image_path(self, row) -> Path:
        """Get processed image path from row's anon_dicom_path."""
        dicom_path = row.get("anon_dicom_path", "")
        if pd.isna(dicom_path) or dicom_path == "":
            raise FileNotFoundError(
                f"No anon_dicom_path for image_id: {row.get(self.image_id_col)}"
            )

        return self._transform_dicom_path(dicom_path)

    def _get_concept_flags(self, idx: int) -> Dict[str, int]:
        """Return dict of concept integer flags (0 or 1)."""
        row = self.df.iloc[idx]
        return {col: int(bool(row[col])) for col in self.CONCEPT_COLUMNS}

    @staticmethod
    def _parse_other_findings(val) -> Dict[str, int]:
        """Parse other_findings column into individual flags."""
        flags = {
            "clip_placement": 0,
            "implant": 0,
            "gynecomastia": 0,
            "post_surgical_changes": 0,
            "marker": 0,
        }

        if pd.isna(val) or val == "":
            return flags

        val_lower = str(val).lower()

        if "clip" in val_lower:
            flags["clip_placement"] = 1
        if "implant" in val_lower:
            flags["implant"] = 1
        if "gynecomastia" in val_lower or "subareolar density" in val_lower:
            flags["gynecomastia"] = 1
        if "post-surgical" in val_lower or "post surgical" in val_lower:
            flags["post_surgical_changes"] = 1
        if "marker" in val_lower and "clip" not in val_lower:
            flags["marker"] = 1

        return flags

    @staticmethod
    def _encode_breast_density(val) -> int:
        """Encode breast density to numeric: A=0, B=1, C=2, D=3, unknown=-1."""
        if pd.isna(val) or val == "":
            return -1
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return mapping.get(str(val).upper().strip(), -1)

    @staticmethod
    def _encode_laterality(val) -> int:
        """Encode laterality to numeric: L=0, R=1, unknown=-1."""
        if pd.isna(val) or val == "":
            return -1
        mapping = {"L": 0, "R": 1}
        return mapping.get(str(val).upper().strip(), -1)

    @staticmethod
    def _encode_birads_4_subcategory(val) -> int:
        """Encode BI-RADS 4 subcategory: 4A=0, 4B=1, 4C=2, unknown=-1."""
        if pd.isna(val) or val == "":
            return -1
        val_str = str(val).upper().strip()
        mapping = {"4A": 0, "4B": 1, "4C": 2}
        return mapping.get(val_str, -1)

    @staticmethod
    def _imread(path: Path) -> np.ndarray:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"cv2 failed to read {path}")
        return im

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = row[self.image_id_col]
        path = self._get_image_path(row)

        img = self._imread(path)
        label = self.labels[idx]
        apply_overlay = (
            bool(self.overlay_cfg.get("enabled", False))
            and float(self.overlay_cfg.get("percent", 0)) > 0
        )

        if self.augmentations is not None:
            img = self.augmentations(image=img)["image"]

        if apply_overlay:
            uid = str(row["image_id"])
            img, spurious_applied, spurious_type, _ = overlay_spurious(
                img, label, uid, self.overlay_cfg, overlay_mask=self._overlay_mask
            )
            if spurious_type is None:
                spurious_type = "none"
        else:
            spurious_applied = False
            spurious_type = "none"

        if self.preprocessing is not None:
            img = self.preprocessing(image=img)["image"]

        concept_flags = self._get_concept_flags(idx)
        other_findings_flags = self._parse_other_findings(row.get("other_findings", ""))

        sample = {
            "filename": image_id,
            "x": img,
            "y": torch.as_tensor(label, dtype=torch.long),
            "laterality": self._encode_laterality(row[self.laterality_col]),
            "breast_density": self._encode_breast_density(
                row.get("breast_density", None)
            ),
            "birads_4_subcategory": self._encode_birads_4_subcategory(
                row.get("birads_4_subcategory", None)
            ),
            "spurious_applied": int(spurious_applied),
            "spurious_type": spurious_type,
            **concept_flags,
            **other_findings_flags,
        }

        return sample


class EmbedAnnotatedConceptDataset(EmbedAnnotatedDataset):
    """EmbedAnnotatedDataset with concept labels tensor and caption generation."""

    DENSITY_COLUMNS = [
        "has_density_A",
        "has_density_B",
        "has_density_C",
        "has_density_D",
    ]

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        *,
        augmentations: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        overlay_cfg: Optional[Dict[str, Any]] = None,
        label_mode: str = "image_birads",
        image_id_col: str = "image_id",
        birads_col: str = "birads_image",
        laterality_col: str = "laterality",
    ):
        super().__init__(
            csv_path=csv_path,
            image_root=image_root,
            augmentations=augmentations,
            preprocessing=preprocessing,
            overlay_cfg=overlay_cfg,
            label_mode=label_mode,
            image_id_col=image_id_col,
            birads_col=birads_col,
            laterality_col=laterality_col,
        )

        # One-hot encode breast density into has_density_A/B/C/D columns
        self._build_density_columns()

        # Full concept keys: finding concepts + density concepts (sorted)
        self.concept_keys = sorted(self.CONCEPT_COLUMNS + self.DENSITY_COLUMNS)

    def _build_density_columns(self) -> None:
        """One-hot encode the breast_density column into has_density_{A,B,C,D}."""
        if "breast_density" not in self.df.columns:
            print(
                "[EmbedAnnotatedConceptDataset] WARNING: 'breast_density' column not found, density columns will be all zeros."
            )
            for grade in ("A", "B", "C", "D"):
                self.df[f"has_density_{grade}"] = 0
            return
        raw = self.df["breast_density"].astype(str).str.strip().str.upper()
        for grade in ("A", "B", "C", "D"):
            self.df[f"has_density_{grade}"] = (raw == grade).astype(int)

    @staticmethod
    def _norm_lat(x: str) -> str:
        s = str(x).strip().upper()
        return "Left breast" if s == "L" else "Right breast"

    @staticmethod
    def _parse_json_dict(val) -> Dict:
        """Parse JSON-like dict string from CSV."""
        if pd.isna(val) or val == "":
            return {}
        try:
            return ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return {}

    @staticmethod
    def _parse_json_list(val) -> List:
        """Parse JSON-like list string from CSV."""
        if pd.isna(val) or val == "":
            return []
        try:
            result = ast.literal_eval(str(val))
            return result if isinstance(result, list) else [result]
        except (ValueError, SyntaxError):
            return []

    def _build_mass_descriptions(self, row) -> List[str]:
        """Build descriptions for each mass: shape + margin + density."""
        shapes = self._parse_json_dict(row.get("mass_shape", ""))
        margins = self._parse_json_dict(row.get("mass_margin", ""))
        densities = self._parse_json_dict(row.get("mass_density", ""))

        if not shapes:
            return []

        descriptions = []
        for key in shapes.keys():
            parts = []
            if key in shapes:
                parts.append(shapes[key].lower())
            if key in margins:
                parts.append(margins[key].lower())
            if key in densities:
                parts.append(f"{densities[key].lower()} density")
            if parts:
                descriptions.append(" ".join(parts) + " mass")

        return descriptions

    def _build_asymmetry_description(self, row) -> Optional[str]:
        """Build asymmetry description based on type."""
        asymmetry = str(row.get("asymmetry", "")).lower()
        if asymmetry != "yes":
            return None

        asymmetry_type = self._parse_json_list(row.get("asymmetry_type", ""))
        if not asymmetry_type:
            return "asymmetry"

        # Check for focal
        for t in asymmetry_type:
            if "focal" in str(t).lower():
                return "focal asymmetry"

        return "asymmetry"

    def _build_associated_features_descriptions(self, row) -> List[str]:
        """Build descriptions for associated features."""
        features = self._parse_json_list(row.get("associated_features", ""))
        if not features:
            return []

        descriptions = []
        for feature in features:
            feature_lower = str(feature).lower()
            if (
                "calcification" not in feature_lower
            ):  # calcifications handled separately
                descriptions.append(f"{feature_lower} associated with a mass")

        return descriptions

    def _build_calcification_description(self, row) -> Optional[str]:
        """Build calcification description: morphology + subtype (if suspicious) + distribution."""
        # Check if calcifications in associated_features
        assoc_features = self._parse_json_list(row.get("associated_features", ""))
        has_assoc_calc = any("calcification" in str(f).lower() for f in assoc_features)

        calc = str(row.get("calcifications", "")).lower()
        if calc != "yes" and not has_assoc_calc:
            return None

        parts = []

        # Morphology
        morphology = str(row.get("calcification_morphology", "")).strip()
        if morphology and not pd.isna(row.get("calcification_morphology")):
            parts.append(morphology.lower())

            # Subtype only if suspicious
            if "suspicious" in morphology.lower():
                subtype = self._parse_json_list(
                    row.get("calcification_morphology_subtype", "")
                )
                if subtype:
                    parts.append(", ".join(s.lower() for s in subtype))

        # Distribution
        distribution = str(row.get("calcifications_distribution", "")).strip()
        if distribution and not pd.isna(row.get("calcifications_distribution")):
            parts.append(f"{distribution.lower()} distribution")

        if parts:
            return " ".join(parts) + " calcifications"
        elif has_assoc_calc:
            return "calcifications associated with a mass"

        return "calcifications"

    def build_caption(self, idx: int) -> str:
        row = self.df.iloc[idx]
        lat = self._norm_lat(row[self.laterality_col])

        findings = []

        # Masses
        mass_descs = self._build_mass_descriptions(row)

        findings.extend(mass_descs)

        # Asymmetry
        asymmetry_desc = self._build_asymmetry_description(row)

        if asymmetry_desc:
            findings.append(asymmetry_desc)

        # Architectural distortion
        if str(row.get("architectural_distortion", "")).lower() == "yes":
            findings.append("architectural distortion")

        # Associated features (excluding calcifications)
        assoc_descs = self._build_associated_features_descriptions(row)

        findings.extend(assoc_descs)

        # Calcifications
        calc_desc = self._build_calcification_description(row)

        if calc_desc:
            findings.append(calc_desc)

        if len(findings) == 0:
            caption = lat + ": No findings."
        else:
            caption = lat + ". Findings: " + "; ".join(findings) + "."

        # Append breast density
        density_val = str(row.get("breast_density", "")).strip().upper()
        density_map = {
            "A": "fatty",
            "B": "scattered fibroglandular",
            "C": "heterogeneously dense",
            "D": "extremely dense",
        }
        if density_val in density_map:
            caption += f" Breast density: {density_map[density_val]}."

        return caption

    def build_concept_labels(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        return torch.tensor(
            [int(bool(row[k])) for k in self.concept_keys], dtype=torch.float32
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        caption = self.build_caption(idx)
        concept_labels = self.build_concept_labels(idx)

        # Add density flags
        row = self.df.iloc[idx]
        density_flags = {k: int(row[k]) for k in self.DENSITY_COLUMNS}

        sample.update(
            {"caption": caption, "concept_labels": concept_labels, **density_flags}
        )
        return sample
