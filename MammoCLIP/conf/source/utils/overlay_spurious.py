import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from source.utils.misc import hash_prob
from source.utils.artifacts import (
    add_collimator_misalignment,
    add_grid_misplacement,
    add_thin_breast_corner_artifact,
    _get_processing_bars_patch,
    overlay_patch_in_right_corner,
)


def get_applied_overlay_pct(ds: Dataset, overlay_cfg: DictConfig) -> None:
    applied = 0
    N = len(ds)
    print(f"{overlay_cfg["mode"]=}")
    for i in range(N):
        uid = str(ds.df.iloc[i]["image_id"])
        p = ds.overlay_cfg["percent"]
        applied += hash_prob(uid, overlay_cfg["seed"]) <= p
    print(f"Expected overlay rate ≈ {applied/N:.3f}")


def log_overlay_stats_per_class(
    ds: Dataset,
    split_name: str = "dataset",
    uid_col: str = "image_id",
) -> None:
    import logging

    if not hasattr(ds, "_overlay_mask") or ds._overlay_mask is None:
        logging.info(f"[{split_name.upper()}] Overlay disabled or 0%")
        return

    overlay_mask = ds._overlay_mask
    labels = np.array(ds.labels, dtype=int)
    classes = sorted(set(labels.tolist()))
    uids = [str(ds.df.iloc[i][uid_col]) for i in range(len(ds))]

    percent = float(ds.overlay_cfg.get("percent", 0))
    mode = ds.overlay_cfg.get("mode", "same")
    # ---- CHANGED: respect skip_class_zero config ----
    skip_class_zero = ds.overlay_cfg.get("artifacts", {}).get("skip_class_zero", True)

    logging.info(
        f"[{split_name.upper()}] Overlay stats (mode={mode}, target={percent*100:.0f}%):"
    )

    for c in classes:
        class_uids = [uids[i] for i in range(len(uids)) if labels[i] == c]
        n_class = len(class_uids)

        if skip_class_zero and c == 0:
            n_overlayed = 0
        else:
            n_overlayed = sum(1 for uid in class_uids if uid in overlay_mask)

        pct = (n_overlayed / n_class * 100) if n_class > 0 else 0
        logging.info(f"  Class {c}: {n_overlayed}/{n_class} = {pct:.1f}%")

    # Overall
    if skip_class_zero:
        relevant_uids = [uids[i] for i in range(len(uids)) if labels[i] != 0]
    else:
        relevant_uids = uids
    n_total = len(relevant_uids)
    n_overlayed_total = sum(1 for uid in relevant_uids if uid in overlay_mask)
    overall_pct = (n_overlayed_total / n_total * 100) if n_total > 0 else 0
    label = "all classes" if not skip_class_zero else "excl. class 0"
    logging.info(
        f"  Overall ({label}): {n_overlayed_total}/{n_total} = {overall_pct:.1f}%"
    )


def precompute_balanced_overlay_mask(
    uids: list[str],
    labels: list[int],
    percent: float,
    seed: int = 0,
    skip_class: int | None = 0,  # <-- changed: None means overlay ALL classes
) -> set[str]:
    """
    Precompute which images should receive overlays, ensuring each class
    (except skip_class, if set) has exactly the same percentage of overlays.

    Args:
        uids: List of unique image identifiers
        labels: List of class labels (same length as uids)
        percent: Target overlay percentage (0.0 to 1.0)
        seed: Random seed for reproducibility
        skip_class: Class that never receives overlays. Set to None to
                     overlay all classes (e.g., binary mammography).

    Returns:
        Set of uids that should receive overlays
    """
    percent = _normalize_percent(percent)
    if percent <= 0:
        return set()

    uids = np.array(uids)
    labels = np.array(labels)

    classes = sorted(set(labels.tolist()))
    overlay_uids = set()

    for c in classes:
        if skip_class is not None and c == skip_class:  # <-- changed
            continue

        class_mask = labels == c
        class_indices = np.where(class_mask)[0]
        class_uids = uids[class_indices]
        n_class = len(class_uids)

        if n_class == 0:
            continue

        hash_vals = np.array([hash_prob(uid, seed) for uid in class_uids])
        sorted_indices = np.argsort(hash_vals)
        n_select = max(1, int(round(percent * n_class)))
        n_select = min(n_select, n_class)

        selected_uids = class_uids[sorted_indices[:n_select]]
        overlay_uids.update(selected_uids.tolist())

    return overlay_uids


def _normalize_percent(p):
    p = float(p)
    return p / 100.0 if p > 1.0 else p


def _choose_index(mode: str, class_idx: int, n: int) -> int:
    mode = (mode or "same").lower()
    if n <= 0:
        return 0
    if mode == "same":
        return class_idx % n
    if mode == "inverted":
        if n == 2:
            # Binary case: simple swap (0->1, 1->0)
            return (class_idx + 1) % n
        else:
            # Multi-class: original shift
            shift = (n // 2) + 1
            return (class_idx + shift) % n
    # "none" handled by caller
    return class_idx % n


def overlay_spurious(
    img_u8: np.ndarray,
    class_idx: int,
    uid: str,
    cfg: dict,
    *,
    laterality: str | None = None,
    overlay_mask: set[str] | None = None,
):
    """
    Unified overlay that can draw realistic artifacts.
    When skip_class_zero is False (e.g., binary mammography),
    class 0 also receives artifacts.
    Returns: (out_img, applied(bool), tag(str), meta/val)
    """
    # Quick exits
    if cfg.get("mode", "same").lower() == "none":
        return img_u8, False, None, None

    # Decide whether to apply overlay
    if overlay_mask is not None:
        should_overlay = uid in overlay_mask
    else:
        p = _normalize_percent(cfg.get("percent", 0.0))
        should_overlay = hash_prob(uid, cfg.get("seed", 0)) <= p

    if not should_overlay:
        return img_u8, False, None, None

    art_cfg = cfg.get("artifacts", {})
    if art_cfg.get("enabled", False):
        # ---- CHANGED: configurable class 0 skip ----
        skip_class_zero = art_cfg.get("skip_class_zero", True)
        if skip_class_zero and class_idx == 0:
            return img_u8, False, None, None
        # ---- END CHANGE ----

        kinds = art_cfg.get(
            "kinds",
            [
                "collimator",
                "grid_misplacement",
                "thin_breast_corner",
                "processing_bars",
            ],
        )
        if not kinds:
            return img_u8, False, None, None

        idx = _choose_index(cfg.get("mode", "same"), class_idx, len(kinds))
        kind = kinds[idx]

        seed = int(art_cfg.get("seed", cfg.get("seed", 0)))
        target_size = art_cfg.get("model_input_size", None)
        _PATCH_CACHE = {}

        if kind == "collimator":
            out, meta = add_collimator_misalignment(
                img_u8,
                uid=uid,
                seed=seed,
                laterality=laterality,
                target_size=target_size,
                width_frac=tuple(art_cfg.get("width_frac", (0.03, 0.06))),
                offset_frac=tuple(art_cfg.get("offset_frac", (0.35, 0.55))),
                top_trim_frac=tuple(art_cfg.get("top_trim_frac", (0.02, 0.10))),
                bot_trim_frac=tuple(art_cfg.get("bot_trim_frac", (0.02, 0.10))),
                intensity=int(art_cfg.get("intensity", 250)),
                opacity=float(art_cfg.get("opacity", 0.9)),
                feather_px=int(art_cfg.get("feather_px", 21)),
                taper_px=int(art_cfg.get("taper_px", 25)),
            )
            return out, True, "artifact:collimator", meta

        elif kind == "grid_misplacement":
            out, meta = add_grid_misplacement(
                img_u8,
                uid=uid,
                seed=seed,
                laterality=laterality,
                side=art_cfg.get("side", None),
                panel_frac=tuple(art_cfg.get("panel_frac", (0.07, 0.14))),
                panel_intensity=tuple(art_cfg.get("panel_intensity", (110, 170))),
                panel_opacity=float(art_cfg.get("panel_opacity", 0.85)),
                vignette_strength=float(art_cfg.get("vignette_strength", 0.25)),
                seam_intensity=int(art_cfg.get("seam_intensity", 245)),
                seam_width_px=int(art_cfg.get("seam_width_px", 3)),
                feather_px=int(art_cfg.get("feather_px", 11)),
                add_grid_texture=bool(art_cfg.get("add_grid_texture", True)),
                texture_period_px=tuple(art_cfg.get("texture_period_px", (18, 28))),
                texture_contrast=float(art_cfg.get("texture_contrast", 0.05)),
            )
            return out, True, "artifact:grid_misplacement", meta

        elif kind == "thin_breast_corner":
            out, meta = add_thin_breast_corner_artifact(
                img_u8,
                uid=uid,
                seed=seed,
                corner_frac=float(art_cfg.get("corner_frac", 0.20)),
                corner_depth_frac=float(art_cfg.get("corner_depth_frac", 0.15)),
                intensity=int(art_cfg.get("intensity", 255)),
                opacity=float(art_cfg.get("opacity", 1.0)),
                blur_ksize=int(art_cfg.get("blur_ksize", 10)),
                roundness=float(art_cfg.get("roundness", 0.25)),
                target_size=target_size,
            )
            return out, True, "artifact:thin_breast_corner", meta

        elif kind == "processing_bars":
            patch = _get_processing_bars_patch(_PATCH_CACHE, art_cfg)
            out, meta = overlay_patch_in_right_corner(
                img_u8,
                patch,
                uid,
                seed=art_cfg.get("seed", cfg.get("seed", 0)),
                corner=art_cfg.get("corner", "auto"),
                top_prob=float(art_cfg.get("top_prob", 0.6)),
                right_margin_px=int(art_cfg.get("right_margin_px", 18)),
                top_margin_px=int(art_cfg.get("top_margin_px", 18)),
                bottom_margin_px=int(art_cfg.get("bottom_margin_px", 18)),
                alpha_softness_px=int(art_cfg.get("alpha_softness_px", 3)),
                opacity=float(art_cfg.get("opacity", 1.0)),
            )
            return out, True, "artifact:processing_bars_patch", meta
        else:
            return img_u8, False, None, None
