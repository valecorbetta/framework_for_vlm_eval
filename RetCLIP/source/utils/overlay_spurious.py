import logging
from typing import Any
import numpy as np
import hashlib
import random
import cv2
from omegaconf import DictConfig
from torch.utils.data import Dataset
import pandas as pd

from RetCLIP.source.utils.shapes import build_texture_from_spec, draw_shape
from RetCLIP.source.utils.artifacts import (
    add_fundus_illumination_circle,
    add_fundus_out_of_focus_quarter,
    add_fundus_bluish_circle,
    add_fundus_reflection_double_dot,
    add_fundus_eyelash_shadow_band,
)
from RetCLIP.source.utils.misc import hash_prob


def get_applied_overlay_pct(ds: Dataset, overlay_cfg: DictConfig) -> None:
    applied = 0
    N = len(ds)
    print(f"{overlay_cfg["mode"]=}")
    for i in range(N):
        uid = str(ds.df.iloc[i]["image"])
        p = ds.overlay_cfg["percent"]
        applied += hash_prob(uid, overlay_cfg["seed"]) <= p
    print(f"Expected overlay rate ≈ {applied/N:.3f}")


def log_overlay_stats_per_class(
    ds: Dataset,
    split_name: str = "dataset",
    uid_col: str = "image",
) -> None:
    """
    Log per-class overlay statistics for a dataset using ds._overlay_mask.

    Supports both:
      - legacy percent-based overlays (balanced mask, skip class 0 convention)
      - waterbirds schedule (p_by_class / rho), where class 0 MAY have overlays
    """

    if not hasattr(ds, "_overlay_mask") or ds._overlay_mask is None:
        logging.info(f"[{split_name.upper()}] Overlay disabled or 0%")
        return

    overlay_mask = ds._overlay_mask
    labels = np.asarray(ds.labels, dtype=int)
    classes = sorted(set(labels.tolist()))
    uids = [str(ds.df.iloc[i][uid_col]) for i in range(len(ds))]

    mode = str(getattr(ds, "overlay_cfg", {}).get("mode", "same")).lower()

    # Detect waterbirds schedule
    has_wb = (
        mode == "waterbirds"
        or getattr(ds, "overlay_cfg", {}).get("p_by_class", None) is not None
        or getattr(ds, "overlay_cfg", {}).get("rho", None) is not None
    )

    if has_wb:
        p_by_class = getattr(ds, "overlay_cfg", {}).get("p_by_class", None)
        rho = getattr(ds, "overlay_cfg", {}).get("rho", None)
        target_str = (
            f"p_by_class={list(p_by_class)}" if p_by_class is not None else f"rho={rho}"
        )
    else:
        percent = float(getattr(ds, "overlay_cfg", {}).get("percent", 0.0))
        target_str = f"{percent*100:.0f}%"

    logging.info(
        f"[{split_name.upper()}] Overlay stats (mode={mode}, target={target_str}):"
    )

    for c in classes:
        idx_c = np.where(labels == c)[0]
        n_class = int(idx_c.size)
        if n_class == 0:
            logging.info(f"  Class {c}: 0/0 = 0.0%")
            continue

        class_uids = [uids[i] for i in idx_c]

        # Legacy convention: class 0 never gets overlays (ONLY if not waterbirds)
        if (not has_wb) and c == 0:
            n_overlayed = 0
        else:
            n_overlayed = sum(1 for uid in class_uids if uid in overlay_mask)

        pct = (n_overlayed / n_class * 100.0) if n_class > 0 else 0.0
        logging.info(f"  Class {c}: {n_overlayed}/{n_class} = {pct:.1f}%")

    # Overall (always print both; avoids confusion)
    n_total = len(uids)
    n_overlayed_total = sum(1 for uid in uids if uid in overlay_mask)
    overall_pct = (n_overlayed_total / n_total * 100.0) if n_total > 0 else 0.0
    logging.info(f"  Overall: {n_overlayed_total}/{n_total} = {overall_pct:.1f}%")

    # Optional: keep the excl-class-0 summary (useful for your legacy setting)
    non0 = np.where(labels != 0)[0]
    n_total_ex0 = int(non0.size)
    if n_total_ex0 > 0:
        non0_uids = [uids[i] for i in non0]
        n_overlayed_ex0 = sum(1 for uid in non0_uids if uid in overlay_mask)
        overall_pct_ex0 = n_overlayed_ex0 / n_total_ex0 * 100.0
        logging.info(
            f"  Overall (excl. class 0): {n_overlayed_ex0}/{n_total_ex0} = {overall_pct_ex0:.1f}%"
        )


def precompute_balanced_overlay_mask(
    uids: list[str],
    labels: list[int],
    percent: float,
    seed: int = 0,
    skip_class: int = 0,
) -> set[str]:
    """
    Precompute which images should receive overlays, ensuring each class
    (except skip_class) has exactly the same percentage of overlays.

    Args:
        uids: List of unique image identifiers
        labels: List of class labels (same length as uids)
        percent: Target overlay percentage (0.0 to 1.0)
        seed: Random seed for reproducibility
        skip_class: Class that never receives overlays (default: 0)

    Returns:
        Set of uids that should receive overlays
    """
    percent = _normalize_percent(percent)
    if percent <= 0:
        return set()

    uids = np.array(uids)
    labels = np.array(labels)

    # Group indices by class
    classes = sorted(set(labels.tolist()))
    overlay_uids = set()

    for c in classes:
        if c == skip_class:
            continue

        # Get indices for this class
        class_mask = labels == c
        class_indices = np.where(class_mask)[0]
        class_uids = uids[class_indices]
        n_class = len(class_uids)

        if n_class == 0:
            continue

        # Compute hash values for deterministic ordering
        hash_vals = np.array([hash_prob(uid, seed) for uid in class_uids])

        # Sort by hash value and select top percent
        sorted_indices = np.argsort(hash_vals)
        n_select = max(1, int(round(percent * n_class)))  # at least 1 if percent > 0
        n_select = min(n_select, n_class)  # cap at class size

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
        shift = (n // 2) + 1
        return (class_idx + shift) % n
    if mode == "random":
        return 0  # Always use the first artifact/shape
    # "none" handled by caller
    return class_idx % n


def rand_from_uid(uid: str, seed: int = 0) -> random.Random:
    h = hashlib.sha256((str(uid) + str(seed)).encode()).digest()
    return random.Random(h)


def random_center_in_retina(img_u8, shape_size, min_margin=0):
    """
    Returns (cx, cy) such that the shape of 'shape_size' fits entirely
    inside the non-black retina region.

    - img_u8: HxW or HxWx3 uint8 image
    - shape_size: same 'size' you pass to draw_shape (radius / half-side)
    - min_margin: extra safety margin (pixels) from the fundus border
    """
    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_u8

    H, W = gray.shape

    # 1) foreground mask (retina ≈ non-black)
    # tweak '10' if your background isn’t perfectly black
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 2) distance transform: how far each pixel is from the background
    # distance is 0 outside retina and along border, larger inside
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # we need pixels that have enough room for the shape
    required = shape_size + min_margin
    valid_yx = np.argwhere(dist >= required)

    if valid_yx.size == 0:
        # fallback: just use image center if something went wrong
        return [W // 2, H // 2]

    # 3) pick a random valid pixel
    iy = random.randint(0, len(valid_yx) - 1)
    y, x = valid_yx[iy]
    return [int(x), int(y)]


def retina_mask_from_img(img_u8: np.ndarray, thr: int = 10, erode_px: int = 1):
    """
    Returns a uint8 mask (H,W) with 255 on the retina region and 0 on background.

    - thr: threshold for "non-black" (tweak if your background isn't pure 0)
    - erode_px: how many pixels to erode to stay safely inside the retina
    """
    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_u8

    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

    if erode_px > 0:
        ksz = 2 * erode_px + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask = cv2.erode(mask, k, iterations=1)

    return mask  # 0/255


#### ----- Waterbird-correlations ----- ####
def compute_p_by_class_from_cfg(overlay_cfg: DictConfig, n_classes: int) -> list[float]:
    """
    Ordinal-aware schedule to simulate Waterbird style correlations in multiclass settings.
    """

    p_by_class = list(overlay_cfg["p_by_class"])
    return [float(max(0.0, min(1.0, p))) for p in p_by_class]


def precompute_waterbirds_overlay_mask(
    df: pd.DataFrame, labels: list[int], overlay_cfg: DictConfig, uid_col: str = "image"
) -> set[str] | None:
    """
    Deterministically select a subset of UIDs per class so that:
      P(a=1 | y=c) ~= p_by_class[c]
    where a=1 means "overlay applied".

    Selection is deterministic via hash_prob(uid, seed), so it is stable across
    dataloader workers and reproducible across runs.
    """

    if overlay_cfg["enabled"] == False:
        return None

    labels_np = np.asarray(labels, dtype=int)
    n_classes = int(labels_np.max()) + 1
    seed = int(overlay_cfg.get("seed", 0))
    p_by_class = compute_p_by_class_from_cfg(overlay_cfg, n_classes)

    uids = np.array([str(v) for v in df[uid_col].tolist()])
    overlay_uids: set[str] = set()

    for c in range(n_classes):
        idx_c = np.where(labels_np == c)[0]
        if idx_c.size == 0:
            continue

        uids_c = uids[idx_c]
        n_c = int(uids_c.size)
        p_c = float(p_by_class[c])

        # target count
        n_select = int(round(p_c * n_c))

        # Waterbirds-style requires both (y=c,a=0) and (y=c,a=1) to exist when possible.
        # Enforce at least 1 sample in each subgroup for non-degenerate classes.
        if n_c >= 2 and 0.0 < p_c < 1.0:
            n_select = max(1, min(n_select, n_c - 1))

        # deterministic ordering
        scores = np.array([hash_prob(uid, seed) for uid in uids_c])
        order = np.argsort(scores)

        selected = uids_c[order[:n_select]]
        overlay_uids.update(selected.tolist())

    return overlay_uids


def overlay_spurious(
    img_u8: np.ndarray,
    class_idx: int,
    uid: str,
    cfg: dict,
    *,
    overlay_mask: set[str] | None = None,  # precomputed set of uids to overlay
):
    """
    Apply spurious correlation overlay to an image.

    If overlay_mask is provided, it overrides the hash-based percent gate.
    This allows for balanced per-class overlay percentages.
    """
    if not cfg.get("enabled", False):
        return img_u8, False, None, None, None

    mode = cfg.get("mode", "same").lower()

    if mode == "none":
        return img_u8, False, None, None, None

    # Skip class 0 only for class-correlated modes (same/inverted)
    if mode != "random" and class_idx == 0:
        return img_u8, False, None, None, None

    # Decide whether to apply overlay
    if overlay_mask is not None:
        # Use precomputed mask (for balanced per-class percentages)
        should_overlay = uid in overlay_mask
    else:
        # Fall back to hash-based probabilistic gating
        p = _normalize_percent(cfg.get("percent", 0.0))
        should_overlay = hash_prob(uid, cfg.get("seed", 0)) <= p

    if not should_overlay:
        return img_u8, False, None, None, None

    if cfg.get("name") == "shapes":
        shapes = cfg.get("shapes", [])

        if not shapes:
            return img_u8, False, None, None, None

        idx = _choose_index(mode, class_idx, len(shapes))
        shape = shapes[idx]
        # is_rgb & value / color selection
        is_rgb = img_u8.ndim == 3 and img_u8.shape[-1] == 3
        prefer_c = cfg.get("prefer_colors_on_rgb", True)

        value = None  # used only in non-texture branch
        tex_color = None  # NEW: color used to tint textures
        contour_color = cfg.get("contour_color", 255)

        if is_rgb and prefer_c and ("colors" in cfg) and cfg["colors"]:
            colors = cfg["colors"]
            c = colors[idx % len(colors)]  # BGR
            tex_color = c
            value = c  # if we fall back to non-texture, use same color
            # If no explicit contour color is given, reuse the texture color
            if "contour_color" not in cfg:
                contour_color = c
        elif "intensities" in cfg and cfg["intensities"]:
            intensities = cfg["intensities"]
            value = int(np.clip(intensities[idx % len(intensities)], 0, 255))
        else:
            value = (255, 255, 255) if is_rgb else 255

        # placement & size (unchanged)
        H, W = img_u8.shape[:2]
        rng = rand_from_uid(uid, cfg.get("seed", 0))
        size_cfg = cfg.get("size_px", 24)
        if isinstance(size_cfg, (list, tuple)):
            if len(size_cfg) == 2:
                smin, smax = map(int, size_cfg)
                s = smin if smin == smax else rng.randint(max(2, smin), max(3, smax))
            elif len(size_cfg) == 1:
                s = int(size_cfg[0])
            else:
                raise ValueError("size_px must be an int or a (min, max) tuple")
        else:
            s = int(size_cfg)

        margin = int(cfg.get("margin", 12))
        m = margin + s

        out = img_u8.copy()
        alpha = float(cfg.get("alpha", 1.0))

        center = random_center_in_retina(out, shape_size=s, min_margin=1)

        if cfg.get("textures") is not None:
            seed_int = cfg.get("seed", 0)
            built = [
                build_texture_from_spec(spec, seed_int) for spec in cfg["textures"]
            ]

            # use the same idx that picked the shape (stronger coupling of shape/texture/color)
            used_texture = built[idx % len(built)]

            draw_shape(
                out,
                center,
                s,
                value=None,  # color is driven via texture_color instead
                shape=shape,
                alpha=alpha,
                texture=used_texture,
                blend_mode=cfg.get("blend_mode", "multiply"),
                tex_angle=float(cfg.get("tex_angle", 0.0)),
                tex_scale=float(cfg.get("tex_scale", 1.0)),
                tex_offset=tuple(cfg.get("tex_offset", (0, 0))),
                contour_thickness=int(cfg.get("contour_thickness", 5)),
                contour_color=contour_color,
                texture_color=tex_color,
            )
        else:
            draw_shape(
                out,
                center,
                s,
                value=value,
                shape=shape,
                alpha=alpha,
                contour_thickness=int(cfg.get("contour_thickness", 5)),
                contour_color=contour_color,
            )

        return out, True, shape, value

    elif cfg.get("name") == "artifacts":
        # which artifacts are allowed?
        kinds = cfg.get(
            "kinds",
            [
                "illumination_semicircle",
                "out_of_focus_quarter",
                "eyelash_shadow",
                "reflection_double_dot",
            ],
        )
        if not kinds:
            return img_u8, False, None, None, None

        idx = _choose_index(mode, class_idx, len(kinds))
        kind = kinds[idx]

        seed = cfg.get("seed", 0)

        # 1) apply the chosen artifact to a copy of the image
        art_img = img_u8.copy()

        if kind == "illumination_semicircle":
            art_img, meta, art_mask = add_fundus_illumination_circle(
                art_img,
                uid,
                seed=seed,
                # side=cfg.get("side", "auto"),
                color_bgr=tuple(cfg.get("color_bgr", (0, 230, 0))),
                strength=float(cfg.get("strength", 1.0)),
                radius_frac=tuple(cfg.get("radius_frac", (0.95, 1.05))),
                softness_frac=float(cfg.get("softness_frac", 0.25)),
                center_jitter_frac=float(cfg.get("center_jitter_frac", 0.12)),
            )

        elif kind == "out_of_focus_quarter":
            art_img, meta, art_mask = add_fundus_out_of_focus_quarter(
                art_img,
                uid,
                seed=seed,
                corner=cfg.get("corner", "auto"),
                radius_frac=tuple(cfg.get("radius_frac", (0.55, 0.85))),
                softness_frac=float(cfg.get("softness_frac", 0.25)),
                blur_ksize=int(cfg.get("blur_ksize", 51)),
                darkness=float(cfg.get("darkness", 0.65)),
                strength=float(cfg.get("strength", 0.95)),
            )

        # elif kind == "bluish_circle":
        #     art_img, meta, art_mask = add_fundus_bluish_circle(
        #         art_img,
        #         uid,
        #         seed=seed,
        #         radius_frac=tuple(cfg.get("radius_frac", (0.12, 0.30))),
        #         center_x_frac=tuple(cfg.get("center_x_frac", (0.30, 0.75))),
        #         center_y_frac=tuple(cfg.get("center_y_frac", (0.25, 0.75))),
        #         color_bgr=tuple(cfg.get("color_bgr", (255, 80, 30))),
        #         strength=float(cfg.get("strength", 0.85)),
        #         blur_ksize=int(cfg.get("blur_ksize", 19)),
        #         brighten=float(cfg.get("brighten", 1.10)),
        #     )

        elif kind == "reflection_double_dot":
            art_img, meta, art_mask = add_fundus_reflection_double_dot(
                art_img,
                uid,
                seed=seed,
                base_radius_frac=tuple(cfg.get("base_radius_frac", (0.055, 0.077))),
                size_ratio=tuple(cfg.get("size_ratio", (0.45, 0.75))),
                separation_frac=float(cfg.get("separation_frac", 2.0)),
                center_x_frac=tuple(cfg.get("center_x_frac", (0.35, 0.70))),
                center_y_frac=tuple(cfg.get("center_y_frac", (0.35, 0.65))),
                halo_strength=float(cfg.get("halo_strength", 0.9)),
                core_strength=float(cfg.get("core_strength", 1.0)),
                halo_color_bgr=tuple(cfg.get("halo_color_bgr", (255, 235, 200))),
                core_color_bgr=tuple(cfg.get("core_color_bgr", (255, 255, 255))),
                halo_softness=float(cfg.get("halo_softness", 1.8)),
                core_frac=float(cfg.get("core_frac", 0.35)),
            )

        elif kind == "eyelash_shadow":
            art_img, meta, art_mask = add_fundus_eyelash_shadow_band(
                art_img,
                uid,
                seed=seed,
                side=cfg.get("side", "auto"),
                band_frac=tuple(cfg.get("band_frac", (0.12, 0.22))),
                num_lashes=tuple(cfg.get("num_lashes", (14, 20))),
                thickness_px=tuple(cfg.get("thickness_px", (4, 10))),
                darkness=float(cfg.get("darkness", 0.45)),
                strength=float(cfg.get("strength", 0.85)),
                blur_ksize=int(cfg.get("blur_ksize", 31)),
            )

        else:
            # unknown kind
            return img_u8, False, None, None, None

        # 2) restrict artifact to retina only
        mask = retina_mask_from_img(
            img_u8,
            thr=cfg.get("retina_thr", 10),
            erode_px=int(cfg.get("retina_erode_px", 0)),
        )
        m = mask.astype(np.float32) / 255.0

        base_f = img_u8.astype(np.float32)
        art_f = art_img.astype(np.float32)

        if img_u8.ndim == 3:
            m = m[..., None]

        out_f = base_f * (1.0 - m) + art_f * m
        out_u8 = np.clip(out_f, 0, 255).astype(np.uint8)

        # 3) compute final artifact mask: intersection of artifact mask and retina mask
        # art_mask is HxW uint8 with 1s where artifact was applied
        # retina_mask is HxW uint8 with 255 inside retina, 0 outside
        # final_artifact_mask is HxW uint8 with 1s where artifact is actually visible
        final_artifact_mask = (art_mask & (mask > 0)).astype(np.uint8)

        return out_u8, True, kind, meta, final_artifact_mask

    else:
        raise ValueError(
            "Spurious correlation mode not supported, available modes: shapes and artifacts."
        )


def overlay_spurious(
    img_u8: np.ndarray,
    class_idx: int,
    uid: str,
    cfg: dict,
    *,
    overlay_mask: set[str] | None = None,  # precomputed set of uids to overlay
):
    """
    Apply a spurious correlation overlay to an image.

    Modes (cfg["mode"]):
      - "none": never apply
      - "same": (class-correlated) choose overlay index = class_idx
      - "inverted": (class-correlated) choose overlay index = reversed class_idx
      - "random": choose overlay index independent of class (but deterministic per uid/seed)
      - "waterbirds": like "random" for choosing the overlay type, but selection of
         which samples get overlaid MUST come from overlay_mask (precomputed via p_by_class / rho).
         This makes P(a=1 | y=c) controlled outside this function.

    Gating (deciding whether to apply):
      - If mode == "waterbirds": requires overlay_mask, uses uid ∈ overlay_mask
      - Else:
          - if overlay_mask provided: uid ∈ overlay_mask
          - else: hash_prob(uid, seed) <= percent

    Returns:
      - out_img_u8
      - spurious_applied (bool)
      - spurious_type (str | None)        # e.g., shape name or artifact kind
      - meta (Any | None)                 # artifact meta dict for artifacts; value/color for shapes
      - artifact_mask (np.ndarray | None) # HxW uint8 mask (1 where artifact visible), only for artifacts
    """
    if not cfg.get("enabled", False):
        return img_u8, False, None, None, None

    mode = str(cfg.get("mode", "same")).lower()

    if mode == "none":
        return img_u8, False, None, None, None

    # ----------------------------
    # Decide whether to apply overlay
    # ----------------------------
    if mode == "waterbirds":
        # Waterbirds selection is *always* via precomputed overlay_mask.
        if overlay_mask is None:
            raise ValueError(
                "overlay_spurious: mode='waterbirds' requires overlay_mask "
                "(precomputed from p_by_class / rho schedule)."
            )
        should_overlay = uid in overlay_mask
    else:
        # Backward compatible behavior:
        # - If overlay_mask is provided, it overrides percent gating.
        # - Else use hash-based percent gating.
        if overlay_mask is not None:
            should_overlay = uid in overlay_mask
        else:
            p = _normalize_percent(cfg.get("percent", 0.0))
            should_overlay = hash_prob(uid, cfg.get("seed", 0)) <= p

    if not should_overlay:
        return img_u8, False, None, None, None

    # Skip class 0 only for class-correlated modes (same/inverted).
    # Do NOT skip for random/waterbirds (those allow overlays in class 0 if scheduled).
    if mode in {"same", "inverted"} and class_idx == 0:
        return img_u8, False, None, None, None

    # For choosing overlay index, treat waterbirds the same as random:
    pick_mode = "random" if mode == "waterbirds" else mode

    # ----------------------------
    # Shapes overlays
    # ----------------------------
    if cfg.get("name") == "shapes":
        shapes = cfg.get("shapes", [])
        if not shapes:
            return img_u8, False, None, None, None

        idx = _choose_index(pick_mode, class_idx, len(shapes))
        shape = shapes[idx]

        is_rgb = img_u8.ndim == 3 and img_u8.shape[-1] == 3
        prefer_c = cfg.get("prefer_colors_on_rgb", True)

        value = None  # used only in non-texture branch
        tex_color = None  # used to tint textures
        contour_color = cfg.get("contour_color", 255)

        if is_rgb and prefer_c and ("colors" in cfg) and cfg["colors"]:
            colors = cfg["colors"]
            c = colors[idx % len(colors)]  # BGR
            tex_color = c
            value = c
            if "contour_color" not in cfg:
                contour_color = c
        elif "intensities" in cfg and cfg["intensities"]:
            intensities = cfg["intensities"]
            value = int(np.clip(intensities[idx % len(intensities)], 0, 255))
        else:
            value = (255, 255, 255) if is_rgb else 255

        H, W = img_u8.shape[:2]
        rng = rand_from_uid(uid, cfg.get("seed", 0))
        size_cfg = cfg.get("size_px", 24)

        if isinstance(size_cfg, (list, tuple)):
            if len(size_cfg) == 2:
                smin, smax = map(int, size_cfg)
                s = smin if smin == smax else rng.randint(max(2, smin), max(3, smax))
            elif len(size_cfg) == 1:
                s = int(size_cfg[0])
            else:
                raise ValueError("size_px must be an int or a (min, max) tuple")
        else:
            s = int(size_cfg)

        out = img_u8.copy()
        alpha = float(cfg.get("alpha", 1.0))

        center = random_center_in_retina(out, shape_size=s, min_margin=1)

        if cfg.get("textures") is not None:
            seed_int = cfg.get("seed", 0)
            built = [
                build_texture_from_spec(spec, seed_int) for spec in cfg["textures"]
            ]
            used_texture = built[idx % len(built)]

            draw_shape(
                out,
                center,
                s,
                value=None,
                shape=shape,
                alpha=alpha,
                texture=used_texture,
                blend_mode=cfg.get("blend_mode", "multiply"),
                tex_angle=float(cfg.get("tex_angle", 0.0)),
                tex_scale=float(cfg.get("tex_scale", 1.0)),
                tex_offset=tuple(cfg.get("tex_offset", (0, 0))),
                contour_thickness=int(cfg.get("contour_thickness", 5)),
                contour_color=contour_color,
                texture_color=tex_color,
            )
        else:
            draw_shape(
                out,
                center,
                s,
                value=value,
                shape=shape,
                alpha=alpha,
                contour_thickness=int(cfg.get("contour_thickness", 5)),
                contour_color=contour_color,
            )

        # Shapes branch: no artifact mask
        return out, True, shape, value, None

    # ----------------------------
    # Artifacts overlays (fundus)
    # ----------------------------
    elif cfg.get("name") == "artifacts":
        kinds = cfg.get(
            "kinds",
            [
                "illumination_semicircle",
                "out_of_focus_quarter",
                "eyelash_shadow",
                "reflection_double_dot",
            ],
        )
        if not kinds:
            return img_u8, False, None, None, None

        idx = _choose_index(pick_mode, class_idx, len(kinds))
        kind = kinds[idx]

        seed = cfg.get("seed", 0)

        art_img = img_u8.copy()
        meta: Any = None
        art_mask: np.ndarray | None = None

        if kind == "illumination_semicircle":
            art_img, meta, art_mask = add_fundus_illumination_circle(
                art_img,
                uid,
                seed=seed,
                color_bgr=tuple(cfg.get("color_bgr", (0, 230, 0))),
                strength=float(cfg.get("strength", 1.0)),
                radius_frac=tuple(cfg.get("radius_frac", (0.95, 1.05))),
                softness_frac=float(cfg.get("softness_frac", 0.25)),
                center_jitter_frac=float(cfg.get("center_jitter_frac", 0.12)),
            )

        elif kind == "out_of_focus_quarter":
            art_img, meta, art_mask = add_fundus_out_of_focus_quarter(
                art_img,
                uid,
                seed=seed,
                corner=cfg.get("corner", "auto"),
                radius_frac=tuple(cfg.get("radius_frac", (0.55, 0.85))),
                softness_frac=float(cfg.get("softness_frac", 0.25)),
                blur_ksize=int(cfg.get("blur_ksize", 51)),
                darkness=float(cfg.get("darkness", 0.65)),
                strength=float(cfg.get("strength", 0.95)),
            )

        elif kind == "reflection_double_dot":
            art_img, meta, art_mask = add_fundus_reflection_double_dot(
                art_img,
                uid,
                seed=seed,
                base_radius_frac=tuple(cfg.get("base_radius_frac", (0.055, 0.077))),
                size_ratio=tuple(cfg.get("size_ratio", (0.45, 0.75))),
                separation_frac=float(cfg.get("separation_frac", 2.0)),
                center_x_frac=tuple(cfg.get("center_x_frac", (0.35, 0.70))),
                center_y_frac=tuple(cfg.get("center_y_frac", (0.35, 0.65))),
                halo_strength=float(cfg.get("halo_strength", 0.9)),
                core_strength=float(cfg.get("core_strength", 1.0)),
                halo_color_bgr=tuple(cfg.get("halo_color_bgr", (255, 235, 200))),
                core_color_bgr=tuple(cfg.get("core_color_bgr", (255, 255, 255))),
                halo_softness=float(cfg.get("halo_softness", 1.8)),
                core_frac=float(cfg.get("core_frac", 0.35)),
            )

        elif kind == "eyelash_shadow":
            art_img, meta, art_mask = add_fundus_eyelash_shadow_band(
                art_img,
                uid,
                seed=seed,
                side=cfg.get("side", "auto"),
                band_frac=tuple(cfg.get("band_frac", (0.12, 0.22))),
                num_lashes=tuple(cfg.get("num_lashes", (14, 20))),
                thickness_px=tuple(cfg.get("thickness_px", (4, 10))),
                darkness=float(cfg.get("darkness", 0.45)),
                strength=float(cfg.get("strength", 0.85)),
                blur_ksize=int(cfg.get("blur_ksize", 31)),
            )

        else:
            return img_u8, False, None, None, None

        if art_mask is None:
            # safety: if an artifact function forgot to return a mask
            return img_u8, False, None, None, None

        # Restrict artifact to retina only
        retina = retina_mask_from_img(
            img_u8,
            thr=cfg.get("retina_thr", 10),
            erode_px=int(cfg.get("retina_erode_px", 0)),
        )
        m = retina.astype(np.float32) / 255.0

        base_f = img_u8.astype(np.float32)
        art_f = art_img.astype(np.float32)

        if img_u8.ndim == 3:
            m = m[..., None]

        out_f = base_f * (1.0 - m) + art_f * m
        out_u8 = np.clip(out_f, 0, 255).astype(np.uint8)

        # Final artifact mask: where artifact is visible AND within retina
        final_artifact_mask = (art_mask & (retina > 0)).astype(np.uint8)

        return out_u8, True, kind, meta, final_artifact_mask

    else:
        raise ValueError(
            "overlay_spurious: cfg['name'] must be one of {'shapes','artifacts'}"
        )
