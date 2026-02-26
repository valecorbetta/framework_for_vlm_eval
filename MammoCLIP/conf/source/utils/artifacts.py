import math
import numpy as np
import cv2

from source.utils.misc import hash_prob


def _breast_mask(img: np.ndarray) -> np.ndarray:
    # Otsu to get tissue, keep largest component, smooth
    thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (img > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    num, lab = cv2.connectedComponents(m)
    if num > 1:
        areas = [(lab == i).sum() for i in range(1, num)]
        keep = 1 + int(np.argmax(areas))
        m = (lab == keep).astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (31, 31), 0)
    return (m > 127).astype(np.uint8)


def _chest_wall_edge_x(img: np.ndarray) -> tuple[int, str]:
    """
    Returns (x_edge, side) where side is 'left' or 'right' chest-wall edge.
    Heuristic: the chest wall is the breast boundary that lies closest to an image border.
    """
    bm = _breast_mask(img)
    H, W = bm.shape
    if bm.sum() == 0:
        return W // 2, "center"

    # left boundary: first foreground col; right boundary: last foreground col
    cols = bm.sum(axis=0)
    # guard against empty leading/trailing columns
    x_left = int(np.argmax(cols > 0))
    x_right = W - int(np.argmax((cols[::-1] > 0))) - 1

    gap_left = x_left  # distance to left border
    gap_right = (W - 1) - x_right  # distance to right border

    if gap_left < gap_right:
        return x_left, "left"
    else:
        return x_right, "right"


def add_collimator_misalignment(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    laterality: str | None = None,  # "L" or "R" if you want side-aware placement
    target_size: (
        tuple[int, int] | None
    ) = None,  # e.g., (456, 456); draw after resize if set
    width_frac: tuple[float, float] = (
        0.01,
        0.03,
    ),  # band width as fraction of image width
    offset_frac: tuple[float, float] = (0.35, 0.55),  # x-position as fraction of width
    top_trim_frac: tuple[float, float] = (0.02, 0.10),  # trim at top
    bot_trim_frac: tuple[float, float] = (0.02, 0.10),  # trim at bottom
    intensity: int = 250,  # bright line
    opacity: float = 0.9,  # alpha for blending
    feather_px: int = 21,  # lateral feather (odd)
    taper_px: int = 25,  # taper edges vertically (rounded ends)
) -> tuple[np.ndarray, dict]:
    """
    Add a bright, solid vertical band (collimator misalignment).
    - img: grayscale uint8 (H, W)
    - laterality: if provided, uses side-aware default placement (see below)
    - width_frac: band width range relative to W
    - offset_frac: x position relative to W (if laterality is None)
    Returns (out_img_uint8, meta)
    """
    assert img.ndim == 2 and img.dtype == np.uint8, "expect single-channel uint8"
    H0, W0 = img.shape[:2]
    im = img

    resized = False
    if target_size is not None:
        Wt, Ht = target_size
        im = cv2.resize(im, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        H, W = Ht, Wt
        resized = True
    else:
        H, W = H0, W0

    rng = hash_prob(uid, seed)
    rng2 = hash_prob(uid, seed + 101)
    rng3 = hash_prob(uid, seed + 202)
    rng4 = hash_prob(uid, seed + 303)

    # width in px
    w_frac = width_frac[0] + (width_frac[1] - width_frac[0]) * rng
    band_w = max(2, int(round(W * w_frac)))

    # choose x position:
    if laterality is not None:
        # (kept) side rule if you *do* trust metadata
        if laterality.upper().startswith("R"):
            x_center = int(W * (0.35 + 0.1 * rng2))  # 0.35–0.45*W
        else:
            x_center = int(W * (0.65 - 0.1 * rng2))  # 0.55–0.65*W
    else:
        # *** place band at chest-wall edge (opposite nipple) ***
        x_edge, edge_side = _chest_wall_edge_x(
            im
        )  # im is the (optionally resized) image
        # put the band just inside the field, hugging the chest wall
        inward = max(2, int(0.01 * W))  # a few pixels into the image
        if edge_side == "left":
            x_center = np.clip(x_edge + inward + band_w // 2, 0, W - 1)
        elif edge_side == "right":
            x_center = np.clip(x_edge - inward - band_w // 2, 0, W - 1)
        else:
            # fallback if mask failed
            x_center = int(W * (0.50 + 0.05 * (2 * rng2 - 1)))

    # vertical trims & taper
    t_trim = int(
        round(H * (top_trim_frac[0] + (top_trim_frac[1] - top_trim_frac[0]) * rng3))
    )
    b_trim = int(
        round(H * (bot_trim_frac[0] + (bot_trim_frac[1] - bot_trim_frac[0]) * rng4))
    )
    y0, y1 = max(0, t_trim), min(H, H - b_trim)

    # build float mask [0..1]
    mask = np.zeros((H, W), dtype=np.float32)
    x0 = max(0, x_center - band_w // 2)
    x1 = min(W, x_center + (band_w - band_w // 2))

    # main rectangular shaft
    mask[y0:y1, x0:x1] = 1.0

    # rounded/tapered ends
    if taper_px > 0:
        k = max(3, taper_px | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # lateral feather to soften edges
    if feather_px > 0:
        kf = max(3, feather_px | 1)
        mask = cv2.GaussianBlur(mask, (kf, kf), 0)

    alpha = np.clip(mask * float(opacity), 0.0, 1.0)
    out = (alpha * float(intensity) + (1.0 - alpha) * im.astype(np.float32)).astype(
        np.uint8
    )

    if resized:
        out = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)

    meta = dict(
        band_w_px=band_w,
        x_center=x_center,
        y_top=y0,
        y_bot=y1,
        laterality=laterality,
        resized=resized,
        intensity=intensity,
        opacity=opacity,
    )
    return out, meta


def add_grid_misplacement(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    laterality: str | None = None,  # "L" or "R" if available
    side: str | None = None,  # override: "left" or "right"
    panel_frac: tuple[float, float] = (0.07, 0.14),  # panel width as frac of W
    panel_intensity: tuple[int, int] = (110, 170),  # gray value of panel (uint8)
    panel_opacity: float = 0.85,  # blend factor
    vignette_strength: float = 0.25,  # vertical gradient inside panel (0..1)
    seam_intensity: int = 245,  # bright seam line
    seam_width_px: int = 3,  # seam thickness
    feather_px: int = 11,  # soften transition to field
    add_grid_texture: bool = True,  # faint vertical ribs inside panel
    texture_period_px: tuple[int, int] = (18, 28),
    texture_contrast: float = 0.05,  # 0..0.2 recommended
) -> tuple[np.ndarray, dict]:
    """
    Simulate a 'grid not fully inserted' side panel:
      - uniform grey side panel with subtle vertical gradient and ribs
      - sharp/bright seam line against the image field
    Expects grayscale uint8 HxW input. Returns (img_uint8, meta).
    """
    assert img.ndim == 2 and img.dtype == np.uint8, "expect single-channel uint8"
    H, W = img.shape

    # ----- choose side deterministically -----
    if side is None:
        if laterality is not None:
            # Heuristic: for RMLO, panel tends to appear on the medial (left image) side;
            # flip if needed for your dataset; this keeps side consistent with anatomy.
            if laterality.upper().startswith("R"):
                side = "left"
            else:
                side = "right"
        else:
            side = "left" if hash_prob(uid, seed) < 0.5 else "right"

    # width and panel tone
    w_frac = panel_frac[0] + (panel_frac[1] - panel_frac[0]) * hash_prob(uid, seed + 11)
    panel_w = max(8, int(round(W * w_frac)))
    tone = int(
        round(
            panel_intensity[0]
            + (panel_intensity[1] - panel_intensity[0]) * hash_prob(uid, seed + 22)
        )
    )
    period = int(
        round(
            texture_period_px[0]
            + (texture_period_px[1] - texture_period_px[0]) * hash_prob(uid, seed + 33)
        )
    )
    period = max(6, period)

    # panel region coords
    if side == "left":
        x0, x1 = 0, min(W, panel_w)
        seam_x = x1
    else:
        x0, x1 = max(0, W - panel_w), W
        seam_x = x0

    # base panel mask (float 0..1)
    panel_mask = np.zeros((H, W), dtype=np.float32)
    panel_mask[:, x0:x1] = 1.0

    # vertical vignette (brighter center, darker top/bottom or vice versa)
    if vignette_strength > 0:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        # cosine ramp: 0 at ends, 1 at center
        vert = (1 - np.cos(2 * np.pi * (yy - 0.5))) * 0.5
        vert = (1 - vignette_strength) + vignette_strength * vert
        panel_mask *= vert

    # soft feather to avoid overly sharp rectangle into field
    if feather_px > 0:
        k = feather_px if feather_px % 2 == 1 else feather_px + 1
        panel_mask = cv2.GaussianBlur(panel_mask, (k, k), 0)

    # faint vertical grid ribs inside the panel
    texture = np.ones((H, W), dtype=np.float32)
    if add_grid_texture and period > 0 and texture_contrast > 0:
        # create sinusoidal vertical modulation only inside panel
        xx = np.arange(W, dtype=np.float32)[None, :]
        rib = 1.0 + texture_contrast * np.sin(2 * np.pi * xx / float(period))
        texture *= rib
        # clip to panel area
        texture = 1.0 + (texture - 1.0) * (panel_mask > 0.01).astype(np.float32)

    # compose panel onto image (bright-ish additive/alpha blend)
    imf = img.astype(np.float32)
    panel_img = np.full((H, W), tone, dtype=np.float32) * texture
    alpha = np.clip(panel_mask * panel_opacity, 0.0, 1.0)
    out = (alpha * panel_img + (1.0 - alpha) * imf).astype(np.float32)

    # draw bright seam at the contact edge
    if seam_width_px > 0:
        ww = max(1, int(seam_width_px))
        if side == "left":
            x_s0 = np.clip(seam_x - ww // 2, 0, W - 1)
            x_s1 = np.clip(seam_x + (ww - ww // 2), 0, W)
        else:
            x_s0 = np.clip(seam_x - ww // 2, 0, W - 1)
            x_s1 = np.clip(seam_x + (ww - ww // 2), 0, W)
        out[:, x_s0:x_s1] = np.maximum(out[:, x_s0:x_s1], float(seam_intensity))

        # a very light lateral blur to prevent a razor-thin digital edge
        out = cv2.GaussianBlur(out, (3, 3), 0)

    out = np.clip(out, 0, 255).astype(np.uint8)

    meta = dict(
        side=side,
        panel_w_px=panel_w,
        tone=tone,
        seam_x=seam_x,
        period_px=period if add_grid_texture else 0,
    )
    return out, meta


def add_thin_breast_corner_artifact(
    img: np.ndarray,
    uid: str,
    seed: int,
    corner_frac: float = 0.5,
    corner_depth_frac: float = 0.10,
    intensity: int = 255,
    opacity: float = 1.0,
    blur_ksize: int = 10,
    roundness: float = 0.30,
    target_size: tuple[int, int] | None = None,
    fill_solid: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Draw 'thin breast' corner artifacts in 3 of 4 corners.

    Args
    ----
    img : np.uint8 HxW (grayscale)
    uid, seed : used for deterministic corner choice and minor randomness
    corner_frac : size of the corner triangle relative to min(H,W)
    corner_depth_frac : thickness of the 'band' when fill_solid=False
    intensity : additive target intensity (0..255) blended by `opacity`
    opacity : final alpha for blending (0..1)
    blur_ksize : Gaussian blur (odd) to soften edges
    roundness : 0..0.5, larger = more rounded triangle tips
    target_size : (W,H). If set, draw at this size then resize back
    fill_solid : True -> solid white triangle; False -> thin wedge band

    Returns
    -------
    (img_out_uint8, meta_dict)
    """
    assert img.ndim == 2 and img.dtype == np.uint8, "expected single-channel uint8"
    H0, W0 = img.shape
    im = img.copy()

    # Optionally draw at model input size
    resized = False
    if target_size is not None:
        Wt, Ht = target_size
        im = cv2.resize(im, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        H, W = Ht, Wt
        resized = True
    else:
        H, W = H0, W0

    base = min(H, W)
    corner_sz = max(2, int(round(base * float(corner_frac))))
    corner_depth = max(2, int(round(base * float(corner_depth_frac))))

    # Deterministic choice: omit exactly one corner
    p = hash_prob(uid, seed)  # ∈[0,1)
    a = (int((p * 1e9) % 2**31) * 1103515245 + 12345) & 0x7FFFFFFF
    rng_val = a / 2**31
    omit_corner = int(math.floor(rng_val * 4))  # 0..3
    chosen = [c for c in range(4) if c != omit_corner]

    mask = np.zeros((H, W), dtype=np.float32)

    for c in chosen:
        if c == 0:  # top-left
            tri = np.array([[0, 0], [corner_sz, 0], [0, corner_sz]], np.int32)
        elif c == 1:  # top-right
            tri = np.array(
                [[W - 1, 0], [W - 1 - corner_sz, 0], [W - 1, corner_sz]], np.int32
            )
        elif c == 2:  # bottom-right
            tri = np.array(
                [
                    [W - 1, H - 1],
                    [W - 1 - corner_sz, H - 1],
                    [W - 1, H - 1 - corner_sz],
                ],
                np.int32,
            )
        else:  # bottom-left
            tri = np.array(
                [[0, H - 1], [corner_sz, H - 1], [0, H - 1 - corner_sz]], np.int32
            )

        tmp = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(tmp, tri, 255)

        if fill_solid:
            tri_mask = tmp.astype(np.float32) / 255.0
        else:
            # Thin wedge band near the corner by erode-then-subtract
            se = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, corner_depth), max(3, corner_depth))
            )
            band = tmp - cv2.erode(tmp, se)
            tri_mask = band.astype(np.float32) / 255.0

        # Optional rounding of outer tips (keeps fill)
        round_sz = max(1, int(round(base * roundness)))
        if round_sz > 1:
            smooth = cv2.GaussianBlur(tmp, (round_sz | 1, round_sz | 1), 0)
            _, smooth_bin = cv2.threshold(smooth, 10, 255, cv2.THRESH_BINARY)
            tri_mask *= smooth_bin.astype(np.float32) / 255.0

        mask += tri_mask

    # Soft edges so it blends like the figure
    mask = np.clip(mask, 0.0, 1.0)
    k = blur_ksize if (blur_ksize % 2 == 1) else blur_ksize + 1
    if k >= 3:
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = np.clip(mask * float(opacity), 0.0, 1.0)
    out = (alpha * float(intensity) + (1.0 - alpha) * im.astype(np.float32)).astype(
        np.uint8
    )

    if resized:
        out = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)

    meta = {
        "chosen_corners": chosen,
        "omit_corner": omit_corner,
        "corner_sz_px": corner_sz,
        "corner_depth_px": corner_depth,
        "intensity": intensity,
        "opacity": opacity,
        "blur_ksize": blur_ksize,
        "roundness": roundness,
        "fill_solid": fill_solid,
        "resized": resized,
    }
    return out, meta


def _get_processing_bars_patch(_PATCH_CACHE, art_cfg):
    # read params (with safe defaults)
    Wp = int(art_cfg.get("patch_W", 300))
    Hp = int(art_cfg.get("patch_H", 190))
    bws = tuple(art_cfg.get("bar_widths", (34, 46, 34, 34)))
    gaps = tuple(art_cfg.get("gaps", (0, 5, 0)))
    base = int(art_cfg.get("base_intensity", 60))
    bright = int(art_cfg.get("bright_intensity", 120))
    soft = int(art_cfg.get("softness_px", 5))
    vprof = bool(art_cfg.get("add_v_profile", True))
    jitter_frac = float(art_cfg.get("height_jitter_frac", 0.50))

    key = (Wp, Hp, bws, gaps, base, bright, soft, vprof)
    if key not in _PATCH_CACHE:
        _PATCH_CACHE[key] = make_processing_bars_patch(
            W_patch=Wp,
            H_patch=Hp,
            bar_widths=bws,
            gaps=gaps,
            base_intensity=base,
            bright_intensity=bright,
            softness_px=soft,
            add_v_profile=vprof,
            height_jitter_frac=jitter_frac,
        )
    return _PATCH_CACHE[key]


def make_processing_bars_patch(
    W_patch: int = 280,
    H_patch: int = 160,
    *,
    bar_widths=(26, 36, 26, 26),
    gaps=(22, 28, 22),
    base_intensity: int = 60,
    bright_intensity: int = 120,
    softness_px: int = 5,
    add_v_profile: bool = True,
    height_jitter_frac: float = 0.25,  # NEW: how much each bar’s height can vary
) -> np.ndarray:
    """
    Returns a fixed uint8 patch (H_patch x W_patch) with vertical bars of different heights.
    """
    n_bars = len(bar_widths)
    assert len(gaps) == n_bars - 1, "gaps must be length n_bars-1"

    mask = np.zeros((H_patch, W_patch), dtype=np.float32)
    add = np.zeros((H_patch, W_patch), dtype=np.float32)

    total_w = int(sum(bar_widths) + sum(gaps))
    x0 = max(0, (W_patch - total_w) // 2)

    # We'll center vertically, but shorten each bar a bit randomly
    rng = np.random.RandomState(12345)  # fixed for reproducibility

    for i, bw in enumerate(bar_widths):
        # deterministic per-bar jitter in height
        jitter = (rng.rand() * 2 - 1) * height_jitter_frac  # ±fraction
        bar_h = int(round(H_patch * (1.0 - abs(jitter))))
        y0 = int((H_patch - bar_h) // 2)
        y1 = y0 + bar_h

        xL = int(round(x0))
        xR = int(round(x0 + bw))
        xL = np.clip(xL, 0, W_patch - 1)
        xR = np.clip(xR, 0, W_patch)

        if xR > xL:
            mask[y0:y1, xL:xR] = 1.0
            add[y0:y1, xL:xR] = float(bright_intensity if i == 1 else base_intensity)

        if i < len(gaps):
            x0 += bw + gaps[i]

    # Soften edges
    if softness_px > 0:
        k = max(3, softness_px | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # Optional vertical modulation
    if add_v_profile:
        yy = np.linspace(0, 1, H_patch, dtype=np.float32)[:, None]
        band = 1.0 - 0.20 * np.exp(-(((yy - 0.12) / 0.06) ** 2))
        undul = 1.0 + 0.15 * np.sin(2 * np.pi * (yy * 1.0))
        profile = np.clip(band * undul, 0.6, 1.4)
        add *= profile

    patch = np.clip(mask * add, 0, 255).astype(np.uint8)
    return patch


def overlay_patch_in_right_corner(
    img: np.ndarray,
    patch: np.ndarray,
    uid: str,
    *,
    seed: int = 0,
    corner: str = "auto",  # "upper-right" | "lower-right" | "auto"
    top_prob: float = 0.6,  # used when corner="auto"
    right_margin_px: int = 18,
    top_margin_px: int = 18,
    bottom_margin_px: int = 18,
    alpha_softness_px: int = 7,  # a bit of extra feathering post-placement
    opacity: float = 1.0,  # 0..1, global strength
) -> tuple[np.ndarray, dict]:
    """
    Paste the given patch at the right upper or lower corner with soft alpha.
    We *add* the patch intensities (no background mask), allowing slight leakage into breast.
    """
    assert img.ndim == 2 and img.dtype == np.uint8, "expect single-channel uint8"
    H, W = img.shape
    h, w = patch.shape[:2]

    # Decide corner
    if corner not in ("upper-right", "lower-right", "auto"):
        corner = "upper-right"
    if corner == "auto":
        try:
            p = hash_prob(uid, seed)  # your existing deterministic helper
        except NameError:
            import hashlib

            p = (
                int(hashlib.md5((str(uid) + str(seed)).encode()).hexdigest()[:8], 16)
                / 0xFFFFFFFF
            )
        corner = "upper-right" if p < float(top_prob) else "lower-right"

    # Compute placement rectangle
    x1 = max(0, W - right_margin_px)  # right edge of placement box
    x0 = max(0, x1 - w)  # left edge
    if corner == "upper-right":
        y0 = max(0, top_margin_px)
        y1 = min(H, y0 + h)
    else:
        y1 = max(0, H - bottom_margin_px)
        y0 = max(0, y1 - h)

    # Clip if patch bigger than available space
    px0 = 0
    py0 = 0
    px1 = w
    py1 = h
    if x0 < 0:
        px0 += -x0
        x0 = 0
    if y0 < 0:
        py0 += -y0
        y0 = 0
    if x1 > W:
        px1 -= x1 - W
        x1 = W
    if y1 > H:
        py1 -= y1 - H
        y1 = H
    if (x1 <= x0) or (y1 <= y0) or (px1 <= px0) or (py1 <= py0):
        return img, {"applied": False, "reason": "no_space"}

    # Build soft alpha matching the patch support
    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[y0:y1, x0:x1] = (patch[py0:py1, px0:px1].astype(np.float32) > 0).astype(
        np.float32
    )
    if alpha_softness_px > 0:
        k = max(3, alpha_softness_px | 1)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha * float(opacity), 0.0, 1.0)

    # Additive compose
    add = np.zeros((H, W), dtype=np.float32)
    add[y0:y1, x0:x1] = patch[py0:py1, px0:px1].astype(np.float32)

    out = np.clip(img.astype(np.float32) + alpha * add, 0, 255).astype(np.uint8)

    meta = {
        "applied": True,
        "corner": corner,
        "xywh": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        "patch_hw": (int(h), int(w)),
    }
    return out, meta
