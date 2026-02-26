import numpy as np
import cv2
from RetCLIP.source.utils.misc import hash_prob


def add_fundus_illumination_circle(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    *,
    color_bgr=(0, 230, 0),  # greenish tint
    strength: float = 0.9,  # 0..1, overall strength of the artifact
    radius_frac=(0.95, 1.05),  # mid-radius as fraction of fundus radius
    softness_frac: float = 0.65,  # 0..1, rim thickness as fraction of radius
    center_jitter_frac: float = 0.1,  # random jitter of the ring center
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Add a full circular peripheral illumination artifact to a fundus image.

    Geometry:
      * The circle is centered near the fundus center with small jitter.
      * radius_frac is expressed as a fraction of the fundus radius
        (≈ 0.5 * min(H, W)).
      * We build a radial band (a rim) around that radius for the FULL circle.

    Returns:
      (out_img_uint8, meta_dict, binary_mask)
      binary_mask: HxW uint8 array with 1s where artifact is applied, 0s elsewhere
    """
    assert img.dtype == np.uint8
    H, W = img.shape[:2]

    # --------------------- deterministic randomness -------------------------
    def _rp(off: int) -> float:
        return hash_prob(uid, seed + off)

    t_u = _rp(1)
    s_u = _rp(2)

    # ------------------- radius & rim thickness (peripheral) ----------------
    # treat base as fundus radius
    base = 0.5 * float(min(H, W))

    r_min, r_max = radius_frac
    # mid radius of the rim (where alpha is strongest)
    mid_frac = 0.5 * (r_min + r_max)
    R_mid = base * mid_frac
    R_mid = max(5.0, float(R_mid))
    # keep rim just inside the fundus radius
    margin_px = 2.0
    R_mid = min(R_mid, base - margin_px)

    # thickness of the rim band
    soft = max(1e-3, float(softness_frac))
    thickness = max(1.0, R_mid * soft)
    half_t = thickness * 0.5

    R_inner = R_mid - half_t
    R_outer = R_mid + half_t

    # ------------------------ ring center near fundus -----------------------
    cx0 = W * 0.5
    cy0 = H * 0.5

    # small jitter around center (relative to fundus radius)
    jitter_px = float(center_jitter_frac) * base
    cx = cx0 + (2.0 * t_u - 1.0) * jitter_px
    cy = cy0 + (2.0 * s_u - 1.0) * jitter_px

    # --------------------- build radial FULL-circle mask --------------------
    yy, xx = np.indices((H, W), dtype=np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)
    dist = np.sqrt(dx * dx + dy * dy)

    # distance from the mid radius
    dist_mid = np.abs(dist - R_mid)

    alpha = np.zeros((H, W), dtype=np.float32)
    band = dist_mid <= half_t
    # triangular falloff inside the rim band
    alpha[band] = 1.0 - (dist_mid[band] / half_t)

    # no directional mask here → full circle
    alpha = np.clip(alpha * float(strength), 0.0, 1.0)

    # ------------------------- tint + slight brightening --------------------
    img_f = img.astype(np.float32) / 255.0
    col = np.array(color_bgr, dtype=np.float32) / 255.0  # BGR in 0..1
    col_img = np.ones_like(img_f) * col.reshape(1, 1, 3)

    # brighten original under the rim a bit
    brighten = np.clip(img_f * 1.25, 0.0, 1.0)
    # mix with the tint
    target = 0.6 * brighten + 0.4 * col_img

    a3 = alpha[..., None]
    out_f = (1.0 - a3) * img_f + a3 * target
    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    meta = dict(
        cx=float(cx),
        cy=float(cy),
        R_mid=float(R_mid),
        R_inner=float(R_inner),
        R_outer=float(R_outer),
        thickness=float(thickness),
        strength=float(strength),
        color_bgr=tuple(int(c) for c in color_bgr),
    )
    # Binary mask: 1 where artifact is applied (alpha > 0), 0 elsewhere
    binary_mask = (alpha > 0).astype(np.uint8)
    return out_u8, meta, binary_mask


def add_fundus_out_of_focus_quarter(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    *,
    corner: str = "auto",  # "tl" | "tr" | "bl" | "br" | "auto"
    radius_frac=(0.75, 1.05),  # radius as fraction of min(H,W)
    softness_frac: float = 0.30,  # width of falloff band (0..1 of radius)
    blur_ksize: int = 51,  # Gaussian blur kernel (odd, large → strong defocus)
    darkness: float = 0.55,  # 0..1 (multiply brightness in artifact region)
    strength: float = 0.85,  # 0..1 (overall mixing with blurred+darkened version)
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Add a dark, out-of-focus quarter-circle artifact to a fundus image.

    - img: HxWx3 uint8.
    - uid, seed: used for deterministic pseudo-randomness (via hash_prob).
    - corner: which corner the quarter-circle comes from:
        "tl" (top-left), "tr", "bl", "br", or "auto" to choose per-image.
    - radius_frac: (min,max) radius as fraction of min(H,W).
    - softness_frac: fraction of radius used for soft fade-out at the edge.
    - blur_ksize: Gaussian kernel size for defocus (must be odd).
    - darkness: how much darker the defocused region is (multiply factor).
    - strength: how strongly to blend artifact vs original (0..1).

    Returns:
        out_img_uint8, meta_dict, binary_mask
        binary_mask: HxW uint8 array with 1s where artifact is applied, 0s elsewhere
    """
    assert img.dtype == np.uint8
    H, W = img.shape[:2]
    base = float(min(H, W))

    # deterministic helpers using your hash_prob
    def _rp(off: int) -> float:
        return hash_prob(uid, seed + off)

    # choose corner if auto
    if corner == "auto":
        u = _rp(0)
        if u < 0.25:
            corner = "tl"
        elif u < 0.50:
            corner = "tr"
        elif u < 0.75:
            corner = "bl"
        else:
            corner = "br"

    # radius and inner/outer falloff
    rmin, rmax = radius_frac
    mid_frac = 0.5 * (rmin + rmax)
    R = base * mid_frac
    R = max(10.0, float(R))

    soft = max(1e-3, float(softness_frac))
    R_inner = R * (1.0 - soft)  # full-strength defocus
    R_outer = R  # fades to 0

    # circle center at (slightly outside) chosen corner
    offset = 0.05 * base
    if corner == "tl":
        cx, cy = -offset, -offset
    elif corner == "tr":
        cx, cy = W - 1 + offset, -offset
    elif corner == "bl":
        cx, cy = -offset, H - 1 + offset
    else:  # "br"
        cx, cy = W - 1 + offset, H - 1 + offset

    # --- build radial quarter-circle alpha mask ---
    yy, xx = np.indices((H, W), dtype=np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)
    dist = np.sqrt(dx * dx + dy * dy)

    alpha = np.zeros((H, W), dtype=np.float32)
    inside = dist <= R_inner
    alpha[inside] = 1.0
    band = (dist > R_inner) & (dist < R_outer)
    alpha[band] = 1.0 - (dist[band] - R_inner) / (R_outer - R_inner)

    # global strength scaling
    alpha = np.clip(alpha * float(strength), 0.0, 1.0)

    # --- defocus: blur entire image, then darken under the mask ---
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    img_f = img.astype(np.float32) / 255.0
    blur_f = blurred.astype(np.float32) / 255.0

    # darken blurred region
    dark_factor = float(darkness)
    target = np.clip(blur_f * dark_factor, 0.0, 1.0)

    a3 = alpha[..., None]
    out_f = (1.0 - a3) * img_f + a3 * target
    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    meta = dict(
        corner=corner,
        cx=float(cx),
        cy=float(cy),
        R=float(R),
        R_inner=float(R_inner),
        R_outer=float(R_outer),
        blur_ksize=int(k),
        darkness=float(darkness),
        strength=float(strength),
    )
    # Binary mask: 1 where artifact is applied (alpha > 0), 0 elsewhere
    binary_mask = (alpha > 0).astype(np.uint8)
    return out_u8, meta, binary_mask


def add_fundus_bluish_circle(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    *,
    radius_frac=(0.06, 0.12),  # min/max radius as fraction of min(H,W)
    center_x_frac=(0.30, 0.75),  # allowed x-range (fractions of width)
    center_y_frac=(0.25, 0.75),  # allowed y-range (fractions of height)
    color_bgr=(220, 190, 130),  # bluish / cyan-ish tint in BGR
    strength: float = 0.85,  # 0..1 overall artifact strength
    blur_ksize: int = 19,  # local blur kernel (odd)
    brighten: float = 1.10,  # >1 brightens the spot slightly
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Add a small bluish, soft-edged circular artifact to a fundus image.

    - img: HxWx3 (BGR) uint8.
    - uid, seed: used with hash_prob for deterministic radius & position.
    - radius_frac: min,max circle radius as fraction of min(H,W).
    - center_x_frac, center_y_frac: range (in fractions) for center location.
    - color_bgr: BGR tint color for the artifact.
    - strength: how strongly the artifact overrides the original (0..1).
    - blur_ksize: Gaussian blur size used to slightly defocus the spot.
    - brighten: factor to brighten the region (e.g. 1.1 = +10%).

    Returns:
        out_img_uint8, meta_dict, binary_mask
        binary_mask: HxW uint8 array with 1s where artifact is applied, 0s elsewhere
    """
    assert img.dtype == np.uint8
    H, W = img.shape[:2]
    base = float(min(H, W))

    # deterministic helpers
    def _rp(off: int) -> float:
        return hash_prob(uid, seed + off)

    # radius
    r_u = _rp(0)
    rmin, rmax = radius_frac
    R = base * (rmin + (rmax - rmin) * r_u)
    R = max(5.0, float(R))

    # center (x,y) constrained to middle-ish region
    xu = _rp(1)
    yu = _rp(2)
    cx = int(W * (center_x_frac[0] + (center_x_frac[1] - center_x_frac[0]) * xu))
    cy = int(H * (center_y_frac[0] + (center_y_frac[1] - center_y_frac[0]) * yu))

    # --- radial alpha mask for the circle ---
    yy, xx = np.indices((H, W), dtype=np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # smooth falloff: 1 at center → 0 at radius R
    alpha = np.zeros((H, W), dtype=np.float32)
    inside = dist <= R
    # cosine falloff for softness
    alpha[inside] = 0.5 * (1 + np.cos(np.pi * dist[inside] / R))
    alpha = np.clip(alpha * float(strength), 0.0, 1.0)

    # --- build tinted, slightly blurred version for the spot ---
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    img_f = img.astype(np.float32) / 255.0
    blur_f = blurred.astype(np.float32) / 255.0

    # brighten blurred region a bit
    blur_f = np.clip(blur_f * float(brighten), 0.0, 1.0)

    # mix blurred region with bluish tint
    col = np.array(color_bgr, dtype=np.float32) / 255.0  # BGR
    col_img = np.ones_like(blur_f) * col.reshape(1, 1, 3)
    spot = np.clip(0.4 * blur_f + 0.6 * col_img, 0.0, 1.0)

    # --- blend into image with radial alpha ---
    a3 = alpha[..., None]
    out_f = (1.0 - a3) * img_f + a3 * spot
    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    meta = dict(
        cx=int(cx),
        cy=int(cy),
        R=float(R),
        strength=float(strength),
        blur_ksize=int(k),
        brighten=float(brighten),
        color_bgr=tuple(int(c) for c in color_bgr),
    )
    # Binary mask: 1 where artifact is applied (alpha > 0), 0 elsewhere
    binary_mask = (alpha > 0).astype(np.uint8)
    return out_u8, meta, binary_mask


def add_fundus_reflection_double_dot(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    *,
    base_radius_frac=(0.018, 0.035),  # radius as fraction of min(H,W)
    size_ratio=(0.45, 0.75),  # small_radius = big_radius * U(range)
    separation_frac: float = 2.0,  # center distance in units of big_radius
    center_x_frac=(0.35, 0.70),  # where the pair is roughly placed
    center_y_frac=(0.35, 0.65),
    halo_strength: float = 0.9,  # 0..1
    core_strength: float = 1.0,  # 0..1 for saturated core
    halo_color_bgr=(255, 235, 200),  # warm highlight color
    core_color_bgr=(255, 255, 255),  # specular core (near white)
    halo_softness: float = 1.8,  # larger = softer falloff
    core_frac: float = 0.35,  # radius of hard core = core_frac * radius
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Add a pair of bright reflection artifacts (one big, one small) to a fundus image.

    - img: HxWx3 uint8 (BGR).
    - uid, seed: used with hash_prob to make placement deterministic per image.
    - base_radius_frac: min/max for the *larger* spot radius as fraction of min(H,W).
    - size_ratio: range of (small_radius / big_radius).
    - separation_frac: distance between centers in multiples of big_radius.
    - center_*_frac: box (in relative coords) for the midpoint of the pair.
    - halo_strength: blend amount for the colored halo (0..1).
    - core_strength: blend amount for the saturated core (0..1).
    - halo_color_bgr: BGR color for the halo.
    - core_color_bgr: BGR color for the core.
    - halo_softness: controls how quickly halo fades (gamma).
    - core_frac: core radius as fraction of each spot radius.

    Returns:
        out_img_uint8, meta_dict, binary_mask
        binary_mask: HxW uint8 array with 1s where artifact is applied, 0s elsewhere
    """
    assert img.dtype == np.uint8
    H, W = img.shape[:2]
    base = float(min(H, W))

    # deterministic random from your hash_prob
    def _rp(off: int) -> float:
        return hash_prob(uid, seed + off)

    # larger radius
    rmin, rmax = base_radius_frac
    base_frac = 0.5 * (rmin + rmax)
    R_big = base * base_frac
    R_big = max(3.0, float(R_big))

    # smaller radius
    smin, smax = size_ratio
    ratio = 0.5 * (smin + smax)
    R_small = R_big * ratio
    R_small = max(2.0, float(R_small))

    # pair midpoint location
    xu = _rp(2)
    yu = _rp(3)
    cx_mid = int(W * (center_x_frac[0] + (center_x_frac[1] - center_x_frac[0]) * xu))
    cy_mid = int(H * (center_y_frac[0] + (center_y_frac[1] - center_y_frac[0]) * yu))

    # separation and orientation (roughly horizontal with slight tilt)
    # sep_u = _rp(4)
    # sep_min, sep_max = separation_frac
    # sep = (sep_min + (sep_max - sep_min) * sep_u) * R_big
    sep = float(separation_frac) * R_big

    angle = (_rp(5) - 0.5) * (np.pi / 7.0)  # small tilt around horizontal
    dx = sep * np.cos(angle) / 2.0
    dy = sep * np.sin(angle) / 2.0

    # big dot left/right randomly
    swap = _rp(6) < 0.5
    if swap:
        cx_big = cx_mid - dx
        cy_big = cy_mid - dy
        cx_small = cx_mid + dx
        cy_small = cy_mid + dy
    else:
        cx_big = cx_mid + dx
        cy_big = cy_mid + dy
        cx_small = cx_mid - dx
        cy_small = cy_mid - dy

    # clamp centers inside image
    cx_big = float(np.clip(cx_big, 0, W - 1))
    cy_big = float(np.clip(cy_big, 0, H - 1))
    cx_small = float(np.clip(cx_small, 0, W - 1))
    cy_small = float(np.clip(cy_small, 0, H - 1))

    # --- build alpha masks ---
    yy, xx = np.indices((H, W), dtype=np.float32)

    def spot_alpha(cx, cy, R):
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        a = np.clip(1.0 - dist / R, 0.0, 1.0)
        # soften with gamma
        return a**halo_softness, dist

    halo_big, dist_big = spot_alpha(cx_big, cy_big, R_big)
    halo_small, dist_small = spot_alpha(cx_small, cy_small, R_small)
    halo = np.clip(halo_big + halo_small, 0.0, 1.0) * float(halo_strength)

    # cores: smaller disc inside each spot
    Rc_big = R_big * float(core_frac)
    Rc_small = R_small * float(core_frac)
    core = np.zeros_like(halo, dtype=np.float32)
    core[dist_big <= Rc_big] += 1.0
    core[dist_small <= Rc_small] += 1.0
    core = np.clip(core, 0.0, 1.0) * float(core_strength)

    # --- build halo and core colors ---
    img_f = img.astype(np.float32) / 255.0

    halo_col = np.array(halo_color_bgr, dtype=np.float32) / 255.0
    core_col = np.array(core_color_bgr, dtype=np.float32) / 255.0
    halo_img = np.ones_like(img_f) * halo_col.reshape(1, 1, 3)
    core_img = np.ones_like(img_f) * core_col.reshape(1, 1, 3)

    # brighten underlying pixels slightly under the halo
    halo_target = np.clip(img_f * 1.3, 0.0, 1.0)
    halo_target = 0.5 * halo_target + 0.5 * halo_img

    a_h = halo[..., None]
    a_c = core[..., None]

    # first mix halo, then overlay core on top
    out_f = (1.0 - a_h) * img_f + a_h * halo_target
    out_f = (1.0 - a_c) * out_f + a_c * core_img

    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    meta = dict(
        cx_big=float(cx_big),
        cy_big=float(cy_big),
        R_big=float(R_big),
        cx_small=float(cx_small),
        cy_small=float(cy_small),
        R_small=float(R_small),
        sep=float(sep),
        halo_strength=float(halo_strength),
        core_strength=float(core_strength),
    )
    # Binary mask: 1 where artifact is applied (halo or core > 0), 0 elsewhere
    binary_mask = ((halo > 0) | (core > 0)).astype(np.uint8)
    return out_u8, meta, binary_mask


def add_fundus_eyelash_shadow_band(
    img: np.ndarray,
    uid: str,
    seed: int = 0,
    *,
    side: str = "auto",  # "top" | "bottom" | "auto"
    band_frac=(0.25, 0.35),  # vertical band height as fraction of H
    num_lashes=(7, 11),  # number of lashes in the central block
    thickness_px=(6, 14),  # lash thickness in pixels
    darkness: float = 0.40,  # multiply factor (<1, smaller = darker)
    strength: float = 1.0,  # blending strength
    blur_ksize: int = 23,  # blur for softness (odd)
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Strong eyelash/eyelid shadow band, like a vertical curtain from the margin.

    - PATTERN (shape of the lash block) is fixed for a given seed.
    - LOCATION (top/bottom) is random per image via uid.
    - Lashes are mostly vertical, concentrated in a central horizontal block.

    Returns:
        out_img_uint8, meta_dict, binary_mask
        binary_mask: HxW uint8 array with 1s where artifact is applied, 0s elsewhere
    """
    assert img.dtype == np.uint8
    H, W = img.shape[:2]

    def _rp_pattern(off: int) -> float:
        # pattern randomness: same across images if seed is fixed
        return hash_prob("eyelash_pattern", seed + off)

    def _rp_pos(off: int) -> float:
        # position randomness: varies per image via uid
        return hash_prob(uid, seed + off)

    # -------- choose side (LOCATION) ----------
    if side == "auto":
        side = "bottom" if _rp_pos(0) < 0.7 else "top"  # slight bias to bottom

    # -------- band geometry (PATTERN) ---------
    u_band = _rp_pattern(1)
    fmin, fmax = band_frac
    band_h = int(H * (fmin + (fmax - fmin) * u_band))
    band_h = max(8, min(H // 2, band_h))

    # small inset so the band overlaps retina, not just border
    inset = int(0.02 * H)
    if side == "top":
        y0, y1 = inset, inset + band_h
    else:  # "bottom"
        y0, y1 = H - band_h - inset, H - inset

    # base alpha for the band (veil)
    alpha = np.zeros((H, W), dtype=np.float32)
    yy = np.linspace(0, 1, band_h, endpoint=True).astype(np.float32)

    # Strong at margin, decays into retina
    if side == "bottom":
        # 1 at bottom, fading to 0 at top of band
        band_profile = 1.0 - yy
    else:  # top
        band_profile = yy

    # slightly soften extremes
    band_profile = 0.2 + 0.8 * band_profile
    alpha_band = band_profile[:, None]
    alpha[y0:y1, :] = alpha_band

    # -------- lashes in central block (PATTERN) --------
    nmin, nmax = num_lashes
    n_lashes = int(nmin + (nmax - nmin) * _rp_pattern(2))
    n_lashes = max(1, n_lashes)

    lash_mask = np.zeros_like(alpha, dtype=np.float32)
    tmin, tmax = thickness_px

    # central horizontal block (e.g. 30%–70% of width)
    x_lo = int(0.40 * W)
    x_hi = int(0.60 * W)

    for i in range(n_lashes):
        u = _rp_pattern(10 + i)
        v = _rp_pattern(20 + i)

        # x only in central block
        x_center = int(x_lo + (x_hi - x_lo) * u)
        thickness = int(tmin + (tmax - tmin) * v)
        thickness = max(3, thickness)

        # vertical-ish lashes: base at 90°, small deviation ±15°
        # pattern-fixed tilt, but allow a lot and avoid tiny tilts
        max_delta_deg = 40.0  # max deviation from vertical

        u_tilt = _rp_pattern(30 + i)  # 0..1
        u_tilt = 2.0 * u_tilt - 1.0  # -1..1

        # push values away from 0 so |u_tilt| >= 0.4 → minimum tilt
        min_abs = 0.4
        u_tilt = np.sign(u_tilt) * np.maximum(min_abs, np.abs(u_tilt))

        delta = u_tilt * (max_delta_deg * np.pi / 180.0)
        angle = np.pi / 2.0 + delta  # around vertical
        dx = np.cos(angle)
        dy = np.sin(angle)

        length = 1.4 * H  # tall strokes
        x0_l = x_center - length * dx
        y0_l = (y0 + y1) * 0.5 - length * dy
        x1_l = x_center + length * dx
        y1_l = (y0 + y1) * 0.5 + length * dy

        pts = (
            np.array([[x0_l, y0_l], [x1_l, y1_l]], dtype=np.float32)
            .round()
            .astype(np.int32)
        )

        cv2.line(
            lash_mask,
            (int(pts[0, 0]), int(pts[0, 1])),
            (int(pts[1, 0]), int(pts[1, 1])),
            1.0,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # restrict lashes to band region
    lash_mask *= (alpha > 0).astype(np.float32)

    # blur for softness to mimic defocused lashes
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    lash_blur = cv2.GaussianBlur(lash_mask, (k, k), 0)

    # combine band + lashes; lashes boosted a bit
    alpha = np.clip(lash_blur, 0.0, 1.0)
    alpha *= float(strength)

    # -------- apply darkening ----------
    img_f = img.astype(np.float32) / 255.0
    dark = np.clip(img_f * float(darkness), 0.0, 1.0)

    a3 = alpha[..., None]
    out_f = (1.0 - a3) * img_f + a3 * dark
    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    meta = dict(
        side=side,
        y0=int(y0),
        y1=int(y1),
        band_h=int(band_h),
        num_lashes=int(n_lashes),
        darkness=float(darkness),
        strength=float(strength),
    )
    # Binary mask: 1 where artifact is applied (alpha > 0), 0 elsewhere
    binary_mask = (alpha > 0).astype(np.uint8)
    return out_u8, meta, binary_mask
