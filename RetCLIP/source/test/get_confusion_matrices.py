# plot_confusions.py
# Usage (train + test layout):
#   from plot_confusions import make_confusion_figures
#   make_confusion_figures("/path/to/run/0", class_names=None)
#
# This will look for:
#   /path/to/run/0/pct_000/split*/<cm files>, pct_025/..., pct_050/..., pct_075/..., pct_100/...
#
# Usage (test_only layout):
#   make_confusion_figures(
#       "/path/to/run/0",
#       class_names=None,
#       test_only=True,
#       overlay_name="none",
#   )
#
# This will look for:
#   /path/to/run/0/pct_000/split*/test_only/none/<cm files>, pct_025/..., ...
# and will write figures into:
#   /path/to/run/0/none/figs_confusion/

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_PCT_RE = re.compile(r"pct[_-]?(\d{1,3})$")


def _find_pct_dirs(root: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _PCT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def _find_split_dirs(overlay: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in overlay.iterdir()
            if p.is_dir() and (p.name.startswith(("split_", "split")))
        ]
    )


def _read_cm_file(dir_with_cm: Path) -> Optional[np.ndarray]:
    # try csvs, then npy
    candidates = [
        "confusion_matrix.csv",
        "conf_matrix.csv",
        "cm.csv",
        "confusion_matrix.npy",
        "conf_matrix.npy",
        "cm.npy",
    ]
    for name in candidates:
        p = dir_with_cm / name
        if not p.exists():
            continue
        if p.suffix.lower() == ".csv":
            arr = pd.read_csv(p, header=None).values
        else:
            arr = np.load(p)
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr.astype(float)
    return None


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    rowsum = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(cm, rowsum, where=(rowsum != 0))
    out[~np.isfinite(out)] = 0.0
    return out


def _collect_confusions_train(root: Path) -> Dict[int, List[np.ndarray]]:
    """Return {pct: [cm_split_norm, ...]} using pct_*/split* layout."""
    data: Dict[int, List[np.ndarray]] = {}
    for pct, d in _find_pct_dirs(root):
        split_dirs = _find_split_dirs(d)
        cms = []
        for sd in split_dirs:
            cm = _read_cm_file(sd)
            if cm is None:
                continue
            cms.append(_row_normalize(cm))
        if cms:
            # ensure all same size
            ks = {c.shape[0] for c in cms}
            if len(ks) > 1:
                # keep only largest shape
                kmax = max(ks)
                cms = [c for c in cms if c.shape[0] == kmax]
            if cms:
                data[pct] = cms
        else:
            print(f"[warn] no confusion matrices in {d}")
    return data


def _collect_confusions_test_only(
    root: Path, overlay_name: str, is_mica: bool = False
) -> Dict[int, List[np.ndarray]]:
    """
    Test-only layout:
        root/pct_000/split*/test_only/overlay_name/<cm files>, ...

    For MICA models (is_mica=True):
        root/pct_000/split*/mica_stage2/test_only/overlay_name/<cm files>, ...

    Returns {pct: [cm_split_norm, ...]}.
    """
    data: Dict[int, List[np.ndarray]] = {}
    for pct, d in _find_pct_dirs(root):
        split_dirs = _find_split_dirs(d)
        cms = []
        for sd in split_dirs:
            if is_mica:
                cm_dir = sd / "mica_stage2" / "test_only" / overlay_name
            else:
                cm_dir = sd / "test_only" / overlay_name
            if not cm_dir.is_dir():
                continue
            cm = _read_cm_file(cm_dir)
            if cm is None:
                continue
            cms.append(_row_normalize(cm))
        if cms:
            ks = {c.shape[0] for c in cms}
            if len(ks) > 1:
                kmax = max(ks)
                cms = [c for c in cms if c.shape[0] == kmax]
            if cms:
                data[pct] = cms
        else:
            print(f"[warn] no confusion matrices for overlay '{overlay_name}' in {d}")
    return data


def _mean_std(cms: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.stack(cms, axis=0)  # (S, K, K)
    mu = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mu)
    return mu, sd


def _imshow_cm(ax, mat: np.ndarray, title: str, class_names: Optional[List[str]]):
    im = ax.imshow(mat, interpolation="nearest")  # default colormap
    ax.set_title(title, fontsize=10)
    k = mat.shape[0]
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    if class_names and len(class_names) == k:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticklabels(range(k))
        ax.set_yticklabels(range(k))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _annotate_cells(ax, mu: np.ndarray, sd: Optional[np.ndarray] = None):
    k = mu.shape[0]
    # Avoid clutter for very large K
    if k > 10:
        return
    for i in range(k):
        for j in range(k):
            text = f"{mu[i, j]:.2f}"
            if sd is not None:
                text += f"\n±{sd[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)


def make_confusion_figures(
    root_dir: str,
    out_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    include_panel: bool = True,
    include_delta: bool = True,
    include_per_split_pdf: bool = True,
    task_name: Optional[str] = None,
    test_only: bool = False,
    overlay_name: Optional[str] = None,
    is_mica: bool = False,
):
    """
    Build paper-ready confusion matrix visuals from per-split saved matrices.

    Train + test layout (default):
      - Mean ± SD confusion heatmap per pct_* overlay.
      - Optional Δ heatmap (pct_100 - pct_000).
      - Optional multi-page PDF with all per-split matrices.

    Test-only layout (test_only=True):
      - Reads confusion matrices from split*/test_only/<overlay_name>/ per pct_*.
      - Same outputs, but figures are written to root_dir/<overlay_name>/figs_confusion.

    Test-only layout for MICA (test_only=True, is_mica=True):
      - Reads confusion matrices from split*/mica_stage2/test_only/<overlay_name>/ per pct_*.
    """
    root = Path(root_dir)

    if test_only:
        if overlay_name is None:
            raise ValueError("overlay_name must be provided when test_only=True")
        data = _collect_confusions_test_only(root, overlay_name, is_mica=is_mica)
        # output: root_dir/overlay_name/figs_confusion
        out_root = root / overlay_name
    else:
        data = _collect_confusions_train(root)
        # output: root_dir/figs_confusion (original behavior)
        out_root = root

    if not data:
        raise RuntimeError(
            f"No confusion matrices found under {root} (test_only={test_only})"
        )

    out = Path(out_dir) if out_dir else (out_root / "figs_confusion")
    out.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{task_name} | " if task_name else ""
    if test_only and overlay_name:
        title_prefix = (
            f"{title_prefix}{overlay_name} | " if title_prefix else f"{overlay_name} | "
        )

    # --- 1) Mean ± SD per overlay (pct) --- #
    for pct in sorted(data.keys()):
        cms = data[pct]
        mu, sd = _mean_std(cms)
        fig, ax = plt.subplots(
            figsize=(max(4.5, mu.shape[0] * 0.6), max(3.5, mu.shape[0] * 0.6))
        )
        _imshow_cm(
            ax,
            mu,
            title=f"{title_prefix}Mean row-normalized CM (pct={pct})",
            class_names=class_names,
        )
        _annotate_cells(ax, mu, sd)
        fig.tight_layout()
        fig.savefig(out / f"mean_cm_pct_{pct:03d}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out / f'mean_cm_pct_{pct:03d}.png'}")

    # --- 2) Optional: Panel across overlays (mean ± SD annotations) --- #
    if include_panel:
        pcts_sorted = sorted(data.keys())
        k = next(iter(data.values()))[0].shape[0]
        cols = len(pcts_sorted)
        fig, axes = plt.subplots(
            1, cols, figsize=(cols * 3.2, max(3.0, k * 0.55)), squeeze=False
        )
        for ax, pct in zip(axes[0], pcts_sorted):
            mu, sd = _mean_std(data[pct])
            _imshow_cm(ax, mu, title=f"pct={pct}", class_names=class_names)
            _annotate_cells(ax, mu, sd)
        fig.suptitle(
            f"{title_prefix}Mean (±SD) row-normalized confusion matrices across overlays",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out / "panel_mean_cms.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    # --- 3) Optional: Δ heatmap (100% - 0%) --- #
    if include_delta:
        if 0 in data and 100 in data:
            mu0, _ = _mean_std(data[0])
            mu1, _ = _mean_std(data[100])
            delta = mu1 - mu0
            vmax = np.max(np.abs(delta))
            fig, ax = plt.subplots(
                figsize=(max(4.5, delta.shape[0] * 0.6), max(3.5, delta.shape[0] * 0.6))
            )
            im = ax.imshow(
                delta, interpolation="nearest", cmap="seismic", vmin=-vmax, vmax=vmax
            )
            ax.set_title(
                f"{title_prefix}Δ row-normalized CM (pct=100 − pct=0)", fontsize=10
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            k = delta.shape[0]
            ax.set_xticks(range(k))
            ax.set_yticks(range(k))
            if class_names and len(class_names) == k:
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticklabels(class_names)
            else:
                ax.set_xticklabels(range(k))
                ax.set_yticklabels(range(k))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _annotate_cells(ax, delta, None)
            fig.tight_layout()
            fig.savefig(
                out / "delta_cm_pct_100_minus_000.png", dpi=250, bbox_inches="tight"
            )
            plt.close(fig)
            print(f"[saved] {out / 'delta_cm_pct_100_minus_000.png'}")
        else:
            print("[info] Skipping Δ heatmap (need both pct_000 and pct_100).")

    # --- 4) Optional: per-split multi-page PDF (row-normalized) --- #
    if include_per_split_pdf:
        pdf_path = out / "per_split_confusions.pdf"
        with PdfPages(pdf_path) as pdf:
            for pct in sorted(data.keys()):
                cms = data[pct]
                k = cms[0].shape[0]
                cols = 4
                rows = int(np.ceil(len(cms) / cols))
                fig, axes = plt.subplots(
                    rows,
                    cols,
                    figsize=(cols * 2.8, max(2.4, rows * 2.8)),
                    squeeze=False,
                )
                fig.suptitle(
                    f"{title_prefix}Row-normalized per-split confusion matrices (pct={pct})",
                    fontsize=11,
                )
                for idx, ax in enumerate(axes.ravel()):
                    if idx < len(cms):
                        _imshow_cm(
                            ax,
                            cms[idx],
                            title=f"split {idx+1}",
                            class_names=class_names,
                        )
                        _annotate_cells(ax, cms[idx], None)
                    else:
                        ax.axis("off")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"[saved] {pdf_path}")
