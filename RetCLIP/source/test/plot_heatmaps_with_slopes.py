"""
Enhanced heatmap with integrated slope column and significance markers.
Adds a 'Slope' column to the right of the standard model×pct heatmap,
with stars indicating significant differences vs a reference model.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from scipy import stats


_PCT_PATTERN = re.compile(r"pct[_-]?(\d{1,3})$")


def _truncated_cmap(cmap_name, low=0.15, high=0.85):
    """Return a colormap that uses only the [low, high] portion of the original."""
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(low, high, 256))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)


def _find_pct_dirs(root: Path):
    pairs = []
    for p in root.iterdir():
        if p.is_dir():
            m = _PCT_PATTERN.match(p.name)
            if m:
                pairs.append((int(m.group(1)), p))
    return sorted(pairs, key=lambda t: t[0])


def _collect_per_split(model_configs, metric="balanced_acc"):
    """Collect per-split values: {label: {pct: [val, ...]}}"""
    all_data = {}
    for config in model_configs:
        root = Path(config["root_dir"])
        overlay = config["overlay_name"]
        label = config.get("label", root.name)
        model_is_mica = config.get("is_mica", False)

        pct_dirs = _find_pct_dirs(root)
        model_vals = {}

        for pct_val, pct_path in pct_dirs:
            split_dirs = sorted(
                p
                for p in pct_path.iterdir()
                if p.is_dir() and p.name.startswith(("split_", "split"))
            )
            vals = []
            for split_dir in split_dirs:
                if model_is_mica:
                    mp = (
                        split_dir
                        / "mica_stage2"
                        / "test_only"
                        / overlay
                        / "metrics.json"
                    )
                else:
                    mp = split_dir / "test_only" / overlay / "metrics.json"
                if not mp.exists():
                    continue
                try:
                    with open(mp) as f:
                        m = json.load(f)
                    val = m.get(metric)
                    if val is None and metric == "balanced_acc":
                        val = m.get("bal_acc")
                    if val is not None:
                        vals.append(float(val))
                except (json.JSONDecodeError, KeyError):
                    continue
            if vals:
                model_vals[pct_val] = vals

        if model_vals:
            all_data[label] = model_vals
    return all_data


def _compute_slopes(all_data):
    """Compute per-split slopes for each model."""
    slopes = {}
    for label, pct_vals in all_data.items():
        all_pcts = sorted(pct_vals.keys())
        n_splits = min(len(v) for v in pct_vals.values())
        label_slopes = []
        for i in range(n_splits):
            x_vals, y_vals = [], []
            for pct in all_pcts:
                if pct in pct_vals and i < len(pct_vals[pct]):
                    x_vals.append(float(pct))
                    y_vals.append(pct_vals[pct][i])
            if len(x_vals) >= 2:
                slope, _ = np.polyfit(x_vals, y_vals, 1)
                label_slopes.append(slope)
        if label_slopes:
            slopes[label] = np.array(label_slopes)
    return slopes


def _permutation_test_paired(x, y, n_perms=10000, alternative="less"):
    diffs = x - y
    observed = np.mean(diffs)
    rng = np.random.default_rng(seed=42)
    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = np.mean(diffs * signs)
        if alternative == "two-sided":
            if abs(perm_mean) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if perm_mean >= observed:
                count += 1
        elif alternative == "less":
            if perm_mean <= observed:
                count += 1
    return (count + 1) / (n_perms + 1)


def _sig_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def plot_heatmap_with_slopes(
    model_configs: list[dict],
    metric: str = "balanced_acc",
    reference_label: str = "VLM Baseline",
    out_path: str | None = None,
    dpi=300,
    title: str | None = None,
    cmap: str = "RdYlGn",
    cmap_low: float = 0.0,  # skip darkest red (0=full dark)
    cmap_high: float = 1.0,  # skip darkest green (1=full dark)
    figsize: tuple | None = None,
    use_permutation: bool = True,
    n_perms: int = 10000,
    slope_scale: float = 1e3,  # multiply slopes by this for display (1e3 → ×10⁻³)
    fontsize_cells: int = 11,
    fontsize_labels: int = 11,
    fontsize_title: int = 12,
    show_slope_column: bool = True,
    show_pct_stars: bool = False,  # stars in each cell comparing delta vs ref
    metric_label: (
        str | None
    ) = None,  # custom colorbar label (default: auto from metric name)
    vmin: float | None = None,  # colormap min (default: auto from data)
    vmax: float | None = None,  # colormap max (default: auto from data)
):
    """
    Heatmap of model × spurious fraction with an extra slope column.

    The slope column shows degradation rate (slope × 10³) with significance
    stars comparing each model's slope to the reference model.

    Args:
        model_configs: list of dicts with root_dir, overlay_name, label, is_mica
        metric: metric key in metrics.json
        reference_label: model to compare against for significance
        slope_scale: display multiplier for slopes (default 1e3 so -0.0048 → -4.8)
        show_slope_column: add slope column on right
        show_pct_stars: add stars in individual cells for per-pct significance
    """
    # ── collect data ──
    all_data = _collect_per_split(model_configs, metric)
    if not all_data:
        print("[heatmap] No data found.")
        return None, None

    model_labels = [c.get("label", Path(c["root_dir"]).name) for c in model_configs]
    model_labels = [l for l in model_labels if l in all_data]

    pcts = sorted(set(p for v in all_data.values() for p in v.keys()))
    n_models = len(model_labels)
    n_pcts = len(pcts)

    # ── compute means and stds ──
    means = np.full((n_models, n_pcts), np.nan)
    stds = np.full((n_models, n_pcts), np.nan)
    for m_idx, label in enumerate(model_labels):
        for p_idx, pct in enumerate(pcts):
            vals = all_data[label].get(pct, [])
            if vals:
                means[m_idx, p_idx] = np.mean(vals)
                stds[m_idx, p_idx] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    # ── compute slopes + significance ──
    slopes = _compute_slopes(all_data)
    slope_means = {}
    slope_stds = {}
    slope_pvals = {}
    slope_stars = {}

    for label in model_labels:
        if label in slopes:
            slope_means[label] = np.mean(slopes[label])
            slope_stds[label] = (
                np.std(slopes[label], ddof=1) if len(slopes[label]) > 1 else 0.0
            )

            if label != reference_label and reference_label in slopes:
                ref_s = slopes[reference_label]
                oth_s = slopes[label]
                n = min(len(ref_s), len(oth_s))
                # Test: ref slope < other slope (ref degrades faster)
                if use_permutation:
                    p = _permutation_test_paired(ref_s[:n], oth_s[:n], n_perms, "less")
                else:
                    _, p = stats.ttest_rel(ref_s[:n], oth_s[:n], alternative="less")
                slope_pvals[label] = p
                slope_stars[label] = _sig_stars(p)
            else:
                slope_pvals[label] = np.nan
                slope_stars[label] = "ref"

    # ── per-pct significance (optional) ──
    pct_stars = {}
    if show_pct_stars and reference_label in all_data:
        ref_baseline = np.array(all_data[reference_label].get(0, []))
        for label in model_labels:
            if label == reference_label:
                continue
            oth_baseline = np.array(all_data[label].get(0, []))
            for pct in pcts:
                if pct == 0:
                    continue
                ref_vals = np.array(all_data[reference_label].get(pct, []))
                oth_vals = np.array(all_data[label].get(pct, []))
                n = min(
                    len(ref_baseline), len(ref_vals), len(oth_baseline), len(oth_vals)
                )
                if n < 2:
                    continue
                ref_delta = ref_baseline[:n] - ref_vals[:n]
                oth_delta = oth_baseline[:n] - oth_vals[:n]
                # Test: ref drops more than other
                if use_permutation:
                    p = _permutation_test_paired(
                        ref_delta, oth_delta, n_perms, "greater"
                    )
                else:
                    _, p = stats.ttest_rel(ref_delta, oth_delta, alternative="greater")
                pct_stars[(label, pct)] = _sig_stars(p)

    # ── figure layout ──
    extra_cols = 1 if show_slope_column else 0
    total_cols = n_pcts + extra_cols

    if figsize is None:
        figsize = (total_cols * 1.5 + 1.8, n_models * 0.85 + 1.2)

    fig, ax = plt.subplots(figsize=figsize)

    # ── draw main heatmap ──
    _vmin = vmin if vmin is not None else np.nanmin(means)
    _vmax = vmax if vmax is not None else np.nanmax(means)
    _cmap = _truncated_cmap(cmap, cmap_low, cmap_high)
    im = ax.imshow(
        means,
        aspect="auto",
        cmap=_cmap,
        vmin=_vmin,
        vmax=_vmax,
        extent=[-0.5, n_pcts - 0.5, n_models - 0.5, -0.5],
    )

    # ── annotate main cells ──
    for m_idx in range(n_models):
        label = model_labels[m_idx]
        for p_idx in range(n_pcts):
            mu = means[m_idx, p_idx]
            sd = stds[m_idx, p_idx]
            if not np.isfinite(mu):
                continue

            text = f"{mu:.2f}\n±{sd:.2f}"

            # Add per-pct significance star if enabled
            if show_pct_stars and pcts[p_idx] > 0:
                star = pct_stars.get((label, pcts[p_idx]), "")
                if star:
                    text += f"\n{star}"

            ax.text(
                p_idx,
                m_idx,
                text,
                ha="center",
                va="center",
                fontsize=fontsize_cells,
                fontweight="bold",
                color="black",
            )

    # ── slope column ──
    if show_slope_column:
        # Draw a separator line
        sep_x = n_pcts - 0.5
        ax.axvline(sep_x, color="black", linewidth=1.5)

        # Slope column background — use a neutral color
        for m_idx, label in enumerate(model_labels):
            sm = slope_means.get(label, np.nan)
            if not np.isfinite(sm):
                continue

            rect = FancyBboxPatch(
                (sep_x + 0.05, m_idx - 0.45),
                0.9,
                0.9,
                boxstyle="square,pad=0",
                facecolor="#f0f0f0",
                edgecolor="none",
            )
            ax.add_patch(rect)

            # Text: slope value + stars
            ss = slope_stds.get(label, 0)
            star = slope_stars.get(label, "")
            disp_slope = sm * slope_scale
            disp_std = ss * slope_scale

            if star == "ref":
                star_text = "(ref)"
            else:
                star_text = star

            text = f"{disp_slope:.1f}\n±{disp_std:.1f}"
            if star_text:
                text += f" {star_text}"

            ax.text(
                sep_x + 0.5,
                m_idx,
                text,
                ha="center",
                va="center",
                fontsize=fontsize_cells - 1,
                fontweight="bold",
                color="black",
            )

    # ── axes ──
    ax.set_xticks(
        list(range(n_pcts)) + ([n_pcts - 0.5 + 0.5] if show_slope_column else [])
    )
    x_labels = [f"{p}%" for p in pcts]
    if show_slope_column:
        x_labels.append(f"Slope\n(×10⁻³)")
    ax.set_xticklabels(x_labels, fontsize=fontsize_labels)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=fontsize_labels)

    ax.set_xlabel("Spurious Fraction (%)", fontsize=fontsize_labels)

    # Extend axis limits for slope column
    ax.set_xlim(-0.5, n_pcts - 0.5 + (1 if show_slope_column else 0))
    ax.set_ylim(n_models - 0.5, -0.5)

    # Colorbar for main heatmap
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=fontsize_labels - 2)
    cbar.set_label(
        metric_label or metric.replace("_", " ").title(), fontsize=fontsize_labels - 1
    )

    if title:
        ax.set_title(title, fontsize=fontsize_title, pad=10)

    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_heatmap_with_slopes_from_csvs(
    summary_csv: str,
    slope_csv: str,
    out_path: str | None = None,
    dpi: int = 300,
    title: str | None = None,
    reference_label: str = "VLM Baseline",
    cmap: str = "RdYlGn",
    figsize: tuple | None = None,
    fontsize_cells: int = 11,
    fontsize_labels: int = 11,
    fontsize_title: int = 12,
    slope_scale: float = 1e3,
    metric_label: str = "Balanced Acc",
    pairwise_csv: str | None = None,
    show_pct_stars: bool = False,
):
    """
    Build the heatmap directly from the CSVs your script already outputs.
    No need to re-read all the split-level JSON files.

    Args:
        summary_csv: path to summary.csv (has acc_0pct_mean, acc_100pct_mean, slope_mean, etc.)
        slope_csv: path to slope_comparison.csv (has reference, comparison, t_pval, perm_pval)
        pairwise_csv: optional path to pairwise_deltas.csv for per-pct stars
        show_pct_stars: whether to show per-pct significance stars
    """
    summary = pd.read_csv(summary_csv)
    slope_comp = pd.read_csv(slope_csv)

    model_labels = summary["model"].tolist()
    n_models = len(model_labels)

    # Reconstruct mean/std at each pct from summary
    # We only have 0% and 100% from summary — need pairwise for intermediate
    # Better approach: use the pairwise_deltas to reconstruct, or just show 0% and 100%
    # For a full heatmap we need the per-pct data. Let's check what columns exist.

    # The summary has: acc_0pct_mean, acc_0pct_std, acc_100pct_mean, total_drop_mean, ...
    # For intermediate pcts, we'd need the pairwise or raw data.
    # → This function works best when called with the raw model_configs.
    # For CSV-only, let's build a simpler 2-column version or require additional data.

    print(
        "[info] For full heatmaps with all pct levels, use plot_heatmap_with_slopes() "
        "with model_configs pointing to your result directories."
    )
    print("[info] Building summary-only version from CSVs...")

    # Build slope significance
    slope_stars = {}
    for _, row in slope_comp.iterrows():
        p = row["perm_pval"]
        slope_stars[row["comparison"]] = _sig_stars(p)
    slope_stars[reference_label] = "ref"

    # Per-pct stars from pairwise
    pct_stars_map = {}
    if show_pct_stars and pairwise_csv:
        pw = pd.read_csv(pairwise_csv)
        for _, row in pw.iterrows():
            p = row["perm_pval"]
            pct_stars_map[(row["comparison"], int(row["pct"]))] = _sig_stars(p)

    # Collect pcts from pairwise if available
    if pairwise_csv:
        pw = pd.read_csv(pairwise_csv)
        pcts = sorted(set([0] + pw["pct"].unique().tolist()))
    else:
        pcts = [0, 100]

    # For now, print a formatted table since we don't have per-pct means from CSVs
    print("\nModel Robustness Summary:")
    print(
        f"{'Model':<20} {'Slope (×10⁻³)':<15} {'Sig vs Ref':<12} {'Drop (0→100%)':<15}"
    )
    print("-" * 62)
    for _, row in summary.iterrows():
        label = row["model"]
        sm = row["slope_mean"] * slope_scale
        ss = row["slope_std"] * slope_scale
        star = slope_stars.get(label, "")
        drop = row["total_drop_mean"]
        print(f"{label:<20} {sm:>6.1f} ± {ss:<5.1f}  {star:<12} {drop:>6.3f}")

    return summary, slope_stars


# ── convenience: build both test conditions side by side ──


def plot_dual_heatmap(
    model_configs: list[dict],
    overlay_none: str = "none",
    overlay_inverted: str = "inverted",
    metric: str = "balanced_acc",
    reference_label: str = "VLM Baseline",
    out_path: str | None = None,
    dpi: int = 300,
    title_none: str = "Balanced Generalization",
    title_inverted: str = "Swapped Robustness",
    suptitle: str | None = None,
    figsize: tuple | None = None,
    fontsize_cells: int = 10,
    fontsize_labels: int = 10,
    cmap: str = "RdYlGn",
    cmap_low: float = 0.0,  # skip darkest red (0=full dark)
    cmap_high: float = 1.0,  # skip darkest green (1=full dark)
    slope_scale: float = 1e3,
    use_permutation: bool = True,
    n_perms: int = 10000,
    metric_label: (
        str | None
    ) = None,  # custom colorbar label (default: auto from metric name)
    vmin: float | None = None,  # colormap min (default: auto from data)
    vmax: float | None = None,  # colormap max (default: auto from data)
):
    """
    Two heatmaps side by side: none (left) and inverted (right),
    each with integrated slope column. Saves significant vertical space
    compared to four separate figures.
    """
    configs_none = [{**c, "overlay_name": overlay_none} for c in model_configs]
    configs_inv = [{**c, "overlay_name": overlay_inverted} for c in model_configs]

    # Collect data for both
    data_none = _collect_per_split(configs_none, metric)
    data_inv = _collect_per_split(configs_inv, metric)

    model_labels = [c.get("label", Path(c["root_dir"]).name) for c in model_configs]
    model_labels = [l for l in model_labels if l in data_none or l in data_inv]
    n_models = len(model_labels)

    def _build_matrices(all_data):
        pcts = sorted(set(p for v in all_data.values() for p in v.keys()))
        n_pcts = len(pcts)
        means = np.full((n_models, n_pcts), np.nan)
        stds_ = np.full((n_models, n_pcts), np.nan)
        for m_idx, label in enumerate(model_labels):
            for p_idx, pct in enumerate(pcts):
                vals = all_data.get(label, {}).get(pct, [])
                if vals:
                    means[m_idx, p_idx] = np.mean(vals)
                    stds_[m_idx, p_idx] = np.std(vals, ddof=1) if len(vals) > 1 else 0
        slopes = _compute_slopes(all_data)

        # Significance vs reference
        slope_info = {}
        for label in model_labels:
            if label not in slopes:
                slope_info[label] = (np.nan, 0, "")
                continue
            sm = np.mean(slopes[label])
            ss = np.std(slopes[label], ddof=1) if len(slopes[label]) > 1 else 0
            if label == reference_label:
                slope_info[label] = (sm, ss, "(ref)")
            elif reference_label in slopes:
                ref_s = slopes[reference_label]
                oth_s = slopes[label]
                n = min(len(ref_s), len(oth_s))
                if use_permutation:
                    p = _permutation_test_paired(ref_s[:n], oth_s[:n], n_perms, "less")
                else:
                    _, p = stats.ttest_rel(ref_s[:n], oth_s[:n], alternative="less")
                slope_info[label] = (sm, ss, _sig_stars(p))
            else:
                slope_info[label] = (sm, ss, "")

        return pcts, means, stds_, slope_info

    pcts_n, means_n, stds_n, slopes_n = _build_matrices(data_none)
    pcts_i, means_i, stds_i, slopes_i = _build_matrices(data_inv)

    # ── figure: two subplots + dedicated colorbar axis ──
    n_pcts = max(len(pcts_n), len(pcts_i))
    if figsize is None:
        figsize = ((n_pcts + 1) * 1.45 * 2 + 2.0, n_models * 0.82 + 1.4)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # Global vmin/vmax for consistent color scale
    _vmin = vmin if vmin is not None else min(np.nanmin(means_n), np.nanmin(means_i))
    _vmax = vmax if vmax is not None else max(np.nanmax(means_n), np.nanmax(means_i))
    _cmap = _truncated_cmap(cmap, cmap_low, cmap_high)

    def _draw_panel(ax, pcts, means, stds_, slope_info, title_str, show_ylabel=True):
        n_p = len(pcts)
        im = ax.imshow(
            means,
            aspect="auto",
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            extent=[-0.5, n_p - 0.5, n_models - 0.5, -0.5],
        )

        # Main cell annotations
        for m_idx in range(n_models):
            for p_idx in range(n_p):
                mu = means[m_idx, p_idx]
                sd = stds_[m_idx, p_idx]
                if np.isfinite(mu):
                    ax.text(
                        p_idx,
                        m_idx,
                        f"{mu:.2f}\n±{sd:.2f}",
                        ha="center",
                        va="center",
                        fontsize=fontsize_cells,
                        fontweight="bold",
                        color="black",
                    )

        # Slope column
        sep_x = n_p - 0.5
        ax.axvline(sep_x, color="black", linewidth=1.5)

        for m_idx, label in enumerate(model_labels):
            sm, ss, star = slope_info.get(label, (np.nan, 0, ""))
            if not np.isfinite(sm):
                continue

            rect = FancyBboxPatch(
                (sep_x + 0.05, m_idx - 0.45),
                0.9,
                0.9,
                boxstyle="square,pad=0",
                facecolor="#f0f0f0",
                edgecolor="none",
            )
            ax.add_patch(rect)

            disp = sm * slope_scale
            disp_s = ss * slope_scale
            txt = f"{disp:+.1f}\n±{disp_s:.1f}"
            if star:
                txt += f" {star}"
            ax.text(
                sep_x + 0.5,
                m_idx,
                txt,
                ha="center",
                va="center",
                fontsize=fontsize_cells - 1,
                fontweight="bold",
                color="black",
            )

        # Axes
        xticks = list(range(n_p)) + [n_p]
        xlabels = [f"{p}%" for p in pcts] + ["Slope\n(×10⁻³)"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=fontsize_labels)
        ax.set_xlim(-0.5, n_p + 0.5)
        ax.set_ylim(n_models - 0.5, -0.5)

        if show_ylabel:
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(model_labels, fontsize=fontsize_labels)

        ax.set_xlabel("Spurious Fraction (%)", fontsize=fontsize_labels)
        ax.set_title(title_str, fontsize=fontsize_cells + 1, pad=8)

        return im

    im1 = _draw_panel(
        ax1, pcts_n, means_n, stds_n, slopes_n, title_none, show_ylabel=True
    )
    im2 = _draw_panel(
        ax2, pcts_i, means_i, stds_i, slopes_i, title_inverted, show_ylabel=False
    )

    # Share y-axis
    ax2.sharey(ax1)
    ax2.set_yticks([])

    # Colorbar in dedicated axis
    cbar = fig.colorbar(im2, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize_labels - 2)
    cbar.set_label(
        metric_label or metric.replace("_", " ").title(), fontsize=fontsize_labels - 1
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize_labels + 2, y=1.02)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, (ax1, ax2)


if __name__ == "__main__":
    pass
