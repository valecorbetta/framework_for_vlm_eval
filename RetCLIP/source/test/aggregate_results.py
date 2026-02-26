import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------- helpers ----------------- #


def _isnumber(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _safe_mean_std(series: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(series, errors="coerce").astype(float).dropna().values
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1))  # sample std


def _read_metrics_json(p: Path) -> Optional[Dict]:
    if not p.exists():
        return None
    with open(p, "r") as f:
        d = json.load(f)
    # unify naming across earlier variants
    if "bal_acc" in d and "balanced_acc" not in d:
        d["balanced_acc"] = d.pop("bal_acc")
    return d


# ----------------- metrics.json aggregation ----------------- #


def _aggregate_metrics_json(
    overlay_dir: Path, split_dirs: List[Path]
) -> Optional[pd.DataFrame]:
    rows = []
    for sd in split_dirs:
        d = _read_metrics_json(sd / "metrics.json")
        if d is None:
            continue
        d_row = {k: v for k, v in d.items() if _isnumber(v)}
        d_row["split"] = sd.name
        rows.append(d_row)

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("split")
    out = []
    for col in df.columns:
        mu, st = _safe_mean_std(df[col])
        out.append(
            {"metric": col, "mean": mu, "std": st, "n_splits": df[col].notna().sum()}
        )
    return pd.DataFrame(out)


# ----------------- per-class accuracy aggregation ----------------- #


def _aggregate_per_class_accuracy(
    overlay_dir: Path, split_dirs: List[Path]
) -> Optional[pd.DataFrame]:
    dfs = []
    for sd in split_dirs:
        p = sd / "per_class_accuracy.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        # expected columns: class, acc, n
        if "class" not in d.columns or "acc" not in d.columns:
            continue
        d = d.copy()
        d["split"] = sd.name
        dfs.append(d)
    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    # group by class id
    gb = all_df.groupby("class", dropna=False)
    out_rows = []
    for cls, g in gb:
        mu_acc, st_acc = _safe_mean_std(g["acc"])
        # n is not a “metric”, but useful; we report total and mean across splits
        n_total = int(
            pd.to_numeric(g.get("n", pd.Series([np.nan] * len(g))), errors="coerce")
            .fillna(0)
            .sum()
        )
        out_rows.append(
            {
                "class": int(cls) if not pd.isna(cls) else cls,
                "acc_mean": mu_acc,
                "acc_std": st_acc,
                "n_total": n_total,
                "n_splits": g["acc"].notna().sum(),
            }
        )
    return pd.DataFrame(out_rows).sort_values("class")


# ----------------- subgroup metrics aggregation ----------------- #


def _aggregate_subgroup_metrics(
    overlay_dir: Path, split_dirs: List[Path]
) -> Optional[pd.DataFrame]:
    dfs = []
    for sd in split_dirs:
        p = sd / "subgroup_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
        d = d.copy()
        d["split"] = sd.name
        dfs.append(d)
    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)

    # Identify metric columns to aggregate (numeric)
    metric_cols = [
        c
        for c in all_df.columns
        if c
        not in {
            "split",
            "task",
            "slice",
            # old schema:
            "subgroup",
            "subgroup_type",
            "subgroup_value",
            "value",
            "birads",
            "class_birads",
            "note",
            # new schema:
            "concept",
            "class_label",
        }
        and np.issubdtype(all_df[c].dtype, np.number)
    ]

    # Treat "n" specially: we will sum it; for the others we do mean/std
    sum_cols = [c for c in metric_cols if c == "n"]
    mean_std_cols = [c for c in metric_cols if c != "n"]

    # Grouping keys: everything non-numeric and not “split”
    id_cols = [c for c in all_df.columns if c not in metric_cols + ["split"]]

    def _agg_fn(g: pd.DataFrame) -> pd.Series:
        out: Dict[str, float | int] = {}
        # sum "n" across splits (if present)
        for c in sum_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            out[f"{c}_sum"] = float(vals.fillna(0).sum())
            out[f"{c}_mean"] = float(vals.mean())
            out[f"{c}_std"] = float(vals.std(ddof=1))
        # mean/std for other numeric metrics
        for c in mean_std_cols:
            mu, st = _safe_mean_std(g[c])
            out[f"{c}_mean"] = mu
            out[f"{c}_std"] = st
            out[f"{c}_n_splits"] = int(g[c].notna().sum())
        # also count contributing splits
        out["splits_count"] = int(g["split"].nunique())
        return pd.Series(out)

    agg = all_df.groupby(id_cols, dropna=False).apply(_agg_fn).reset_index()

    # Stabilize column order a bit: support both old and new schemas
    front = [
        c
        for c in [
            "task",
            "slice",
            # old schema:
            "class_birads",
            "birads",
            "subgroup",
            "subgroup_type",
            "value",
            "subgroup_value",
            # new schema:
            "concept",
            "class_label",
        ]
        if c in agg.columns
    ]
    metrics_ordered = [c for c in agg.columns if c not in front]
    return agg[front + metrics_ordered]


# ----------------- test_only layout aggregation ----------------- #


def _aggregate_test_only_layout(root: Path, split_dirs, is_mica: bool = False) -> None:
    """
    Layout assumed:

        root/
          split0/
            test_only/
              overlay_a/
                metrics.json, per_class_accuracy.csv, subgroup_metrics.csv, ...
              overlay_b/
                ...
          split1/
            test_only/
              overlay_a/
              overlay_b/
          ...

    For MICA models (is_mica=True), the layout is:

        root/
          split0/
            mica_stage2/
              test_only/
                overlay_a/
                ...

    We aggregate per overlay (overlay_a, overlay_b, ...) across the splits
    that are *direct children* of `root`.
    """
    # # IMPORTANT: only splits directly under root, do not recurse into pct_* etc.
    # split_dirs = sorted(
    #     p
    #     for p in root.iterdir()
    #     if p.is_dir() and p.name.startswith(("split_", "split"))
    # )
    # if not split_dirs:
    #     print(f"[aggregate] No split* folders found in test_only layout under: {root}")
    #     return

    overlay_to_dirs: Dict[str, List[Path]] = defaultdict(list)

    for sd in split_dirs:
        if is_mica:
            test_root = sd / "mica_stage2" / "test_only"
        else:
            test_root = sd / "test_only"
        if not test_root.is_dir():
            continue
        for ov in test_root.iterdir():
            if ov.is_dir():
                overlay_to_dirs[ov.name].append(ov)

    if not overlay_to_dirs:
        print(f"[aggregate] No test_only overlays found under splits in: {root}")
        return

    for ov_name, metric_dirs in overlay_to_dirs.items():
        print(
            f"[aggregate] processing test_only overlay: {ov_name} "
            f"({len(metric_dirs)} splits)"
        )

        # write per-overlay summaries in root/overlay_name
        out_root = root / ov_name
        out_root.mkdir(exist_ok=True)

        df_metrics = _aggregate_metrics_json(out_root, metric_dirs)
        if df_metrics is not None:
            out_path = out_root / "summary_metrics.csv"
            df_metrics.to_csv(out_path, index=False)
            print(f"[aggregate] wrote {out_path}")

        df_pclass = _aggregate_per_class_accuracy(out_root, metric_dirs)
        if df_pclass is not None:
            out_path = out_root / "summary_per_class_accuracy.csv"
            df_pclass.to_csv(out_path, index=False)
            print(f"[aggregate] wrote {out_path}")

        df_sub = _aggregate_subgroup_metrics(out_root, metric_dirs)
        if df_sub is not None:
            out_path = out_root / "summary_subgroup_metrics.csv"
            df_sub.to_csv(out_path, index=False)
            print(f"[aggregate] wrote {out_path}")


# ----------------- delta metrics (pct_X - pct_0) aggregation ----------------- #


# Import _find_pct_dirs from get_confusion_matrices (same directory)
from RetCLIP.source.test.get_confusion_matrices import _find_pct_dirs


def _aggregate_delta_metrics_test_only(
    root: Path, overlay_name: str, is_mica: bool = False
) -> Optional[pd.DataFrame]:
    """
    Compute delta metrics (pct_X - pct_0) at the split level, then aggregate.

    For each split:
      - Read metrics from pct_0 and pct_X
      - Compute delta = metric(pct_X) - metric(pct_0)
    Then average deltas across splits with std.

    Returns DataFrame with columns: metric, pct, delta_mean, delta_std, n_splits
    """
    pct_dirs = _find_pct_dirs(root)
    if not pct_dirs:
        return None

    # Find baseline (pct_0)
    baseline_pct, baseline_dir = None, None
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            baseline_pct, baseline_dir = pct_val, pct_path
            break

    if baseline_dir is None:
        print(f"[delta] No pct_000 baseline found under {root}")
        return None

    # Get split names from baseline
    split_dirs_baseline = sorted(
        p for p in baseline_dir.iterdir()
        if p.is_dir() and p.name.startswith(("split_", "split"))
    )
    split_names = [sd.name for sd in split_dirs_baseline]

    if not split_names:
        print(f"[delta] No splits found under {baseline_dir}")
        return None

    # Build path to metrics for a given pct_dir and split
    def _get_metrics_path(pct_path: Path, split_name: str) -> Path:
        if is_mica:
            return pct_path / split_name / "mica_stage2" / "test_only" / overlay_name / "metrics.json"
        else:
            return pct_path / split_name / "test_only" / overlay_name / "metrics.json"

    # Read baseline metrics for each split
    baseline_by_split: Dict[str, Dict] = {}
    for split_name in split_names:
        p = _get_metrics_path(baseline_dir, split_name)
        d = _read_metrics_json(p)
        if d is not None:
            baseline_by_split[split_name] = {k: v for k, v in d.items() if _isnumber(v)}

    if not baseline_by_split:
        print(f"[delta] No baseline metrics found for overlay '{overlay_name}'")
        return None

    # For each other pct, compute deltas
    rows = []
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            continue  # skip baseline

        for split_name in split_names:
            if split_name not in baseline_by_split:
                continue

            p = _get_metrics_path(pct_path, split_name)
            d = _read_metrics_json(p)
            if d is None:
                continue

            baseline = baseline_by_split[split_name]
            for metric_name, val in d.items():
                if not _isnumber(val):
                    continue
                if metric_name not in baseline:
                    continue
                delta = float(val) - float(baseline[metric_name])
                rows.append({
                    "metric": metric_name,
                    "pct": pct_val,
                    "split": split_name,
                    "delta": delta,
                })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Aggregate deltas across splits
    out_rows = []
    for (metric, pct), g in df.groupby(["metric", "pct"]):
        mu, st = _safe_mean_std(g["delta"])
        out_rows.append({
            "metric": metric,
            "pct": pct,
            "delta_mean": mu,
            "delta_std": st,
            "n_splits": g["delta"].notna().sum(),
        })

    return pd.DataFrame(out_rows).sort_values(["metric", "pct"])


def _aggregate_delta_per_class_test_only(
    root: Path, overlay_name: str, is_mica: bool = False
) -> Optional[pd.DataFrame]:
    """
    Compute delta per-class accuracy (pct_X - pct_0) at the split level, then aggregate.

    Returns DataFrame with columns: class, pct, delta_acc_mean, delta_acc_std, n_splits
    """
    pct_dirs = _find_pct_dirs(root)
    if not pct_dirs:
        return None

    baseline_pct, baseline_dir = None, None
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            baseline_pct, baseline_dir = pct_val, pct_path
            break

    if baseline_dir is None:
        return None

    split_dirs_baseline = sorted(
        p for p in baseline_dir.iterdir()
        if p.is_dir() and p.name.startswith(("split_", "split"))
    )
    split_names = [sd.name for sd in split_dirs_baseline]

    if not split_names:
        return None

    def _get_per_class_path(pct_path: Path, split_name: str) -> Path:
        if is_mica:
            return pct_path / split_name / "mica_stage2" / "test_only" / overlay_name / "per_class_accuracy.csv"
        else:
            return pct_path / split_name / "test_only" / overlay_name / "per_class_accuracy.csv"

    # Read baseline per-class for each split: {split: {class: acc}}
    baseline_by_split: Dict[str, Dict[int, float]] = {}
    for split_name in split_names:
        p = _get_per_class_path(baseline_dir, split_name)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "class" not in df.columns or "acc" not in df.columns:
            continue
        baseline_by_split[split_name] = dict(zip(df["class"], df["acc"]))

    if not baseline_by_split:
        return None

    rows = []
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            continue

        for split_name in split_names:
            if split_name not in baseline_by_split:
                continue

            p = _get_per_class_path(pct_path, split_name)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "class" not in df.columns or "acc" not in df.columns:
                continue

            baseline = baseline_by_split[split_name]
            for _, row in df.iterrows():
                cls = row["class"]
                if cls not in baseline:
                    continue
                delta = float(row["acc"]) - float(baseline[cls])
                rows.append({
                    "class": int(cls),
                    "pct": pct_val,
                    "split": split_name,
                    "delta_acc": delta,
                })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    out_rows = []
    for (cls, pct), g in df.groupby(["class", "pct"]):
        mu, st = _safe_mean_std(g["delta_acc"])
        out_rows.append({
            "class": cls,
            "pct": pct,
            "delta_acc_mean": mu,
            "delta_acc_std": st,
            "n_splits": g["delta_acc"].notna().sum(),
        })

    return pd.DataFrame(out_rows).sort_values(["class", "pct"])


def _aggregate_delta_subgroups_test_only(
    root: Path, overlay_name: str, is_mica: bool = False
) -> Optional[pd.DataFrame]:
    """
    Compute delta subgroup metrics (pct_X - pct_0) at the split level, then aggregate.

    Returns DataFrame with subgroup identifiers + delta_<metric>_mean, delta_<metric>_std columns.
    """
    pct_dirs = _find_pct_dirs(root)
    if not pct_dirs:
        return None

    baseline_pct, baseline_dir = None, None
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            baseline_pct, baseline_dir = pct_val, pct_path
            break

    if baseline_dir is None:
        return None

    split_dirs_baseline = sorted(
        p for p in baseline_dir.iterdir()
        if p.is_dir() and p.name.startswith(("split_", "split"))
    )
    split_names = [sd.name for sd in split_dirs_baseline]

    if not split_names:
        return None

    def _get_subgroup_path(pct_path: Path, split_name: str) -> Path:
        if is_mica:
            return pct_path / split_name / "mica_stage2" / "test_only" / overlay_name / "subgroup_metrics.csv"
        else:
            return pct_path / split_name / "test_only" / overlay_name / "subgroup_metrics.csv"

    # Identify id columns and metric columns from first available file
    id_cols = None
    metric_cols = None

    for split_name in split_names:
        p = _get_subgroup_path(baseline_dir, split_name)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue

        # Identify columns
        potential_id = {"task", "slice", "subgroup", "subgroup_type", "subgroup_value",
                        "value", "concept", "class_label", "class_birads", "birads"}
        id_cols = [c for c in df.columns if c in potential_id]
        metric_cols = [c for c in df.columns if c not in potential_id and c != "n"
                       and np.issubdtype(df[c].dtype, np.number)]
        break

    if id_cols is None or not metric_cols:
        return None

    # Read baseline subgroups for each split: {split: DataFrame}
    baseline_by_split: Dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        p = _get_subgroup_path(baseline_dir, split_name)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        baseline_by_split[split_name] = df

    if not baseline_by_split:
        return None

    rows = []
    for pct_val, pct_path in pct_dirs:
        if pct_val == 0:
            continue

        for split_name in split_names:
            if split_name not in baseline_by_split:
                continue

            p = _get_subgroup_path(pct_path, split_name)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if df.empty:
                continue

            baseline_df = baseline_by_split[split_name]

            # Merge on id columns to compute deltas
            merged = df.merge(baseline_df, on=id_cols, suffixes=("", "_baseline"))

            for _, row in merged.iterrows():
                row_data = {c: row[c] for c in id_cols}
                row_data["pct"] = pct_val
                row_data["split"] = split_name

                for mc in metric_cols:
                    baseline_col = f"{mc}_baseline"
                    if mc in row and baseline_col in row:
                        if pd.notna(row[mc]) and pd.notna(row[baseline_col]):
                            row_data[f"delta_{mc}"] = float(row[mc]) - float(row[baseline_col])

                rows.append(row_data)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Find delta columns
    delta_cols = [c for c in df.columns if c.startswith("delta_")]

    # Aggregate
    group_cols = id_cols + ["pct"]
    out_rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_data = dict(zip(group_cols, keys))

        for dc in delta_cols:
            mu, st = _safe_mean_std(g[dc])
            row_data[f"{dc}_mean"] = mu
            row_data[f"{dc}_std"] = st

        row_data["n_splits"] = len(g)
        out_rows.append(row_data)

    result = pd.DataFrame(out_rows)

    # Order columns
    front = id_cols + ["pct"]
    rest = [c for c in result.columns if c not in front]
    return result[front + rest].sort_values(["pct"] + id_cols)


def aggregate_delta_metrics(
    root_dir: str,
    overlay_name: str,
    is_mica: bool = False,
) -> None:
    """
    Compute delta metrics (difference from pct_0) at split level, then aggregate.

    This performs paired analysis: for each split, compute delta = metric(pct_X) - metric(pct_0),
    then average deltas across splits with std.

    Outputs are written to root_dir/<overlay_name>/delta_*.csv
    """
    root = Path(root_dir)
    out_root = root / overlay_name
    out_root.mkdir(exist_ok=True)

    # Delta overall metrics
    df_delta = _aggregate_delta_metrics_test_only(root, overlay_name, is_mica=is_mica)
    if df_delta is not None:
        out_path = out_root / "delta_metrics.csv"
        df_delta.to_csv(out_path, index=False)
        print(f"[delta] wrote {out_path}")
    else:
        print(f"[delta] no delta metrics computed for overlay '{overlay_name}'")

    # Delta per-class accuracy
    df_delta_pc = _aggregate_delta_per_class_test_only(root, overlay_name, is_mica=is_mica)
    if df_delta_pc is not None:
        out_path = out_root / "delta_per_class_accuracy.csv"
        df_delta_pc.to_csv(out_path, index=False)
        print(f"[delta] wrote {out_path}")

    # Delta subgroup metrics
    df_delta_sub = _aggregate_delta_subgroups_test_only(root, overlay_name, is_mica=is_mica)
    if df_delta_sub is not None:
        out_path = out_root / "delta_subgroup_metrics.csv"
        df_delta_sub.to_csv(out_path, index=False)
        print(f"[delta] wrote {out_path}")


# ----------------- top-level orchestration ----------------- #


def aggregate_all_overlays(root_dir: str, test_only: bool = False, is_mica: bool = False) -> None:
    """
    Search recursively under `root_dir` and aggregate metrics.

    Supports two layouts:

    1) Train + test (old layout)
       root/.../pct_000/split0/metrics.json

    2) Test-only layout
       root/.../split0/test_only/<overlay_mode>/metrics.json

    3) Test-only layout for MICA (is_mica=True)
       root/.../split0/mica_stage2/test_only/<overlay_mode>/metrics.json
    """
    root = Path(root_dir)

    # -------- train + test layout: pct_* folders that contain split* -------- #
    overlay_dirs = sorted(p for p in root.rglob("pct_*") if p.is_dir())

    if not overlay_dirs:
        print(f"[aggregate] No pct_* folders found under: {root}")
        return

    any_with_splits = False
    for overlay in overlay_dirs:
        split_dirs = sorted(
            p
            for p in overlay.iterdir()
            if p.is_dir() and p.name.startswith(("split_", "split"))
        )
        if not split_dirs:
            continue
        any_with_splits = True
        print(f"[aggregate] processing overlay: {overlay}")
        if test_only:
            _aggregate_test_only_layout(overlay, split_dirs, is_mica=is_mica)
        else:
            df_metrics = _aggregate_metrics_json(overlay, split_dirs)
            if df_metrics is not None:
                out_path = overlay / "summary_metrics.csv"
                df_metrics.to_csv(out_path, index=False)
                print(f"[aggregate] wrote {out_path}")
            else:
                print(
                    f"[aggregate] metrics.json missing across splits for {overlay.name}"
                )

            df_pclass = _aggregate_per_class_accuracy(overlay, split_dirs)
            if df_pclass is not None:
                out_path = overlay / "summary_per_class_accuracy.csv"
                df_pclass.to_csv(out_path, index=False)
                print(f"[aggregate] wrote {out_path}")

            df_sub = _aggregate_subgroup_metrics(overlay, split_dirs)
            if df_sub is not None:
                out_path = overlay / "summary_subgroup_metrics.csv"
                df_sub.to_csv(out_path, index=False)
                print(f"[aggregate] wrote {out_path}")

    if not any_with_splits:
        print(f"[aggregate] No split* folders found under pct_* overlays in: {root}")


# ----------------- CLI ----------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate split-wise metrics over pct_* overlays or test_only overlays."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help=(
            "Root directory under which pct_* or split*/test_only/* folders live. "
            "The script will search recursively."
        ),
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Use test_only layout: split*/test_only/<overlay_mode>/ under root_dir.",
    )
    parser.add_argument(
        "--is-mica",
        action="store_true",
        help="Use MICA layout: split*/mica_stage2/test_only/<overlay_mode>/ under root_dir.",
    )
    args = parser.parse_args()
    aggregate_all_overlays(args.root_dir, test_only=args.test_only, is_mica=args.is_mica)
