import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _acc(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def _maybe_auroc_binary(y_true, scores):
    # returns NaN if only one class present
    try:
        u = set(np.unique(y_true))
        if {0, 1}.issubset(u):
            return float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    return float("nan")


def subgroup_metrics_binary_concepts(
    meta_df, y_true, y_pred, y_prob_pos, concept_cols=None
):
    """
    For each concept column (has_*), compute:
      1) Overall (all classes): metrics for concept == 0 and concept == 1.
      2) Per-class: for each class (0 and 1), metrics for concept == 0 and concept == 1
         restricted to samples where y_true == class.

    Returns: pd.DataFrame with one row per (concept, value, optional class).
    """
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob_pos = np.asarray(y_prob_pos)

    if concept_cols is None:
        concept_cols = [c for c in meta_df.columns if c.startswith("has_")]

    classes = np.sort(np.unique(y_true))

    for concept in concept_cols:
        concept_vals = meta_df[concept].astype(bool).values

        # 1) Overall: concept == 0 and concept == 1 across ALL samples
        for val in [0, 1]:
            m = concept_vals == bool(val)
            yt, yp, pr = y_true[m], y_pred[m], y_prob_pos[m]

            rows.append(
                dict(
                    task="binary",
                    slice="overall_by_concept",
                    concept=concept,
                    value=int(val),
                    class_label="all",
                    n=int(m.sum()),
                    acc=_acc(yt, yp),
                    balanced_acc=(
                        float(balanced_accuracy_score(yt, yp))
                        if (len(yt) and len(np.unique(yt)) > 1)
                        else float("nan")
                    ),
                    auroc=_maybe_auroc_binary(yt, pr),
                )
            )

        # 2) Per-class: for each class, for concept == 0 and concept == 1
        for cls in classes:
            m_cls = y_true == cls
            for val in [0, 1]:
                m = m_cls & (concept_vals == bool(val))
                yt, yp, pr = y_true[m], y_pred[m], y_prob_pos[m]

                rows.append(
                    dict(
                        task="binary",
                        slice="per_class_by_concept",
                        concept=concept,
                        value=int(val),
                        class_label=int(cls),
                        n=int(m.sum()),
                        # In this slice, yt is (mostly) single-class, so acc ~ recall for that class.
                        acc=_acc(yt, yp),
                        balanced_acc=(
                            float(balanced_accuracy_score(yt, yp))
                            if (len(yt) and len(np.unique(yt)) > 1)
                            else float("nan")
                        ),
                        # Usually NaN (single-class), but we stay consistent:
                        auroc=_maybe_auroc_binary(yt, pr),
                    )
                )

    return pd.DataFrame(rows)


def subgroup_metrics_multiclass_concepts(
    meta_df, y_true_0based, y_pred_0based, concept_cols=None
):
    """
    For a multiclass problem (labels 0..C-1):

    For each concept column (has_*):
      1) Overall (all classes): metrics for concept == 0 and concept == 1.
      2) Per-class: for each class c, metrics for concept == 0 and concept == 1
         restricted to samples where y_true == c.

    Returns: pd.DataFrame with one row per (concept, value, optional class).
    """
    rows = []
    y_true = np.asarray(y_true_0based)
    y_pred = np.asarray(y_pred_0based)

    if concept_cols is None:
        concept_cols = [c for c in meta_df.columns if c.startswith("has_")]

    classes = np.sort(np.unique(y_true))

    def add_row(tag_slice, concept, value, mask, cls_label):
        yt, yp = y_true[mask], y_pred[mask]
        rows.append(
            dict(
                task="multiclass",
                slice=tag_slice,
                concept=concept,
                value=int(value),
                class_label=("all" if cls_label is None else int(cls_label)),
                n=int(mask.sum()),
                acc=_acc(yt, yp),
                balanced_acc=(
                    float(balanced_accuracy_score(yt, yp))
                    if (len(yt) and len(np.unique(yt)) > 1)
                    else float("nan")
                ),
            )
        )

    for concept in concept_cols:
        concept_vals = meta_df[concept].astype(bool).values

        # 1) Overall across all classes
        for val in [0, 1]:
            m = concept_vals == bool(val)
            add_row("overall_by_concept", concept, val, m, cls_label=None)

        # 2) Per-class
        for cls in classes:
            m_cls = y_true == cls
            for val in [0, 1]:
                m = m_cls & (concept_vals == bool(val))
                add_row("per_class_by_concept", concept, val, m, cls_label=cls)

    return pd.DataFrame(rows)
