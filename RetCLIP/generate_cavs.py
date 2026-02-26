# generate_cavs.py

from copy import deepcopy
import json
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# your imports
from RetCLIP.source.data.dataset_FGADR import FGADRDataset
from RetCLIP.source.utils.misc import preprocess_paths, set_seed, split_paths
from RetCLIP.source.utils.data_utils import prepare_fgadr_augmentations


def get_concept_names_from_df(df) -> List[str]:
    return [c for c in df.columns if c.startswith("has_")]


def extract_features_and_concepts(
    model,
    dataloader: DataLoader,
    device: torch.device,
    concept_type: str = "clinical",
    artifact_raw_names: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run the (frozen) image encoder on all images and collect:
      - X: image features, shape (N, D)
      - concept labels per concept, shape (N,)
    Returns a dict:
       {
         "features": np.ndarray (N,D),
         "concept_labels": {concept_name: np.ndarray (N,) in {0,1}},
         "filenames": list[str]
       }
    """
    model.eval()
    all_feats: List[np.ndarray] = []
    all_filenames: List[str] = []

    # materialize dataloader so we can do multiple passes
    batch_list: List[Dict[str, Any]] = []
    first_batch = None
    for b in dataloader:
        if first_batch is None:
            first_batch = b
        batch_list.append(b)

    if first_batch is None:
        raise RuntimeError("Dataloader is empty.")

    # -------------------------
    # 1) Decide which concepts
    # -------------------------
    if concept_type == "clinical":
        # same behaviour as before: all has_* keys
        concept_keys = [k for k in first_batch.keys() if k.startswith("has_")]
        concept_buffers: Dict[str, List[int]] = {ck: [] for ck in concept_keys}
        type_to_ck = None  # not used

    elif concept_type == "spurious":
        # expect spurious_applied (bool) and spurious_type (str, "none" for no artifact)
        if "spurious_applied" not in first_batch or "spurious_type" not in first_batch:
            raise KeyError(
                "For concept_type='spurious' expected keys "
                "'spurious_applied' and 'spurious_type' in the dataset batches."
            )

        # Decide which raw artifact names to use
        if artifact_raw_names is not None and len(artifact_raw_names) > 0:
            # use the explicit list you pass in
            raw_names = sorted(set(artifact_raw_names))
        else:
            # discover from data, excluding "none"
            raw_names_set = set()
            for batch in batch_list:
                sp_types = batch["spurious_type"]
                if isinstance(sp_types, (list, tuple)):
                    vals = sp_types
                else:
                    vals = list(sp_types)
                for v in vals:
                    name = str(v)
                    if name != "none":
                        raw_names_set.add(name)
            raw_names = sorted(list(raw_names_set))

        # create concept keys like "artf_<name>"
        concept_keys = [f"artf_{name}" for name in raw_names]
        # map from raw spurious_type string -> concept key
        type_to_ck: Dict[str, str] = {name: f"artf_{name}" for name in raw_names}

        concept_buffers = {ck: [] for ck in concept_keys}

    else:
        raise ValueError(f"Unknown concept_type: {concept_type}")

    # -------------------------
    # 2) Extract features + labels
    # -------------------------
    with torch.no_grad():
        for batch in batch_list:
            x = batch["x"].to(device, non_blocking=True)

            # image features
            feats = model.encode_image(x)  # (B, D)
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats)

            # filenames
            filenames = batch["filename"]
            all_filenames.extend(list(filenames))

            if concept_type == "clinical":
                for ck in concept_keys:
                    concept_buffers[ck].extend(
                        batch[ck].cpu().numpy().astype(int).tolist()
                    )

            elif concept_type == "spurious":
                sp_applied = batch["spurious_applied"]
                sp_types = batch["spurious_type"]

                # normalize sp_applied to numpy array of ints 0/1
                if hasattr(sp_applied, "cpu"):
                    sp_applied = sp_applied.cpu().numpy()
                sp_applied = np.asarray(sp_applied).astype(int)

                # normalize sp_types to list[str]
                if isinstance(sp_types, (list, tuple)):
                    types_list = [str(t) for t in sp_types]
                else:
                    types_list = [str(t) for t in list(sp_types)]

                for a, t_str in zip(sp_applied, types_list):
                    # label=1 if this artifact was applied and type matches
                    # "none" will never match any raw_names, so will be 0 for all
                    for raw_name, ck in (type_to_ck or {}).items():
                        is_pos = int((a == 1) and (t_str == raw_name))
                        concept_buffers[ck].append(is_pos)

    X = np.concatenate(all_feats, axis=0)
    concept_labels = {
        ck: np.array(vals, dtype=np.int64) for ck, vals in concept_buffers.items()
    }

    return {
        "features": X,
        "concept_labels": concept_labels,
        "filenames": all_filenames,
    }


def train_single_cav(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Dict:
    """
    Train a linear SVM CAV for one concept with stratified CV.

    Returns:
      {
        "cav": np.ndarray(D,),
        "train_acc": float,
        "test_acc": float,      # mean CV accuracy
        "cv_auc": float or np.nan,
        "intercept": float,
        "norm": float,
        "margin_info": dict,
      }
    """
    # need both classes
    if len(np.unique(y)) < 2:
        return {
            "cav": None,
            "train_acc": np.nan,
            "test_acc": np.nan,
            "cv_auc": np.nan,
            "intercept": np.nan,
            "norm": np.nan,
            "margin_info": {},
        }

    skf = StratifiedKFold(
        n_splits=min(n_splits, np.bincount(y).min()),
        shuffle=True,
        random_state=random_state,
    )

    cv_accs = []
    cv_aucs = []

    # --- CV loop for evaluation --- #
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = LinearSVC(
            C=0.01,  # can tune
            class_weight="balanced",
            max_iter=5000,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        cv_accs.append(acc)

        try:
            scores = clf.decision_function(X_val)
            auc = roc_auc_score(y_val, scores)
        except Exception:
            auc = np.nan
        cv_aucs.append(auc)

    cv_acc = float(np.nanmean(cv_accs))
    cv_auc = float(np.nanmean(cv_aucs))

    # --- Final model on *all* data (for the actual CAV) --- #
    clf_full = LinearSVC(
        C=0.01,
        class_weight="balanced",
        max_iter=5000,
    )
    clf_full.fit(X, y)
    cav = clf_full.coef_.reshape(-1)  # (D,)
    intercept = float(clf_full.intercept_[0])
    norm = float(np.linalg.norm(cav))

    # training accuracy on the *full* data
    train_acc = float(clf_full.score(X, y))

    # --- Margin info (mimicking CBM) --- #
    # margins = (w·x + b) / ||w||
    if norm > 0:
        train_margins = ((X @ cav) + intercept) / norm
    else:
        train_margins = np.zeros_like(y, dtype=float)

    pos_mask = y == 1
    neg_mask = y == 0

    margin_info = {
        "max": float(np.max(train_margins)),
        "min": float(np.min(train_margins)),
        "pos_mean": (
            float(np.nanmean(train_margins[pos_mask])) if pos_mask.any() else np.nan
        ),
        "pos_std": (
            float(np.nanstd(train_margins[pos_mask])) if pos_mask.any() else np.nan
        ),
        "neg_mean": (
            float(np.nanmean(train_margins[neg_mask])) if neg_mask.any() else np.nan
        ),
        "neg_std": (
            float(np.nanstd(train_margins[neg_mask])) if neg_mask.any() else np.nan
        ),
        "q_90": float(np.quantile(train_margins, 0.9)),
        "q_10": float(np.quantile(train_margins, 0.1)),
        "pos_count": int(pos_mask.sum()),
        "neg_count": int(neg_mask.sum()),
        # you *can* include the raw margins if you want,
        # but MICA explicitly skips "train_margins" when building the tensors
        "train_margins": train_margins.tolist(),
    }

    return {
        "cav": cav,
        "train_acc": train_acc,
        "test_acc": cv_acc,
        "cv_auc": cv_auc,
        "intercept": intercept,
        "norm": norm,
        "margin_info": margin_info,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Usage (example):

      python generate_cavs.py \
        DATASET=fgadr \
        TRAIN=lora \
        MODEL=ConceptAlignedClassifier \
        TASK=retinopathy_grade

    Assumes that:
      - cfg.PATHS.path_to_split_csvs exists
      - cfg.PATHS.data_dir points to FGADR images
      - cfg.MODEL._class is ConceptAlignedClassifier with encode_image(x)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_paths(cfg.PATHS)
    set_seed(cfg.TRAIN.seed)
    concept_type = "clinical"
    hydra_subdir = (
        Path(HydraConfig.get().sweep.dir)
        / HydraConfig.get().sweep.subdir
        / "cavs"
        / concept_type
    )
    (hydra_subdir).mkdir(exist_ok=True, parents=True)
    (hydra_subdir / "logs").mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        filename=hydra_subdir / "logs" / "generate_cavs.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    logging.info("==== Generating CAVs ====")
    logging.info(OmegaConf.to_yaml(cfg))

    # ---- Build dataset on TRAIN split 0 (or concatenate all splits if you prefer) ----
    split_root = cfg.PATHS.path_to_split_csvs
    images_root = cfg.PATHS.data_dir
    overlay_percentages = [0.0, 0.25, 0.50, 0.75, 1.0]
    for pct in overlay_percentages:
        logging.info(f"---------- Overlay Percentage {int(pct*100)}% ----------")
        overlay_cfg_train = OmegaConf.to_container(
            cfg.DATASET.overlay_cfg_train, resolve=True
        )
        overlay_cfg_train = deepcopy(overlay_cfg_train)
        overlay_cfg_train["percent"] = pct
        overlay_cfg_train["enabled"] = bool(pct > 0) and bool(
            overlay_cfg_train.get("mode", "same")
        )
        for split in range(cfg.TRAIN.num_splits):
            # You can choose a specific split for CAVs; for now we'll use split 0's train CSV
            csv_train, _, _ = split_paths(split_root, split_id=split)

            # No overlay, no augmentations for CAVs
            _, preprocessing = prepare_fgadr_augmentations(seed=split)

            dataset: FGADRDataset | Any = instantiate(
                cfg.DATASET._class,
                csv_train,
                images_root,
                overlay_cfg_train,
                augmentations=None,
                preprocessing=preprocessing,
                label_mode=cfg.TASK.label_mode,
            )

            concept_names = get_concept_names_from_df(dataset.df)
            logging.info(f"Found concept columns: {concept_names}")
            # artifact_raw_names = [
            #     "out_of_focus_quarter",
            #     "reflection_double_dot",
            #     "eyelash_shadow",
            #     "illumination_semicircle",
            # ]
            loader = DataLoader(
                dataset,
                batch_size=cfg.TRAIN.batch_size,
                shuffle=False,
                num_workers=cfg.TRAIN.num_workers,
                pin_memory=True,
            )

            # ---- Instantiate model (frozen) ----
            model = instantiate(cfg.MODEL._class)
            model = model.to(device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            # ---- Extract features and concept labels ----
            feats_dict = extract_features_and_concepts(
                model,
                loader,
                device,
                concept_type=concept_type,
                artifact_raw_names=None,
            )
            X = feats_dict["features"]  # (N, D)
            concept_labels = feats_dict["concept_labels"]
            print(f"{concept_labels=}")
            filenames = feats_dict["filenames"]

            logging.info(f"Extracted features: X.shape = {X.shape}")
            concept_names = sorted(concept_labels.keys())
            logging.info(f"Using spurious concepts: {concept_names}")
            # ---- Train a CAV for each concept ----
            # ---- Train a CAV for each concept + build MICA-style dict ----
            cavs = {}
            stats = {}
            mica_concept_dict = {}  # this will become CAV_FILE

            for concept in concept_names:
                print(f"{concept}=")
                y = concept_labels[concept]
                class_counts = np.bincount(y, minlength=2)
                logging.info(
                    f"[{concept}] positives={class_counts[1]}, negatives={class_counts[0]}"
                )

                if class_counts.min() < 10:
                    logging.info(
                        f"[{concept}] Skipping due to too few samples in one class."
                    )
                    cavs[concept] = None
                    stats[concept] = {
                        "cv_acc": np.nan,
                        "cv_auc": np.nan,
                        "train_acc": np.nan,
                        "test_acc": np.nan,
                        "intercept": np.nan,
                        "norm": np.nan,
                    }
                    # You *could* skip this concept in the bank as well
                    continue

                logging.info(f"[{concept}] Training CAV...")
                res = train_single_cav(X, y, n_splits=5, random_state=cfg.TRAIN.seed)

                cavs[concept] = res["cav"]
                stats[concept] = {
                    "cv_acc": res["test_acc"],  # keep your old naming if you want
                    "cv_auc": res["cv_auc"],
                    "train_acc": res["train_acc"],
                    "test_acc": res["test_acc"],
                    "intercept": res["intercept"],
                    "norm": res["norm"],
                }

                # ---- MICA / CBM style entry ----
                # (tensor, train_acc, test_acc, intercept, margin_info)
                mica_concept_dict[concept] = (
                    res["cav"],
                    res["train_acc"],
                    res["test_acc"],
                    res["intercept"],
                    res["margin_info"],
                )

                logging.info(
                    f"[{concept}] CAV trained. "
                    f"train_acc={res['train_acc']:.3f}, "
                    f"test_acc={res['test_acc']:.3f}, "
                    f"cv_auc={res['cv_auc']:.3f}, "
                    f"intercept={res['intercept']:.4f}, norm={res['norm']:.4f}"
                )

            # ---- Save NPZ (optional, like before) ----
            pct_dir = hydra_subdir / f"{pct}"
            pct_dir.mkdir(exist_ok=True, parents=True)

            savez_kwargs = {
                "features": X,
                "concept_names": np.array(concept_names, dtype=object),
                "filenames": np.array(filenames, dtype=object),
            }
            for k, v in cavs.items():
                if v is not None:
                    savez_kwargs[f"cav_{k}"] = v
            np.savez(pct_dir / f"cavs_{split}.npz", **savez_kwargs)

            # ---- Save stats JSON (for your own analysis) ----
            with open(pct_dir / f"cav_stats_{split}.json", "w") as f:
                json.dump(stats, f, indent=2)

            # ---- Save MICA-style concept bank pickle ----
            cav_file = pct_dir / f"mica_concepts_split{split}.pkl"
            with open(cav_file, "wb") as f:
                pickle.dump(mica_concept_dict, f)

            logging.info(f"Saved MICA-style CAV bank to {cav_file}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
