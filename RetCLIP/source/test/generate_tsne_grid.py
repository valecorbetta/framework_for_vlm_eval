"""
generate_tsne_grid.py

Generates individual t-SNE plots for VLM Baseline and MICA
at specified spurious fractions, for both test conditions (none, inverted).
Colored by DR grade with an ordinal sequential palette.

Usage:
    python generate_tsne_grid.py

Output structure:
    OUT_DIR/
      {model}_{overlay}_pct{pct}_split{split}.png
      e.g. vlm_none_pct000_split0.png
           mica_inverted_pct100_split0.png
"""

# ── CONFIG — edit these ───────────────────────────────────────────────────────

SPLIT = 0
PCT_LEVELS = [0, 25, 50, 75, 100]
OVERLAYS = ["none", "inverted"]  # test conditions

VLM_ROOT = "PATH_TO_VLM_RES"
MICA_ROOT = "PATH_TO_MICA_RES"

HYDRA_CONFIG_PATH = "PATH_TO_CONF_FOLDER"
HYDRA_CONFIG_NAME = "config"

CSV_TEST = "PATH_TO_TEST_CSV"

OUT_DIR = (
    "PATH_TO_SAVE_RES"
)

TSNE_SEED = 42
TSNE_PERPLEXITY = 30
DPI = 200

# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.manifold import TSNE
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


# ── ORDINAL DR PALETTE ────────────────────────────────────────────────────────
# Sequential yellow → orange → red to preserve severity ordering.
# Grade 0 (healthy) = light, Grade 4 (severe PDR) = dark red.

GRADE_COLOURS = [
    "#440154",  # Grade 0 — dark purple
    "#3b528b",  # Grade 1 — blue
    "#21918c",  # Grade 2 — teal
    "#5ec962",  # Grade 3 — green
    "#fde725",  # Grade 4 — bright yellow
]
GRADE_LABELS = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (PDR)",
]
N_GRADES = 5


# ── CONFIG & MODEL LOADING ───────────────────────────────────────────────────
# (Reused from your existing generate_tsne_figure.py)


def load_cfg(config_path: str, config_name: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        return compose(config_name)


def _vlm_ckpts(exp_dir: Path):
    mf = exp_dir / "checkpoints" / "manifest.json"
    if mf.exists():
        m = json.load(open(mf))
        head = exp_dir / m["classifier_head"] if "classifier_head" in m else None
        lora = exp_dir / m["lora_dir"] if "lora_dir" in m else None
    else:
        head = exp_dir / "checkpoints" / "best_classifier_head.ckpt"
        lora = exp_dir / "checkpoints" / "best_lora_adapter"
    return head, lora


def load_vlm(cfg, device, exp_dir, split, csv_test, overlay_cfg, ckpt_head, lora_dir):
    from RetCLIP.source.trainers.fundus_classifier_trainer import (
        FundusClassifierTrainer,
    )
    from RetCLIP.source.utils.checkpoints import CheckpointPaths

    trainer = FundusClassifierTrainer(
        cfg=cfg,
        device=device,
        exp_dir=exp_dir,
        seed=int(cfg.TRAIN.seed) + split,
        split_number=split,
        csv_train_path=csv_test,
        csv_val_path=csv_test,
        csv_test_path=csv_test,
        path_to_images=Path(cfg.PATHS.data_dir),
        overlay_cfg_train=overlay_cfg,
        overlay_cfg_test=overlay_cfg,
    )
    best = CheckpointPaths(
        best_classifier_head=ckpt_head,
        best_lora_dir=lora_dir if lora_dir and lora_dir.exists() else None,
    )
    cfg_model = OmegaConf.merge(cfg.MODEL._class, {"vision_encoder": {"lora": False}})
    model = instantiate(cfg_model).float().to(device)
    model = trainer.load_for_test(model, best)
    return trainer, model.eval()


def load_mica_cfg(exp_dir: Path) -> DictConfig:
    candidates = [
        exp_dir / "mica_stage2" / ".hydra" / "config.yaml",
        exp_dir / ".hydra" / "config.yaml",
        exp_dir / "mica_stage1" / ".hydra" / "config.yaml",
        # Config may be at the experiment root (above pct_*/split* dirs)
        exp_dir.parent.parent / ".hydra" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return OmegaConf.load(p)
    raise FileNotFoundError(f"No MICA config found under {exp_dir}")


def load_mica(
    cfg, device, exp_dir, split, csv_test, overlay_cfg, stage1_lora, stage2_heads
):
    from RetCLIP.source.trainers.mica_trainer_stage2 import MICAStage2CBMTrainer
    from RetCLIP.source.utils.checkpoints import CheckpointPaths

    cfg_local = load_mica_cfg(exp_dir)
    OmegaConf.update(
        cfg_local, "MODEL.stage_2.stage_1_lora_dir", str(stage1_lora), force_add=True
    )
    OmegaConf.update(cfg_local, "MODEL.stage_2.stage_1_ckpt", None, force_add=True)
    for key in ("PATHS", "TRAIN"):
        if key in cfg and key not in cfg_local:
            OmegaConf.update(
                cfg_local,
                key,
                OmegaConf.to_container(cfg[key], resolve=True),
                force_add=True,
            )

    trainer = MICAStage2CBMTrainer(
        cfg=cfg_local,
        device=device,
        exp_dir=exp_dir,
        seed=int(cfg_local.TRAIN.seed) + split,
        split_number=split,
        csv_train_path=csv_test,
        csv_val_path=csv_test,
        csv_test_path=csv_test,
        path_to_images=Path(cfg_local.PATHS.data_dir),
        overlay_cfg_train=overlay_cfg,
        overlay_cfg_test=overlay_cfg,
    )
    best = CheckpointPaths(best_classifier_head=stage2_heads)
    model = trainer.build_model().to(device)
    model = trainer.load_for_test(model, best)
    return trainer, model.eval()


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────


@torch.no_grad()
def extract_vlm_features(model, loader, device):
    feats, labels = [], []
    for batch in loader:
        feats.append(model.encode_image(batch["x"].to(device)).cpu().numpy())
        labels.append(batch["y"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def extract_mica_features(model, loader, device):
    feats, labels = [], []
    for batch in loader:
        feats.append(model._encode(batch["x"].to(device)).cpu().numpy())
        labels.append(batch["y"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ── t-SNE ─────────────────────────────────────────────────────────────────────


def run_tsne(features, seed=TSNE_SEED, perplexity=TSNE_PERPLEXITY):
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)


def _grade_legend():
    return [
        mpatches.Patch(
            facecolor=GRADE_COLOURS[g],
            edgecolor="0.3",
            linewidth=0.5,
            label=GRADE_LABELS[g],
        )
        for g in range(N_GRADES)
    ]


def plot_tsne(emb, labels, title, out_path, figsize=(4.5, 4.5)):
    """Save an individual t-SNE plot with ordinal DR coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot from highest to lowest grade so severe cases are on top
    for g in reversed(range(N_GRADES)):
        m = labels == g
        if not m.any():
            continue
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            c=GRADE_COLOURS[g],
            s=14,
            alpha=0.7,
            edgecolors="0.3",
            linewidths=0.3,
            zorder=g + 1,
            rasterized=True,
            label=GRADE_LABELS[g],
        )

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.legend(
        handles=_grade_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=7.5,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_root = Path(VLM_ROOT)
    mica_root = Path(MICA_ROOT)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading config...")
    cfg = load_cfg(HYDRA_CONFIG_PATH, HYDRA_CONFIG_NAME)
    OmegaConf.set_struct(cfg, False)
    from RetCLIP.source.experiments.context import build_overlay_cfg

    csv_test = Path(CSV_TEST)

    # Tokenizer for MICA collate (hardcoded — avoids Hydra config resolution issues)
    from RetCLIP.source.data.dataset_FGADR import mica_collate_fulltokenizer
    from RetCLIP.source.model.model import FullTokenizer

    tok = FullTokenizer()
    max_len = 33  # FGADR concept dataset: text.word_num
    mc_fn = lambda b: mica_collate_fulltokenizer(b, tok, max_len)

    # Dataset targets for switching between VLM and MICA
    VLM_DATASET_TARGET = "RetCLIP.source.data.dataset_FGADR.FGADRDataset"
    MICA_DATASET_TARGET = "RetCLIP.source.data.dataset_FGADR.FGADRConceptDataset"

    for pct in PCT_LEVELS:
        pct_str = f"pct_{pct:03d}"
        print(f"\n{'='*60}")
        print(f"  pct = {pct}%")
        print(f"{'='*60}")

        # ── Resolve checkpoint paths ──
        vlm_split = vlm_root / pct_str / f"split{SPLIT}"
        mica_split = mica_root / pct_str / f"split{SPLIT}"

        vh, vl = _vlm_ckpts(vlm_split)
        ms1_lora = mica_split / "mica_stage1" / "checkpoints" / "best_lora_adapter"
        ms2_heads = mica_split / "mica_stage2" / "checkpoints" / "best_stage2_heads.pt"

        for overlay in OVERLAYS:
            print(f"\n  overlay = {overlay}")

            # Build test overlay config:
            #   - "none":     mode=none (no artifacts applied, percent irrelevant)
            #   - "inverted": mode=inverted, percent matches training pct
            test_ov = build_overlay_cfg(cfg, "test", pct / 100.0)
            OmegaConf.update(test_ov, "mode", overlay)

            # ── VLM ──
            OmegaConf.update(cfg, "DATASET._class._target_", VLM_DATASET_TARGET)
            OmegaConf.update(cfg, "DATASET._class.return_masks", False)
            OmegaConf.update(cfg, "DATASET._class.mask_root", None)
            print(f"    Loading VLM Baseline...")
            vt, vm = load_vlm(cfg, device, vlm_split, SPLIT, csv_test, test_ov, vh, vl)
            lv = vt.build_test_loader()
            # We only need features — disable mask loading to avoid path errors
            lv.dataset.return_masks = False
            lv.dataset.mask_root = None

            print(f"    Extracting VLM features...")
            vf, vl_ = extract_vlm_features(vm, lv, device)

            # Save features for later re-plotting without GPU
            feat_path = out_dir / f"vlm_{overlay}_pct{pct:03d}_split{SPLIT}.npz"
            np.savez_compressed(feat_path, features=vf, labels=vl_)

            print(f"    Running t-SNE...")
            ve = run_tsne(vf)

            # Save t-SNE embeddings for instant re-plotting
            np.savez_compressed(
                out_dir / f"vlm_{overlay}_pct{pct:03d}_split{SPLIT}_tsne.npz",
                embeddings=ve,
                labels=vl_,
            )

            plot_tsne(
                ve,
                vl_,
                title=f"VLM Baseline — {overlay} — {pct}% spurious",
                out_path=out_dir / f"vlm_{overlay}_pct{pct:03d}_split{SPLIT}.png",
            )

            # Free GPU memory
            del vm, vt, lv
            torch.cuda.empty_cache()

            # ── MICA ──
            OmegaConf.update(cfg, "DATASET._class._target_", MICA_DATASET_TARGET)
            OmegaConf.update(cfg, "DATASET._class.return_masks", False)
            OmegaConf.update(cfg, "DATASET._class.mask_root", None)
            print(f"    Loading MICA...")
            mt, mm = load_mica(
                cfg, device, mica_split, SPLIT, csv_test, test_ov, ms1_lora, ms2_heads
            )
            lm = mt.build_test_loader(collate_fn=mc_fn)
            lm.dataset.return_masks = False
            lm.dataset.mask_root = None

            print(f"    Extracting MICA features...")
            mf, ml_ = extract_mica_features(mm, lm, device)

            # Save features for later re-plotting without GPU
            feat_path = out_dir / f"mica_{overlay}_pct{pct:03d}_split{SPLIT}.npz"
            np.savez_compressed(feat_path, features=mf, labels=ml_)

            print(f"    Running t-SNE...")
            me = run_tsne(mf)

            # Save t-SNE embeddings for instant re-plotting
            np.savez_compressed(
                out_dir / f"mica_{overlay}_pct{pct:03d}_split{SPLIT}_tsne.npz",
                embeddings=me,
                labels=ml_,
            )

            plot_tsne(
                me,
                ml_,
                title=f"MICA — {overlay} — {pct}% spurious",
                out_path=out_dir / f"mica_{overlay}_pct{pct:03d}_split{SPLIT}.png",
            )

            del mm, mt, lm
            torch.cuda.empty_cache()

    # ── Summary ──
    n_plots = len(PCT_LEVELS) * len(OVERLAYS) * 2  # 2 models
    print(f"\n[done] Generated {n_plots} t-SNE plots in {out_dir}/")
    print(f"  Naming: {{model}}_{{overlay}}_pct{{pct:03d}}_split{SPLIT}.png")


if __name__ == "__main__":
    main()
