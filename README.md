# Evaluating Domain-Specific Medical VLMs Under Spurious Correlations

This repository provides the code for a controlled evaluation framework that tests the robustness of domain-specific medical vision–language models (VLMs) to spurious correlations, using synthetic artifacts and concept-based supervision across two clinical domains.

## Overview

We inject synthetic confounders modelled after common acquisition artifacts into training images at parametrically varying prevalence, and evaluate models under two complementary test protocols:

- **Artifact removal**: all confounders removed at test time, probing whether models learned clinical features despite training alongside shortcuts.
- **Artifact inversion**: confounder–class associations cyclically shifted at test time, revealing the extent to which models follow artifacts over pathology.

Five architectures are compared, spanning a spectrum from no concept supervision to full multi-level image–concept alignment: Non-VLM baseline, VLM baseline, Multi-Task, PCBM, and MICA.

## Repository Structure

```
├── RetCLIP/          # Diabetic retinopathy grading (fundus photography)
│   ├── conf/         # Hydra configuration files
│   ├── source/       # Models, trainers, data, utils, evaluation
│   ├── runner.py
│   └── generate_cavs.py
│
├── MammoCLIP/        # BI-RADS-based assessment (mammography)
│   ├── conf/         # Hydra configuration files and source code
│   │   ├── conf/     # Hydra YAML configs
│   │   ├── source/   # Models, trainers, data, utils, evaluation
│   │   ├── runner.py
│   │   └── generate_cavs.py
```

## Setup

### Configuration

Before running experiments, update the placeholder paths in the PATHS config files:

- `RetCLIP/conf/PATHS/fgadr.yaml`
- `MammoCLIP/conf/conf/PATHS/embed.yaml`

Set the following to your local paths:

- `root_dir`: project root directory
- `data_dir`: path to image data
- `path_to_ckpt` / `path_to_mammoclip_ckpt`: path to pre-trained VLM checkpoint
- `path_to_cav_folder`: path to CAV output folder

### Dependencies

Core dependencies include PyTorch, Hydra, Optuna, PEFT (LoRA), timm, and transformers. See individual model configs for specific architecture requirements.

## Training

All experiments are managed through Hydra. To run an experiment:

```bash
cd RetCLIP  # or MammoCLIP/conf
python runner.py \
    EXP=fundus_classifier \      # or mammo_classifier, mica, multi_task, pcbm
    MODE=train \                 # or optuna, test_only
    TASK=dr_grading \            # or diagnosis
    SPURIOUS@DATASET.overlay_cfg_train=artifacts \
    SPURIOUS@DATASET.overlay_cfg_test=artifacts_none
```

### Hyperparameter Tuning

Shared hyperparameters (learning rate, LoRA rank and dropout) were tuned via [Optuna](https://github.com/optuna/optuna) at 0% spurious fraction on the VLM baseline and held fixed across all architectures. Architecture-specific parameters were tuned separately. To run Optuna tuning:

```bash
python runner.py EXP=fundus_classifier MODE=optuna
```

### Training Details

- **Optimiser**: AdamW with cosine annealing and linear warmup
- **Fundus (RetCLIP)**: 30 epochs on NVIDIA RTX 2080Ti GPUs
- **Mammography (MammoCLIP)**: 15 epochs on NVIDIA A6000 GPUs
- All backbones are adapted with LoRA
- Models are trained on five random train/validation splits with a fixed held-out test set; performance is averaged across splits

## Generating CAVs

For PCBM and MICA, Concept Activation Vectors must be generated before training:

```bash
python generate_cavs.py
```

## Evaluation

To run test-only evaluation with a trained model:

```bash
python runner.py EXP=fundus_classifier MODE=test_only
```

Evaluation scripts for aggregating results, generating plots, heatmaps, and confusion matrices are available in `source/test/`.

## Acknowledgements

This codebase builds upon several open-source projects:

- [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) — Vision–language foundation model for mammography
- [RET-CLIP](https://github.com/sStonemason/RET-CLIP) — Retinal image foundation model pre-trained with clinical reports
- [MICA](https://github.com/Tommy-Bie/MICA) — Multi-level Image-Concept Alignment for explainable diagnosis
- [Post-hoc CBM](https://github.com/mertyg/post-hoc-cbm) — Post-hoc Concept Bottleneck Models