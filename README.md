# Beyond Clean Test Sets: Spurious Correlations in Medical Vision-language Models and the Role of Concept Supervision [MICCAI '26]

This repository provides the code for a controlled evaluation framework that tests the robustness of domain-specific medical vision–language models (VLMs) to spurious correlations, using synthetic artifacts and concept-based supervision across two clinical domains.

This repository accompanies the MICCAI 2026 paper:

> Corbetta, V., et al. *Beyond Clean Test Sets: Spurious Correlations in Medical Vision-language Models and the Role of Concept Supervision.* MICCAI 2026.

The paper has two released artifacts:
- **Code** — this repository.
- **BI-RADS annotations on EMBED** — tabular and pixel-wise annotations of 400 images (100 cases) from the EMBED dataset, released separately. See [Data and Annotations](#data-and-annotations).

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

## Data and Annotations

The codebase uses two datasets, both obtained from their original distributors. Neither is redistributed here.

### EMBED (mammography)
The MammoCLIP experiments use the [Emory BrEast Imaging Dataset (EMBED)](https://github.com/Emory-HITI/EMBED_Open_Data). Access to EMBED requires registration and agreement to the [EMBED Research Use Agreement](https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/EMBED_license.md).

### FGADR (diabetic retinopathy)
The RetCLIP experiments use the FGADR dataset; please refer to the [FGADR project page](https://csyizhou.github.io/FGADR/) for access. 

### BI-RADS annotations on EMBED *(released with this paper)*

For our mammography experiments we release BI-RADS lexicon annotations for **400 images (100 cases)** from EMBED:

- **Tabular annotations** (CSV) — structured BI-RADS lexicon labels per image
- **Pixel-wise segmentation masks** (.nrrd) — registered to the original EMBED DICOMs

The annotations are released under terms mirroring the EMBED Research Use Agreement, with the written permission of the EMBED dataset creators (Emory University School of Medicine). They **are not usable without independent access to EMBED**.

**Access:**
- Zenodo (restricted access, DOI): [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
- Documentation and access request form: [google form]

The annotations repository contains the full license text, access procedure, and a demonstration notebook for loading the annotations.

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
- `path_to_annotations`: path to the BI-RADS annotations downloaded from Zenodo *(MammoCLIP only)*

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

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{corbetta2026beyond,
  title={Beyond Clean Test Sets: Spurious Correlations in Medical Vision-language Models and the Role of Concept Supervision},
  author={Valentina Corbetta and Portaluri, Antonio and Ze, Muzhen and Boeke, Daniël and Beets-Tan, Regina and Lachi, Veronica and Wetzer, Elisabeth and Jenssen, Robert and Wilson Silva and Kristoffer Wickstr{\o}m},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026}
}
```


If you use the annotations on EMBED please cite:

```bibtex

@article{jeong2023emory,
  title={The EMory BrEast imaging Dataset (EMBED): A racially diverse, granular dataset of 3.4 million screening and diagnostic mammographic images},
  author={Jeong, Jiwoong J and Vey, Brianna L and Bhimireddy, Ananth and Kim, Thomas and Santos, Thiago and Correa, Ramon and Dutt, Raman and Mosunjac, Marina and Oprea-Ilies, Gabriela and Smith, Geoffrey and others},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={1},
  pages={e220047},
  year={2023},
  publisher={Radiological Society of North America}
}

@inproceedings{corbetta2026beyond,
  title={Beyond Clean Test Sets: Spurious Correlations in Medical Vision-language Models and the Role of Concept Supervision},
  author={Valentina Corbetta and Portaluri, Antonio and Ze, Muzhen and Boeke, Daniël and Beets-Tan, Regina and Lachi, Veronica and Wetzer, Elisabeth and Jenssen, Robert and Wilson Silva and Kristoffer Wickstr{\o}m},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026}
}

@dataset{corbetta2026embed_annotations,
  title     = {BI-RADS Annotations for the EMBED Dataset},
  author    = {Corbetta, Valentina and others},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```


## Acknowledgements

This codebase builds upon several open-source projects:

- [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) — Vision–language foundation model for mammography
- [RET-CLIP](https://github.com/sStonemason/RET-CLIP) — Retinal image foundation model pre-trained with clinical reports
- [MICA](https://github.com/Tommy-Bie/MICA) — Multi-level Image-Concept Alignment for explainable diagnosis
- [Post-hoc CBM](https://github.com/mertyg/post-hoc-cbm) — Post-hoc Concept Bottleneck Models


