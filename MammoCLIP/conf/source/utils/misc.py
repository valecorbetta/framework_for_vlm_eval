import hashlib
import os
from omegaconf import ListConfig
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import random
import numpy as np


def set_seed(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_paths(paths):
    for k, v in paths.items():
        if isinstance(v, (list, ListConfig)):
            paths[k] = [Path(p) for p in v]
        else:
            paths[k] = Path(v)


def split_paths(base_dir: Path, split_id: int) -> tuple[Path, Path, Path]:
    """
    Expect structure:
      base_dir/
        test.csv
        splits/
          seed_1/train.csv, val.csv
          seed_2/train.csv, val.csv
          ...
    """
    base_dir = Path(base_dir)
    seed_dir = base_dir / f"{split_id}"
    train_csv = seed_dir / "train.csv"
    val_csv = seed_dir / "val.csv"
    test_csv = base_dir / "test.csv"  # (not used by train(...), but handy to have)
    for p in [train_csv, val_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    return train_csv, val_csv, test_csv


def hash_prob(uid: str, seed: int = 42) -> float:
    """Make the probability that image uid gets a shape inpainted reproducible"""
    h = hashlib.md5((str(uid) + str(seed)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def update_state_dict(state_dict: dict, key_to_replace: str) -> dict:
    return {
        k.replace(key_to_replace, ""): v
        for k, v in state_dict.items()
        if k.startswith(key_to_replace)
    }


def freeze_params(model: nn.Module) -> None:
    print(f"Freezing parameters of {model}...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = (
                torch.tensor(np.concatenate(val_list, axis=0), requires_grad=False)
                .float()
                .to(device)
            )
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        # Ensure each vector is 1D (D,), then stack -> (N, D)
        all_vectors = [np.asarray(v).reshape(-1) for v in all_vectors]
        vecs = np.stack(all_vectors, axis=0)  # (N, D)
        self.concept_info.vectors = (
            torch.tensor(vecs, requires_grad=False).float().to(device)
        )
        print(f"{self.concept_info.vectors=}")
        print(f"{self.concept_info.vectors.shape=}")
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True
        ).detach()
        self.concept_info.intercepts = (
            torch.tensor(np.concatenate(all_intercepts, axis=0), requires_grad=False)
            .float()
            .to(device)
        )
        print(f"{self.concept_info.intercepts=}")
        print(f"{self.concept_info.intercepts.shape=}")
        self.concept_info.concept_names = concept_names
        print(f"{self.concept_info.concept_names=}")
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
