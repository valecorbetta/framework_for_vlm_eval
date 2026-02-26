# runner.py
import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from source.utils.misc import preprocess_paths, set_seed
from source.experiments.context import RunContext
from source.experiments.dispatch import build_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_paths(cfg.PATHS)
    set_seed(int(cfg.TRAIN.seed))

    hydra_subdir = Path(HydraConfig.get().sweep.dir) / HydraConfig.get().sweep.subdir

    logging.basicConfig(
        filename=hydra_subdir / "logs.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(OmegaConf.to_yaml(cfg))

    ctx = RunContext(
        cfg=cfg,
        device=device,
        hydra_subdir=hydra_subdir,
        split_root=cfg.PATHS.path_to_split_csvs,
        images_root=cfg.PATHS.data_dir,
        overlay_percentages=cfg.EXP.overlay_percentages,
        num_splits=cfg.TRAIN.num_splits,
    )

    exp = build_experiment(cfg)

    mode = str(cfg.MODE.mode).lower()

    logging.info(f"=== EXP.name={cfg.EXP.name} | MODE.mode={mode} ===")

    exp.run(ctx)

    logging.info(f"=== EXP.name={cfg.EXP.name} | All runs completed ===")


if __name__ == "__main__":
    main()
