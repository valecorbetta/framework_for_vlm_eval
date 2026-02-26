from omegaconf import DictConfig
from hydra.utils import instantiate

def build_experiment(cfg: DictConfig):
    return instantiate(cfg.EXP._class, cfg)
