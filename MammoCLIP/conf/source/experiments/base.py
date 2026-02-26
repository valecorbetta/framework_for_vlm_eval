import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class OptunaResult:
    """
    Generic return object for experiments that run Optuna.
    Store either:
      - best_params: dict-like
      - best_params_path: path to a saved yaml/json
      - plus any experiment-specific artifacts to reuse downstream
    """

    best_params: Optional[dict] = None
    best_params_path: Optional[Path] = None
    extra: Optional[dict] = None


class BaseExperiment:
    """
    Unified experiment interface.

    Every experiment supports:
      - train (normal sweeps over pct/splits)
      - test_only (evaluate from existing root dir)
      - optuna (cross-val search, return OptunaResult)

    And the orchestration supports:
      - optuna + after_optuna steps (full_train, test)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    # -------- required per-experiment methods --------
    def run_train(
        self, ctx: Any, *, best: Optional[OptunaResult] = None, full_train: bool = False
    ) -> None:
        raise NotImplementedError

    def run_test_only(self, ctx: Any) -> None:
        raise NotImplementedError

    def run_optuna(self, ctx: Any) -> OptunaResult:
        raise NotImplementedError

    def _save_best_params(self, opt_dir: Path, params: dict[str, Any]) -> Path:
        out = opt_dir / "best_params.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(params), out)
        return out

    def _load_best_params(self, path: Path) -> dict[str, Any]:
        return dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    # -------- shared orchestration --------
    def _after_optuna_steps(self) -> list[str]:
        exp_cfg = getattr(self.cfg, "MODE", None)
        steps = []
        if exp_cfg is not None and hasattr(exp_cfg, "after_optuna"):
            steps = (
                list(exp_cfg.after_optuna) if exp_cfg.after_optuna is not None else []
            )
        # normalize to lowercase strings
        return [str(s).strip().lower() for s in steps]

    def run(self, ctx: Any) -> None:
        """
        Main entrypoint used by runner.py.
        Reads:
          cfg.EXP.mode: train | test_only | optuna
          cfg.EXP.after_optuna: [] or [full_train,test]
        """
        mode = str(getattr(self.cfg.MODE, "mode", "train")).strip().lower()
        if mode not in {"train", "test_only", "optuna"}:
            raise ValueError(
                f"EXP.mode must be one of: train | test_only | optuna. Got: {mode}"
            )

        if mode == "train":
            self.run_train(ctx, best=None, full_train=False)
            return

        elif mode == "test_only":
            self.run_test_only(ctx)
            return

        elif mode == "optuna":
            best: OptunaResult = self.run_optuna(ctx)
            steps: list[str] = self._after_optuna_steps()
            if not steps:
                logging.info(
                    "[EXPERIMENT] Optuna complete. No after_optuna steps configured."
                )
                return
            else:
                for step in steps:
                    if step == "full_train":
                        # interpret full_train as: run the full pct/splits sweep using best
                        self.run_train(ctx, best=best)
                    else:
                        raise ValueError(f"Unknown after_optuna step: {step}")
