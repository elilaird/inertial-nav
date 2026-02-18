#!/usr/bin/env python3
"""
Main training entry point using Hydra configuration.

Usage:
    python src/train.py                              # Default config
    python src/train.py training.epochs=100          # Override epochs
    python src/train.py model=iekf_learned_dynamics  # Different model
    python src/train.py logging.use_wandb=false      # Disable WandB
    python src/train.py training.debug.fast_dev_run=true  # Quick test
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


# Ensure the project root is on sys.path so 'src' is importable
# even after Hydra changes the working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.utils.io import seed_normalize_factors


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print resolved config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    from src.training.trainer import Trainer
    from src.data.kitti_dataset import KITTIDataset

    # Build dataset
    paths_cfg = cfg.get("paths")
    seed_normalize_factors(paths_cfg)
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset"), resolve=True)
    dataset_cfg["path_data_save"] = paths_cfg.get("data")
    dataset_cfg["path_results"] = paths_cfg.get("results")
    dataset_cfg["path_temp"] = paths_cfg.get("temp")

    dataset = KITTIDataset(dataset_cfg, split="train")

    print(f"Dataset loaded with {len(dataset)} samples.")

    # Build trainer
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer = Trainer(cfg_dict, dataset)

    # Resume from checkpoint if configured
    resume_cfg = cfg.get("training", {}).get("resume", {})
    if resume_cfg.get("enabled", False):
        ckpt_path = resume_cfg.get("checkpoint_path")
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            trainer.load_model(ckpt_path)

    # Train
    trainer.fit()


if __name__ == "__main__":
    main()
