#!/usr/bin/env python3
"""
Main testing/evaluation entry point using Hydra configuration.

Usage:
    python src/test.py checkpoint=path/to/checkpoint.pth
    python src/test.py checkpoint=path/to/checkpoint.pth logging.use_wandb=true
    python src/test.py checkpoint=path/to/checkpoint.pth dataset.test_sequences=[02]
"""

import os
import sys
import logging
import pickle
import matplotlib.pyplot as plt
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# Ensure the project root is on sys.path so 'src' is importable
# even after Hydra changes the working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.utils.io import seed_normalize_factors

from src.core.torch_iekf import TorchIEKF
from src.evaluation.evaluator import evaluate_sequence, format_metrics
from src.evaluation.visualization import (
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_error_timeline,
    plot_covariance_timeline,
)
from src.data.kitti_dataset import KITTIDataset


# Keep backward-compatible aliases
def test_sequence(iekf, dataset, dataset_name, cfg):
    """Run filter on a single sequence and compute metrics.

    Thin wrapper around ``evaluate_sequence`` for backward compatibility.
    The *cfg* argument is accepted but not used.
    """
    return evaluate_sequence(iekf, dataset, dataset_name)


def print_metrics(results, dataset_name):
    """Print metrics for a sequence."""
    print("\n" + format_metrics(results, dataset_name))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        print(
            "Error: No checkpoint path provided. Use: python src/test.py checkpoint=path/to/model.pth"
        )
        sys.exit(1)

    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Build model
    model_cfg = OmegaConf.to_container(cfg.get("model", {}), resolve=True)
    iekf = TorchIEKF.build_from_cfg(model_cfg)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Use strict=False to gracefully handle checkpoints saved by older code
    # that stored normalization buffers (u_loc, u_std) directly on the model.
    missing, unexpected = iekf.load_state_dict(state_dict, strict=False)
    if unexpected:
        logging.getLogger(__name__).warning(
            "Ignored unexpected key(s) in checkpoint: %s", unexpected
        )
    if missing:
        logging.getLogger(__name__).warning(
            "Missing key(s) not found in checkpoint: %s", missing
        )
    # Legacy checkpoints were saved in double precision; cast to float32 so
    # the model is consistent with float32 input tensors at inference time.
    iekf = iekf.float()
    iekf.eval()

    # Build dataset
    paths_cfg = cfg.get("paths", {})
    seed_normalize_factors(paths_cfg)
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    dataset_cfg["path_data_save"] = paths_cfg.get("data", "../data")
    dataset_cfg["path_results"] = paths_cfg.get("results", "../results")
    dataset_cfg["path_temp"] = paths_cfg.get("temp", "../temp")
    dataset = KITTIDataset(dataset_cfg, split="test")

    # Load normalization statistics into the model so networks receive
    # normalized IMU input. Must be called after dataset is built (which
    # loads normalize_factors.p from paths.temp) and after iekf.float().
    if dataset.normalize_factors is None:
        raise RuntimeError(
            f"normalize_factors.p not found in {dataset_cfg['path_temp']}. "
            "Run data preparation (read_data=1) first to generate it."
        )
    iekf.get_normalize_u(dataset)

    # Optional WandB
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=cfg.get("logging", {}).get("project", "ai-imu-dr"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"test-{cfg.get('experiment', {}).get('name', 'default')}",
            job_type="test",
        )

    # Run on all test sequences
    results_dir = paths_cfg.get("results", "../results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    for dataset_name in dataset.datasets_test:
        try:
            results = test_sequence(iekf, dataset, dataset_name, cfg)
            all_results[dataset_name] = results
            print_metrics(results, dataset_name)

            # Save results pickle (backward compatible)
            result_path = os.path.join(results_dir, f"{dataset_name}_filter.p")
            with open(result_path, "wb") as f:
                pickle.dump(results, f)

            # Generate visualizations
            p = results["p"]
            p_gt = results["p_gt"]

            p_imu = results.get("p_imu")
            fig_2d = plot_trajectory_2d(
                p, p_gt, seq_name=dataset_name, p_imu=p_imu
            )
            fig_2d.savefig(
                os.path.join(results_dir, f"{dataset_name}_trajectory_2d.png"),
                dpi=150,
            )

            fig_err = plot_error_timeline(
                np.linalg.norm(p_gt - p, axis=1),
                timestamps=results["t"] - results["t"][0],
                seq_name=dataset_name,
            )
            fig_err.savefig(
                os.path.join(results_dir, f"{dataset_name}_error.png"), dpi=150
            )

            if use_wandb:
                metrics = results["metrics"]
                log_dict = {
                    f"test/{dataset_name}/t_rel": metrics["rpe"]["t_rel"],
                    f"test/{dataset_name}/r_rel": metrics["rpe"]["r_rel"],
                    f"test/{dataset_name}/ate_rmse": metrics["ate"]["rmse"],
                    f"test/{dataset_name}/orient_mean_deg": metrics[
                        "orientation_error"
                    ]["mean_deg"],
                    f"test/{dataset_name}/trajectory_2d": wandb.Image(fig_2d),
                    f"test/{dataset_name}/error_timeline": wandb.Image(
                        fig_err
                    ),
                }
                if "metrics_imu" in results:
                    mi = results["metrics_imu"]
                    log_dict.update(
                        {
                            f"test/{dataset_name}/imu_t_rel": mi["rpe"][
                                "t_rel"
                            ],
                            f"test/{dataset_name}/imu_r_rel": mi["rpe"][
                                "r_rel"
                            ],
                            f"test/{dataset_name}/imu_ate_rmse": mi["ate"][
                                "rmse"
                            ],
                        }
                    )
                wandb.log(log_dict)

            plt.close("all")

        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    if all_results:
        t_rels = [
            r["metrics"]["rpe"]["t_rel"]
            for r in all_results.values()
            if not np.isnan(r["metrics"]["rpe"]["t_rel"])
        ]
        r_rels = [
            r["metrics"]["rpe"]["r_rel"]
            for r in all_results.values()
            if not np.isnan(r["metrics"]["rpe"]["r_rel"])
        ]
        ate_rmses = [r["metrics"]["ate"]["rmse"] for r in all_results.values()]

        print(f"\n{'='*60}")
        print("Overall Summary")
        print(f"{'='*60}")
        if t_rels:
            print(f"  Mean t_rel: {np.mean(t_rels):.3f}%")
        if r_rels:
            print(f"  Mean r_rel: {np.mean(r_rels):.4f} deg/m")
        if ate_rmses:
            print(f"  Mean ATE RMSE: {np.mean(ate_rmses):.2f}m")

        # IMU baseline summary
        imu_t = [
            r["metrics_imu"]["rpe"]["t_rel"]
            for r in all_results.values()
            if "metrics_imu" in r
            and not np.isnan(r["metrics_imu"]["rpe"]["t_rel"])
        ]
        imu_r = [
            r["metrics_imu"]["rpe"]["r_rel"]
            for r in all_results.values()
            if "metrics_imu" in r
            and not np.isnan(r["metrics_imu"]["rpe"]["r_rel"])
        ]
        if imu_t or imu_r:
            print(f"  --- IMU integration baseline ---")
            if imu_t:
                print(f"  Mean t_rel: {np.mean(imu_t):.3f}%")
            if imu_r:
                print(f"  Mean r_rel: {np.mean(imu_r):.4f} deg/m")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
