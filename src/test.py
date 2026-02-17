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
import pickle

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure the project root is on sys.path so 'src' is importable
# even after Hydra changes the working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.torch_iekf import TorchIEKF
from src.evaluation.metrics import (
    compute_rpe,
    compute_ate,
    compute_orientation_error,
)
from src.evaluation.visualization import (
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_error_timeline,
    plot_covariance_timeline,
)


def test_sequence(iekf, dataset, dataset_name, cfg):
    """
    Run filter on a single sequence and compute metrics.

    Args:
        iekf: TorchIEKF model (in eval mode).
        dataset: Dataset instance.
        dataset_name: Name of the sequence to test.
        cfg: Full config.

    Returns:
        Dict with metrics, predictions, and ground truth.
    """
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # Normalize IMU for neural networks
    u_normalized = dataset.normalize(u)

    # Run neural networks to get covariances
    iekf.eval()
    with torch.no_grad():
        measurements_covs = iekf.forward_nets(u_normalized)

    # Run filter (torch, no grad) for fast inference
    N = len(t)
    with torch.no_grad():
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
            t, u, measurements_covs, v_gt, p_gt, N, ang_gt[0]
        )

    # Convert outputs to numpy
    Rot = Rot.cpu().numpy()
    v = v.cpu().numpy()
    p = p.cpu().numpy()
    b_omega = b_omega.cpu().numpy()
    b_acc = b_acc.cpu().numpy()

    # Convert ground truth to numpy
    p_gt_np = (p_gt - p_gt[0]).numpy()
    ang_gt_np = ang_gt.numpy()
    t_np = t.numpy()
    measurements_covs_np = measurements_covs.cpu().numpy()

    # Build ground truth rotation matrices
    N = Rot.shape[0]
    Rot_gt = np.zeros_like(Rot)
    for i in range(N):
        from src.utils.geometry import from_rpy

        Rot_gt[i] = from_rpy(ang_gt_np[i, 0], ang_gt_np[i, 1], ang_gt_np[i, 2])

    # Compute metrics
    rpe = compute_rpe(Rot, p, Rot_gt, p_gt_np)
    ate = compute_ate(p, p_gt_np, align=True)
    orient_err = compute_orientation_error(Rot, Rot_gt)

    metrics = {
        "rpe": rpe,
        "ate": ate,
        "orientation_error": orient_err,
    }

    results = {
        "metrics": metrics,
        "Rot": Rot,
        "v": v,
        "p": p,
        "b_omega": b_omega,
        "b_acc": b_acc,
        "p_gt": p_gt_np,
        "Rot_gt": Rot_gt,
        "t": t_np,
        "measurements_covs": measurements_covs_np,
        "name": dataset_name,
    }

    return results


def print_metrics(results, dataset_name):
    """Print metrics for a sequence."""
    metrics = results["metrics"]
    rpe = metrics["rpe"]
    ate = metrics["ate"]
    orient = metrics["orientation_error"]

    print(f"\n{'='*60}")
    print(f"Results for: {dataset_name}")
    print(f"{'='*60}")
    print(
        f"  RPE:  mean={rpe['mean']:.3f}%  std={rpe['std']:.3f}%  rmse={rpe['rmse']:.3f}%"
    )
    print(
        f"  ATE:  mean={ate['mean']:.2f}m  rmse={ate['rmse']:.2f}m  max={ate['max']:.2f}m"
    )
    print(
        f"  Orient: mean={orient['mean_deg']:.2f}°  max={orient['max_deg']:.2f}°"
    )


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
        iekf.load_state_dict(checkpoint["model_state_dict"])
    else:
        iekf.load_state_dict(checkpoint)
    iekf.eval()

    # Build dataset
    from src.data.kitti_dataset import KITTIDataset

    paths_cfg = cfg.get("paths", {})
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    dataset_cfg["path_data_save"] = paths_cfg.get("data", "../data")
    dataset_cfg["path_results"] = paths_cfg.get("results", "../results")
    dataset_cfg["path_temp"] = paths_cfg.get("temp", "../temp")
    dataset = KITTIDataset(dataset_cfg, split="test")

    # Optional WandB
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    if use_wandb:
        import wandb

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

            fig_2d = plot_trajectory_2d(p, p_gt, seq_name=dataset_name)
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
                import wandb

                metrics = results["metrics"]
                wandb.log(
                    {
                        f"test/{dataset_name}/rpe_mean": metrics["rpe"][
                            "mean"
                        ],
                        f"test/{dataset_name}/ate_rmse": metrics["ate"][
                            "rmse"
                        ],
                        f"test/{dataset_name}/orient_mean_deg": metrics[
                            "orientation_error"
                        ]["mean_deg"],
                        f"test/{dataset_name}/trajectory_2d": wandb.Image(
                            fig_2d
                        ),
                        f"test/{dataset_name}/error_timeline": wandb.Image(
                            fig_err
                        ),
                    }
                )

            import matplotlib.pyplot as plt

            plt.close("all")

        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    if all_results:
        rpe_means = [
            r["metrics"]["rpe"]["mean"]
            for r in all_results.values()
            if not np.isnan(r["metrics"]["rpe"]["mean"])
        ]
        ate_rmses = [r["metrics"]["ate"]["rmse"] for r in all_results.values()]

        print(f"\n{'='*60}")
        print("Overall Summary")
        print(f"{'='*60}")
        if rpe_means:
            print(f"  Mean RPE: {np.mean(rpe_means):.3f}%")
        if ate_rmses:
            print(f"  Mean ATE RMSE: {np.mean(ate_rmses):.2f}m")

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
