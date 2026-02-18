"""
Evaluation metrics for trajectory estimation.

Provides RPE (Relative Pose Error), ATE (Absolute Trajectory Error),
and orientation error metrics matching the KITTI odometry benchmark.
"""

import numpy as np
import torch

from src.utils.geometry import umeyama_alignment, to_rpy


def compute_rpe(
    Rot_pred, p_pred, Rot_gt, p_gt, distance_windows=None, sampling_rate=10
):
    """
    Compute the two KITTI-benchmark relative pose error metrics from [3]:

    * **t_rel** – averaged relative translation error for all subsequences of
      length 100 m … 800 m, expressed as **% of distance traveled**.
    * **r_rel** – averaged relative rotation error for the same subsequences,
      expressed in **deg / m**.

    Args:
        Rot_pred: Predicted rotations (N, 3, 3) numpy array.
        p_pred: Predicted positions (N, 3) numpy array.
        Rot_gt: Ground truth rotations (N, 3, 3) numpy array.
        p_gt: Ground truth positions (N, 3) numpy array.
        distance_windows: List of distances in metres (default: 100–800 m).
        sampling_rate: Downsample factor from raw rate to 10 Hz before
            computing (default: 10, i.e. 100 Hz → 10 Hz).

    Returns:
        Dict with keys:

        ``t_rel``         – mean translational error across all windows (%)
        ``t_rel_std``     – std of translational error (%)
        ``t_rel_rmse``    – RMSE of translational error (%)
        ``r_rel``         – mean rotational error across all windows (deg/m)
        ``r_rel_std``     – std of rotational error (deg/m)
        ``r_rel_rmse``    – RMSE of rotational error (deg/m)
        ``t_rel_<d>m``    – per-distance mean translational error (%)
        ``r_rel_<d>m``    – per-distance mean rotational error (deg/m)

        Legacy keys ``mean``, ``std``, ``rmse`` are aliases for the
        ``t_rel`` equivalents for backward compatibility.
    """
    if distance_windows is None:
        distance_windows = [100, 200, 300, 400, 500, 600, 700, 800]

    # Downsample to 10 Hz
    Rot_pred_ds = Rot_pred[::sampling_rate]
    p_pred_ds = p_pred[::sampling_rate]
    Rot_gt_ds = Rot_gt[::sampling_rate]
    p_gt_ds = p_gt[::sampling_rate]

    # Cumulative distances along ground truth
    dp = np.diff(p_gt_ds, axis=0)
    distances = np.zeros(p_gt_ds.shape[0])
    distances[1:] = np.cumsum(np.linalg.norm(dp, axis=1))

    step_size = 10  # stride over the 10-Hz array (1 Hz evaluation cadence)
    k_max = int(Rot_gt_ds.shape[0] / step_size) - 1

    t_errors_per_dist = {d: [] for d in distance_windows}
    r_errors_per_dist = {d: [] for d in distance_windows}
    all_t_errors = []
    all_r_errors = []

    for k in range(k_max):
        idx_0 = k * step_size
        for seq_length in distance_windows:
            if seq_length + distances[idx_0] > distances[-1]:
                continue
            idx_shift = np.searchsorted(
                distances[idx_0:], distances[idx_0] + seq_length
            )
            idx_end = idx_0 + idx_shift

            # --- translational error (t_rel) ---
            dp_gt = Rot_gt_ds[idx_0].T @ (p_gt_ds[idx_end] - p_gt_ds[idx_0])
            dp_pred = Rot_pred_ds[idx_0].T @ (
                p_pred_ds[idx_end] - p_pred_ds[idx_0]
            )
            t_err = np.linalg.norm(dp_pred - dp_gt) / seq_length * 100
            t_errors_per_dist[seq_length].append(t_err)
            all_t_errors.append(t_err)

            # --- rotational error (r_rel) ---
            # Relative rotation over the subsequence for GT and prediction
            dR_gt = Rot_gt_ds[idx_0].T @ Rot_gt_ds[idx_end]
            dR_pred = Rot_pred_ds[idx_0].T @ Rot_pred_ds[idx_end]
            # Error rotation
            dR_err = dR_gt.T @ dR_pred
            trace = np.clip(np.trace(dR_err), -1.0, 3.0)
            angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            # deg / m
            r_err = np.degrees(angle_rad) / seq_length
            r_errors_per_dist[seq_length].append(r_err)
            all_r_errors.append(r_err)

    if not all_t_errors:
        nan = float("nan")
        return {
            "t_rel": nan,
            "t_rel_std": nan,
            "t_rel_rmse": nan,
            "r_rel": nan,
            "r_rel_std": nan,
            "r_rel_rmse": nan,
            # legacy aliases
            "mean": nan,
            "std": nan,
            "rmse": nan,
        }

    all_t_errors = np.array(all_t_errors)
    all_r_errors = np.array(all_r_errors)

    results = {
        # Paper metrics
        "t_rel": float(np.mean(all_t_errors)),
        "t_rel_std": float(np.std(all_t_errors)),
        "t_rel_rmse": float(np.sqrt(np.mean(all_t_errors**2))),
        "r_rel": float(np.mean(all_r_errors)),
        "r_rel_std": float(np.std(all_r_errors)),
        "r_rel_rmse": float(np.sqrt(np.mean(all_r_errors**2))),
        # Backward-compatibility aliases
        "mean": float(np.mean(all_t_errors)),
        "std": float(np.std(all_t_errors)),
        "rmse": float(np.sqrt(np.mean(all_t_errors**2))),
    }

    for d in distance_windows:
        t_errs = t_errors_per_dist[d]
        r_errs = r_errors_per_dist[d]
        results[f"t_rel_{d}m"] = (
            float(np.mean(t_errs)) if t_errs else float("nan")
        )
        results[f"r_rel_{d}m"] = (
            float(np.mean(r_errs)) if r_errs else float("nan")
        )
        # legacy key
        results[f"rpe_{d}m"] = results[f"t_rel_{d}m"]

    return results


def compute_ate(p_pred, p_gt, align=True):
    """
    Compute Absolute Trajectory Error.

    Args:
        p_pred: Predicted positions (N, 3) numpy array.
        p_gt: Ground truth positions (N, 3) numpy array.
        align: Whether to apply Umeyama alignment first.

    Returns:
        Dict with keys: 'mean', 'std', 'rmse', 'median', 'max'.
    """
    if align:
        R_align, t_align, _ = umeyama_alignment(p_gt.T, p_pred.T)
        p_aligned = (R_align.T @ p_pred.T).T - R_align.T @ t_align
    else:
        p_aligned = p_pred

    errors = np.linalg.norm(p_gt - p_aligned, axis=1)

    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def compute_orientation_error(Rot_pred, Rot_gt):
    """
    Compute orientation error between predicted and ground truth rotations.

    Error is measured as the angle of the rotation difference R_err = R_gt^T @ R_pred.

    Args:
        Rot_pred: Predicted rotations (N, 3, 3) numpy array.
        Rot_gt: Ground truth rotations (N, 3, 3) numpy array.

    Returns:
        Dict with keys: 'mean_deg', 'std_deg', 'rmse_deg', 'max_deg'.
    """
    N = Rot_pred.shape[0]
    angles = np.zeros(N)

    for i in range(N):
        R_err = Rot_gt[i].T @ Rot_pred[i]
        # Rotation angle from trace: cos(theta) = (trace(R) - 1) / 2
        trace = np.clip(np.trace(R_err), -1.0, 3.0)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        angles[i] = np.degrees(angle)

    return {
        "mean_deg": float(np.mean(angles)),
        "std_deg": float(np.std(angles)),
        "rmse_deg": float(np.sqrt(np.mean(angles**2))),
        "max_deg": float(np.max(angles)),
    }
