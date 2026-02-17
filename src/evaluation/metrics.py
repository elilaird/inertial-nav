"""
Evaluation metrics for trajectory estimation.

Provides RPE (Relative Pose Error), ATE (Absolute Trajectory Error),
and orientation error metrics matching the KITTI odometry benchmark.
"""

import numpy as np
import torch

from src.utils.geometry import umeyama_alignment, to_rpy


def compute_rpe(Rot_pred, p_pred, Rot_gt, p_gt,
                distance_windows=None, sampling_rate=10):
    """
    Compute Relative Pose Error at various distance windows.

    Follows the KITTI odometry benchmark protocol: downsample to 1Hz,
    compute relative displacements at fixed distances, and report
    translational error as percentage of distance traveled.

    Args:
        Rot_pred: Predicted rotations (N, 3, 3) numpy array.
        p_pred: Predicted positions (N, 3) numpy array.
        Rot_gt: Ground truth rotations (N, 3, 3) numpy array.
        p_gt: Ground truth positions (N, 3) numpy array.
        distance_windows: List of distances in meters (default: 100-800m).
        sampling_rate: Downsample rate from raw to evaluation (default: 10).

    Returns:
        Dict with keys: 'mean', 'std', 'rmse', and per-distance metrics.
    """
    if distance_windows is None:
        distance_windows = [100, 200, 300, 400, 500, 600, 700, 800]

    # Downsample
    Rot_pred_ds = Rot_pred[::sampling_rate]
    p_pred_ds = p_pred[::sampling_rate]
    Rot_gt_ds = Rot_gt[::sampling_rate]
    p_gt_ds = p_gt[::sampling_rate]

    # Cumulative distances along ground truth
    dp = np.diff(p_gt_ds, axis=0)
    distances = np.zeros(p_gt_ds.shape[0])
    distances[1:] = np.cumsum(np.linalg.norm(dp, axis=1))

    step_size = 10  # 1Hz sampling from 10Hz
    k_max = int(Rot_gt_ds.shape[0] / step_size) - 1

    errors_per_dist = {d: [] for d in distance_windows}
    all_errors = []

    for k in range(k_max):
        idx_0 = k * step_size
        for seq_length in distance_windows:
            if seq_length + distances[idx_0] > distances[-1]:
                continue
            idx_shift = np.searchsorted(distances[idx_0:],
                                        distances[idx_0] + seq_length)
            idx_end = idx_0 + idx_shift

            # Ground truth relative displacement in local frame
            dp_gt = Rot_gt_ds[idx_0].T @ (p_gt_ds[idx_end] - p_gt_ds[idx_0])
            dp_pred = Rot_pred_ds[idx_0].T @ (p_pred_ds[idx_end] - p_pred_ds[idx_0])

            err = np.linalg.norm(dp_pred - dp_gt) / seq_length * 100  # percentage
            errors_per_dist[seq_length].append(err)
            all_errors.append(err)

    if not all_errors:
        return {'mean': float('nan'), 'std': float('nan'), 'rmse': float('nan')}

    all_errors = np.array(all_errors)
    results = {
        'mean': float(np.mean(all_errors)),
        'std': float(np.std(all_errors)),
        'rmse': float(np.sqrt(np.mean(all_errors ** 2))),
    }

    for d in distance_windows:
        errs = errors_per_dist[d]
        if errs:
            results[f'rpe_{d}m'] = float(np.mean(errs))
        else:
            results[f'rpe_{d}m'] = float('nan')

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
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
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
        'mean_deg': float(np.mean(angles)),
        'std_deg': float(np.std(angles)),
        'rmse_deg': float(np.sqrt(np.mean(angles ** 2))),
        'max_deg': float(np.max(angles)),
    }
