"""
Shared evaluation logic for running the IEKF filter on a sequence and
computing metrics.  Used by both the standalone ``test.py`` entry point
and the ``TestEvalCallback`` during training.
"""

import numpy as np
import torch

from src.evaluation.metrics import (
    compute_rpe,
    compute_ate,
    compute_orientation_error,
)
from src.utils.geometry import from_rpy


# ------------------------------------------------------------------
# IMU direct integration baseline
# ------------------------------------------------------------------


def _so3exp_np(phi):
    """SO(3) exponential map (numpy, Rodrigues)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + _skew(phi)
    axis = phi / angle
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _skew(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def imu_dead_reckoning(t, u, ang0, v0, g=np.array([0.0, 0.0, -9.80655])):
    """
    Pure IMU forward integration (no filter corrections).

    Integrates gyroscope for orientation and accelerometer for velocity /
    position using zero initial biases and no Kalman update.

    Args:
        t:    Timestamps (N,) numpy array.
        u:    IMU measurements (N, 6) numpy array [gyro_x/y/z, acc_x/y/z].
        ang0: Initial orientation (3,) [roll, pitch, yaw] in radians.
        v0:   Initial velocity (3,) numpy array.
        g:    Gravity vector in the world frame.

    Returns:
        Rot_imu: Rotation matrices (N, 3, 3).
        p_imu:   Positions   (N, 3).
        v_imu:   Velocities  (N, 3).
    """
    N = len(t)
    Rot = np.zeros((N, 3, 3))
    p = np.zeros((N, 3))
    v = np.zeros((N, 3))

    Rot[0] = from_rpy(float(ang0[0]), float(ang0[1]), float(ang0[2]))
    v[0] = v0

    for i in range(1, N):
        dt = float(t[i] - t[i - 1])

        omega = u[i, :3] * dt
        Rot[i] = Rot[i - 1] @ _so3exp_np(omega)

        acc_world = Rot[i - 1] @ u[i, 3:6] + g
        v[i] = v[i - 1] + acc_world * dt
        p[i] = p[i - 1] + v[i - 1] * dt + 0.5 * acc_world * dt**2

    return Rot, p, v


def evaluate_sequence(iekf, dataset, dataset_name):
    """
    Run the filter on *dataset_name* and return predictions + metrics.

    The model is put into ``eval()`` mode and all computation is done
    under ``torch.no_grad()``.

    Args:
        iekf: ``TorchIEKF`` model (already on the correct device).
        dataset: Dataset instance that provides ``get_data`` / ``normalize``.
        dataset_name: Sequence identifier string.

    Returns:
        Dict with keys ``metrics``, ``p``, ``p_gt``, ``Rot``, ``Rot_gt``,
        ``v``, ``b_omega``, ``b_acc``, ``t``, ``measurements_covs``, ``name``.
    """
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # Normalize IMU for neural networks
    u_normalized = dataset.normalize(u)

    iekf.eval()
    with torch.no_grad():
        measurements_covs = iekf.forward_nets(u_normalized)
        bias_corrections = iekf.forward_bias_net(u_normalized)

        N = len(t)
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
            t,
            u,
            measurements_covs,
            v_gt,
            p_gt,
            N,
            ang_gt[0],
            bias_corrections=bias_corrections,
        )

    # ---- numpy conversion ----
    Rot = Rot.cpu().numpy()
    v = v.cpu().numpy()
    p = p.cpu().numpy()
    b_omega = b_omega.cpu().numpy()
    b_acc = b_acc.cpu().numpy()

    p_gt_np = (p_gt - p_gt[0]).numpy()
    ang_gt_np = ang_gt.numpy()
    t_np = t.numpy()
    measurements_covs_np = measurements_covs.cpu().numpy()

    # Build ground-truth rotation matrices
    N = Rot.shape[0]
    Rot_gt = np.zeros_like(Rot)
    for i in range(N):
        Rot_gt[i] = from_rpy(ang_gt_np[i, 0], ang_gt_np[i, 1], ang_gt_np[i, 2])

    # ---- metrics ----
    rpe = compute_rpe(Rot, p, Rot_gt, p_gt_np)
    ate = compute_ate(p, p_gt_np, align=True)
    orient_err = compute_orientation_error(Rot, Rot_gt)

    # ---- IMU direct integration baseline ----
    u_np = u.numpy() if isinstance(u, torch.Tensor) else u
    v0_np = v_gt[0].numpy() if isinstance(v_gt, torch.Tensor) else v_gt[0]
    ang0_np = ang_gt_np[0]
    Rot_imu, p_imu, v_imu = imu_dead_reckoning(t_np, u_np, ang0_np, v0_np)

    rpe_imu = compute_rpe(Rot_imu, p_imu, Rot_gt, p_gt_np)
    ate_imu = compute_ate(p_imu, p_gt_np, align=True)
    orient_imu = compute_orientation_error(Rot_imu, Rot_gt)

    return {
        "metrics": {
            "rpe": rpe,
            "ate": ate,
            "orientation_error": orient_err,
        },
        "metrics_imu": {
            "rpe": rpe_imu,
            "ate": ate_imu,
            "orientation_error": orient_imu,
        },
        "Rot": Rot,
        "v": v,
        "p": p,
        "b_omega": b_omega,
        "b_acc": b_acc,
        "p_gt": p_gt_np,
        "v_gt": v_gt.numpy() if isinstance(v_gt, torch.Tensor) else v_gt,
        "Rot_gt": Rot_gt,
        "p_imu": p_imu,
        "v_imu": v_imu,
        "Rot_imu": Rot_imu,
        "u": u_np,
        "u_normalized": (
            u_normalized.numpy()
            if isinstance(u_normalized, torch.Tensor)
            else u_normalized
        ),
        "t": t_np,
        "measurements_covs": measurements_covs_np,
        "name": dataset_name,
    }


def format_metrics(results, dataset_name):
    """Return a human-readable string summarising the evaluation metrics."""
    m = results["metrics"]
    rpe, ate, orient = m["rpe"], m["ate"], m["orientation_error"]

    lines = [
        f"{'='*60}",
        f"Results for: {dataset_name}",
        f"{'='*60}",
        f"  IEKF (ours):",
        f"    t_rel: {rpe['t_rel']:.3f}%   r_rel: {rpe['r_rel']:.4f} deg/m",
        f"    ATE:   mean={ate['mean']:.2f}m  rmse={ate['rmse']:.2f}m  max={ate['max']:.2f}m",
    ]

    if "metrics_imu" in results:
        mi = results["metrics_imu"]
        rpe_i, ate_i = mi["rpe"], mi["ate"]
        lines += [
            f"  IMU integration:",
            f"    t_rel: {rpe_i['t_rel']:.3f}%   r_rel: {rpe_i['r_rel']:.4f} deg/m",
            f"    ATE:   mean={ate_i['mean']:.2f}m  rmse={ate_i['rmse']:.2f}m  max={ate_i['max']:.2f}m",
        ]

    return "\n".join(lines)
