"""
Visualization utilities for trajectory estimation results.

Provides functions to plot trajectories, errors, and covariances,
returning matplotlib Figures for saving or WandB logging.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_trajectory_2d(p_pred, p_gt, seq_name="", aligned=False, p_imu=None):
    """
    Plot predicted vs ground truth trajectory in the XY plane.

    Args:
        p_pred: Predicted positions (N, 3) numpy array.
        p_gt: Ground truth positions (N, 3) numpy array.
        seq_name: Sequence name for the title.
        aligned: Whether positions have been pre-aligned.
        p_imu: Optional IMU dead-reckoning positions (N, 3) for baseline.

    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(p_gt[:, 0], p_gt[:, 1], "b-", label="Ground Truth", linewidth=1.5)
    ax.plot(
        p_pred[:, 0], p_pred[:, 1], "r-", label="IEKF (ours)", linewidth=1.5
    )
    if p_imu is not None:
        # Clip IMU trajectory to a padded bounding box around GT + IEKF so it
        # doesn't blow up the axis limits when integration diverges.
        all_xy = np.concatenate([p_gt[:, :2], p_pred[:, :2]], axis=0)
        xy_min = all_xy.min(axis=0)
        xy_max = all_xy.max(axis=0)
        span = (xy_max - xy_min).max()
        pad = max(span * 0.3, 50.0)  # 30 % margin, at least 50 m
        x_lo, y_lo = xy_min[0] - pad, xy_min[1] - pad
        x_hi, y_hi = xy_max[0] + pad, xy_max[1] + pad

        p_imu_clip = p_imu.copy()
        p_imu_clip[:, 0] = np.clip(p_imu_clip[:, 0], x_lo, x_hi)
        p_imu_clip[:, 1] = np.clip(p_imu_clip[:, 1], y_lo, y_hi)

        ax.plot(
            p_imu_clip[:, 0],
            p_imu_clip[:, 1],
            "g--",
            label="IMU integration (clipped)",
            linewidth=1.0,
            alpha=0.7,
        )
    ax.plot(p_gt[0, 0], p_gt[0, 1], "ko", markersize=8, label="Start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = f"Trajectory - {seq_name}" if seq_name else "Trajectory"
    if aligned:
        title += " (aligned)"
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_trajectory_3d(p_pred, p_gt, seq_name=""):
    """
    Plot predicted vs ground truth trajectory in 3D.

    Args:
        p_pred: Predicted positions (N, 3) numpy array.
        p_gt: Ground truth positions (N, 3) numpy array.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        p_gt[:, 0],
        p_gt[:, 1],
        p_gt[:, 2],
        "b-",
        label="Ground Truth",
        linewidth=1.5,
    )
    ax.plot(
        p_pred[:, 0],
        p_pred[:, 1],
        p_pred[:, 2],
        "r-",
        label="Predicted",
        linewidth=1.5,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    title = f"3D Trajectory - {seq_name}" if seq_name else "3D Trajectory"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_timeline(
    errors, timestamps=None, seq_name="", ylabel="Position Error (m)"
):
    """
    Plot position error over time.

    Args:
        errors: Error values (N,) or (N, 3) numpy array.
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.
        ylabel: Y-axis label.

    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    if timestamps is None:
        timestamps = np.arange(len(errors))

    if errors.ndim == 2 and errors.shape[1] == 3:
        ax.plot(timestamps, errors[:, 0], "r-", alpha=0.8, label="x")
        ax.plot(timestamps, errors[:, 1], "g-", alpha=0.8, label="y")
        ax.plot(timestamps, errors[:, 2], "b-", alpha=0.8, label="z")
        ax.legend()
    else:
        ax.plot(timestamps, errors, "r-", alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    title = f"Error Timeline - {seq_name}" if seq_name else "Error Timeline"
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_covariance_timeline(measurements_covs, timestamps=None, seq_name=""):
    """
    Plot learned measurement covariances over time (log scale).

    Args:
        measurements_covs: Covariance values (N, D) numpy array.
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    if timestamps is None:
        timestamps = np.arange(measurements_covs.shape[0])

    log_covs = np.log10(np.clip(measurements_covs, 1e-10, None))

    labels = ["Lateral velocity", "Vertical velocity"]
    for i in range(min(log_covs.shape[1], len(labels))):
        ax.plot(timestamps, log_covs[:, i], alpha=0.8, label=labels[i])
    for i in range(len(labels), log_covs.shape[1]):
        ax.plot(timestamps, log_covs[:, i], alpha=0.8, label=f"Cov {i}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("log10(covariance)")
    title = (
        f"Measurement Covariances - {seq_name}"
        if seq_name
        else "Measurement Covariances"
    )
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_state_estimates(
    p_pred, p_gt, v_pred=None, v_gt=None, timestamps=None, seq_name=""
):
    """
    Plot position and velocity state estimates vs ground truth.

    Args:
        p_pred: Predicted positions (N, 3).
        p_gt: Ground truth positions (N, 3).
        v_pred: Predicted velocities (N, 3), optional.
        v_gt: Ground truth velocities (N, 3), optional.
        timestamps: Time values (N,).
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    n_rows = 2 if v_pred is not None else 1
    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axs = [axs]

    if timestamps is None:
        timestamps = np.arange(p_pred.shape[0])

    labels = ["x", "y", "z"]
    colors_gt = ["b", "g", "r"]
    colors_pred = ["c", "lime", "orange"]

    for i in range(3):
        axs[0].plot(
            timestamps, p_gt[:, i], colors_gt[i], label=f"{labels[i]} GT"
        )
        axs[0].plot(
            timestamps,
            p_pred[:, i],
            colors_pred[i],
            linestyle="--",
            label=f"{labels[i]} pred",
        )
    axs[0].set_ylabel("Position (m)")
    axs[0].set_title(f"Position - {seq_name}" if seq_name else "Position")
    axs[0].legend(ncol=3)
    axs[0].grid(True)

    if v_pred is not None and v_gt is not None:
        for i in range(3):
            axs[1].plot(
                timestamps, v_gt[:, i], colors_gt[i], label=f"{labels[i]} GT"
            )
            axs[1].plot(
                timestamps,
                v_pred[:, i],
                colors_pred[i],
                linestyle="--",
                label=f"{labels[i]} pred",
            )
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].set_title("Velocity")
        axs[1].legend(ncol=3)
        axs[1].grid(True)

    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
