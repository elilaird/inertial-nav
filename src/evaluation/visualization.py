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


def plot_orientation_and_biases(
    Rot_pred, Rot_gt, b_omega, b_acc, timestamps=None, seq_name=""
):
    """
    Plot orientation (roll/pitch/yaw) and IMU biases over time.

    Corresponds to legacy fig2: orientation, gyro bias, accelerometer bias.

    Args:
        Rot_pred: Predicted rotation matrices (N, 3, 3) numpy array.
        Rot_gt: Ground-truth rotation matrices (N, 3, 3) numpy array.
        b_omega: Gyroscope bias estimates (N, 3) numpy array.
        b_acc: Accelerometer bias estimates (N, 3) numpy array.
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    from src.utils.geometry import to_rpy

    N = Rot_pred.shape[0]
    ang_pred = np.zeros((N, 3))
    ang_gt = np.zeros((N, 3))
    for i in range(N):
        ang_pred[i] = to_rpy(Rot_pred[i])
        ang_gt[i] = to_rpy(Rot_gt[i])

    if timestamps is None:
        timestamps = np.arange(N)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 12))

    # Orientation
    labels = ["roll", "pitch", "yaw"]
    colors_gt = ["b", "g", "r"]
    colors_pred = ["c", "lime", "orange"]
    for i in range(3):
        axs[0].plot(
            timestamps,
            np.rad2deg(ang_gt[:, i]),
            colors_gt[i],
            label=f"{labels[i]} GT",
        )
        axs[0].plot(
            timestamps,
            np.rad2deg(ang_pred[:, i]),
            colors_pred[i],
            linestyle="--",
            label=f"{labels[i]} pred",
        )
    axs[0].set_ylabel("Angle (deg)")
    axs[0].set_title(
        f"Orientation - {seq_name}" if seq_name else "Orientation"
    )
    axs[0].legend(ncol=3)
    axs[0].grid(True)

    # Gyro bias
    for i, c in enumerate(["r", "g", "b"]):
        axs[1].plot(timestamps, b_omega[:, i], c, label=f"axis {i}")
    axs[1].set_ylabel(r"$b^{\omega}$ (rad/s)")
    axs[1].set_title("Gyroscope Bias")
    axs[1].legend()
    axs[1].grid(True)

    # Accelerometer bias
    for i, c in enumerate(["r", "g", "b"]):
        axs[2].plot(timestamps, b_acc[:, i], c, label=f"axis {i}")
    axs[2].set_ylabel(r"$b^{a}$ (m/s²)")
    axs[2].set_title("Accelerometer Bias")
    axs[2].legend()
    axs[2].grid(True)

    axs[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_imu_raw(u, timestamps=None, seq_name=""):
    """
    Plot raw gyroscope and accelerometer inputs.

    Corresponds to legacy fig6.

    Args:
        u: Raw IMU measurements (N, 6) numpy array [gyro_xyz, acc_xyz].
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    if timestamps is None:
        timestamps = np.arange(u.shape[0])

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    # Gyroscope
    labels_gyro = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    for i in range(3):
        axs[0].plot(timestamps, u[:, i], alpha=0.8, label=labels_gyro[i])
    axs[0].set_ylabel("Angular rate (rad/s)")
    axs[0].set_title(f"Gyroscope - {seq_name}" if seq_name else "Gyroscope")
    axs[0].legend()
    axs[0].grid(True)

    # Accelerometer
    labels_acc = [r"$a_x$", r"$a_y$", r"$a_z$"]
    for i in range(3):
        axs[1].plot(timestamps, u[:, 3 + i], alpha=0.8, label=labels_acc[i])
    axs[1].set_ylabel(r"Acceleration (m/s²)")
    axs[1].set_title(
        f"Accelerometer - {seq_name}" if seq_name else "Accelerometer"
    )
    axs[1].legend()
    axs[1].grid(True)

    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_detailed_errors(p_pred, p_gt, timestamps=None, seq_name=""):
    """
    Plot MATE, CATE, and RMSE error breakdowns over time.

    Corresponds to legacy fig7: per-axis error decomposition.

    Args:
        p_pred: Predicted positions (N, 3) numpy array.
        p_gt: Ground-truth positions (N, 3) numpy array.
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    if timestamps is None:
        timestamps = np.arange(p_pred.shape[0])

    error_p = np.abs(p_gt - p_pred)

    # MATE: Mean Absolute Trajectory Error (xy vs z)
    mate_xy = np.mean(error_p[:, :2], axis=1)
    mate_z = error_p[:, 2]

    # CATE: Cumulative Absolute Trajectory Error
    cate_xy = np.cumsum(mate_xy) / np.arange(1, len(mate_xy) + 1)
    cate_z = np.cumsum(mate_z) / np.arange(1, len(mate_z) + 1)

    # RMSE (per-sample)
    rmse_xy = np.sqrt(0.5 * (error_p[:, 0] ** 2 + error_p[:, 1] ** 2))
    rmse_z = error_p[:, 2]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 12))

    # MATE + RMSE
    axs[0].plot(timestamps, mate_xy, label="MATE xy")
    axs[0].plot(timestamps, mate_z, label="MATE z")
    axs[0].plot(timestamps, rmse_xy, linestyle="--", label="RMSE xy")
    axs[0].plot(timestamps, rmse_z, linestyle="--", label="RMSE z")
    axs[0].set_ylabel("Error (m)")
    axs[0].set_title(
        f"MATE & RMSE - {seq_name}" if seq_name else "MATE & RMSE"
    )
    axs[0].legend()
    axs[0].grid(True)

    # CATE
    axs[1].plot(timestamps, cate_xy, label="CATE xy")
    axs[1].plot(timestamps, cate_z, label="CATE z")
    axs[1].set_ylabel("Cumulative Avg Error (m)")
    axs[1].set_title("Cumulative Absolute Trajectory Error (CATE)")
    axs[1].legend()
    axs[1].grid(True)

    # Per-axis SE(3) position error
    se3_error = p_gt - p_pred
    axs[2].plot(timestamps, se3_error[:, 0], "r-", alpha=0.8, label="x")
    axs[2].plot(timestamps, se3_error[:, 1], "g-", alpha=0.8, label="y")
    axs[2].plot(timestamps, se3_error[:, 2], "b-", alpha=0.8, label="z")
    axs[2].set_ylabel("Position Error (m)")
    axs[2].set_title("Per-Axis Position Error")
    axs[2].legend()
    axs[2].grid(True)

    axs[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_body_frame_velocity(
    v_pred, v_gt, Rot_pred, Rot_gt, timestamps=None, seq_name=""
):
    """
    Plot velocity in body frame (R^T v) for predicted and ground truth.

    Corresponds to legacy fig1 subplot 3.

    Args:
        v_pred: Predicted velocities (N, 3) numpy array (world frame).
        v_gt: Ground-truth velocities (N, 3) numpy array (world frame).
        Rot_pred: Predicted rotation matrices (N, 3, 3).
        Rot_gt: Ground-truth rotation matrices (N, 3, 3).
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    if timestamps is None:
        timestamps = np.arange(v_pred.shape[0])

    # Rotate to body frame: v_body = R^T @ v_world
    v_body_pred = np.einsum("nij,nj->ni", Rot_pred.transpose(0, 2, 1), v_pred)
    v_body_gt = np.einsum("nij,nj->ni", Rot_gt.transpose(0, 2, 1), v_gt)

    fig, ax = plt.subplots(figsize=(14, 5))
    labels = ["x", "y (lateral)", "z (vertical)"]
    colors_gt = ["b", "g", "r"]
    colors_pred = ["c", "lime", "orange"]

    for i in range(3):
        ax.plot(
            timestamps,
            v_body_gt[:, i],
            colors_gt[i],
            label=f"{labels[i]} GT",
        )
        ax.plot(
            timestamps,
            v_body_pred[:, i],
            colors_pred[i],
            linestyle="--",
            label=f"{labels[i]} pred",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    title = (
        f"Body-Frame Velocity - {seq_name}"
        if seq_name
        else "Body-Frame Velocity"
    )
    ax.set_title(title)
    ax.legend(ncol=3)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_covariance_with_imu(
    measurements_covs, u_normalized, timestamps=None, seq_name=""
):
    """
    Plot measurement covariances alongside normalized IMU inputs.

    Corresponds to legacy fig5 (all three subplots).

    Args:
        measurements_covs: Covariance values (N, D) numpy array.
        u_normalized: Normalized IMU measurements (N, 6) numpy array.
        timestamps: Time values (N,). If None, uses sample indices.
        seq_name: Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    if timestamps is None:
        timestamps = np.arange(measurements_covs.shape[0])

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 12))

    # Measurement covariances (log scale)
    log_covs = np.log10(np.clip(measurements_covs, 1e-10, None))
    cov_labels = ["Lateral velocity", "Vertical velocity"]
    for i in range(min(log_covs.shape[1], len(cov_labels))):
        axs[0].plot(timestamps, log_covs[:, i], alpha=0.8, label=cov_labels[i])
    for i in range(len(cov_labels), log_covs.shape[1]):
        axs[0].plot(timestamps, log_covs[:, i], alpha=0.8, label=f"Cov {i}")
    axs[0].set_ylabel("log10(covariance)")
    axs[0].set_title(
        f"Measurement Covariances - {seq_name}"
        if seq_name
        else "Measurement Covariances"
    )
    axs[0].legend()
    axs[0].grid(True)

    # Normalized gyro
    gyro_labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    for i in range(3):
        axs[1].plot(
            timestamps, u_normalized[:, i], alpha=0.8, label=gyro_labels[i]
        )
    axs[1].set_ylabel("Normalized value")
    axs[1].set_title("Normalized Gyro Measurements")
    axs[1].legend()
    axs[1].grid(True)

    # Normalized accelerometer
    acc_labels = [r"$a_x$", r"$a_y$", r"$a_z$"]
    for i in range(3):
        axs[2].plot(
            timestamps, u_normalized[:, 3 + i], alpha=0.8, label=acc_labels[i]
        )
    axs[2].set_ylabel("Normalized value")
    axs[2].set_title("Normalized Accelerometer Measurements")
    axs[2].legend()
    axs[2].grid(True)

    axs[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_world_model_uncertainty(
    uncertainty,
    p_pred,
    p_gt,
    timestamps=None,
    seq_name="",
):
    """
    Plot world model epistemic and aleatoric uncertainty over the trajectory.

    Top subplot: position error alongside aleatoric uncertainty (encoder
    sigma_z, the model's self-reported noise level).

    Middle subplot: epistemic uncertainty of measurement covariance
    predictions (std across MC z samples), showing where the model is
    unsure about its own predictions.

    Bottom subplot: epistemic uncertainty of bias correction predictions
    (acc + gyro std across MC z samples), if the process decoder is active.

    Args:
        uncertainty: Dict from ``_compute_world_model_uncertainty``.
        p_pred:      Predicted positions (N, 3) numpy array.
        p_gt:        Ground truth positions (N, 3) numpy array.
        timestamps:  Time values (N,). If None, uses sample indices.
        seq_name:    Sequence name for the title.

    Returns:
        matplotlib.Figure
    """
    has_process = (
        uncertainty.get("epistemic_acc_std") is not None
        or uncertainty.get("epistemic_gyro_std") is not None
    )
    n_rows = 3 if has_process else 2

    if timestamps is None:
        timestamps = np.arange(len(p_pred))

    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(14, 4 * n_rows))

    # ---- (1) Position error + aleatoric sigma_z ----
    pos_err = np.linalg.norm(p_gt - p_pred, axis=1)
    ax1 = axs[0]
    color_err = "tab:red"
    color_aleatoric = "tab:blue"

    ax1.plot(
        timestamps, pos_err, color=color_err, alpha=0.8, label="Position error"
    )
    ax1.set_ylabel("Position error (m)", color=color_err)
    ax1.tick_params(axis="y", labelcolor=color_err)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        timestamps,
        uncertainty["aleatoric_sigma_z"],
        color=color_aleatoric,
        alpha=0.7,
        label=r"Aleatoric $\bar{\sigma}_z$",
    )
    ax1_twin.set_ylabel(r"Aleatoric $\bar{\sigma}_z$", color=color_aleatoric)
    ax1_twin.tick_params(axis="y", labelcolor=color_aleatoric)

    title = (
        f"World Model Uncertainty — {seq_name}"
        if seq_name
        else "World Model Uncertainty"
    )
    ax1.set_title(title)
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- (2) Epistemic uncertainty: measurement covariances ----
    ax2 = axs[1]
    if uncertainty.get("epistemic_meas_std") is not None:
        ax2.plot(
            timestamps,
            uncertainty["epistemic_meas_std"],
            color="tab:purple",
            alpha=0.8,
            label="Measurement cov std (MC)",
        )
    ax2.set_ylabel("Epistemic std")
    ax2.set_title("Epistemic Uncertainty — Measurement Covariance Decoder")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Overlay position error (light) for reference
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        timestamps, pos_err, color=color_err, alpha=0.2, linewidth=0.8
    )
    ax2_twin.set_ylabel("Position error (m)", color=color_err, alpha=0.4)
    ax2_twin.tick_params(axis="y", labelcolor=color_err)

    # ---- (3) Epistemic uncertainty: bias corrections ----
    if has_process:
        ax3 = axs[2]
        if uncertainty.get("epistemic_acc_std") is not None:
            ax3.plot(
                timestamps,
                uncertainty["epistemic_acc_std"],
                color="tab:orange",
                alpha=0.8,
                label="Acc bias corr std (MC)",
            )
        if uncertainty.get("epistemic_gyro_std") is not None:
            ax3.plot(
                timestamps,
                uncertainty["epistemic_gyro_std"],
                color="tab:green",
                alpha=0.8,
                label="Gyro bias corr std (MC)",
            )
        ax3.set_ylabel("Epistemic std")
        ax3.set_title("Epistemic Uncertainty — Bias Correction Decoders")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(
            timestamps, pos_err, color=color_err, alpha=0.2, linewidth=0.8
        )
        ax3_twin.set_ylabel("Position error (m)", color=color_err, alpha=0.4)
        ax3_twin.tick_params(axis="y", labelcolor=color_err)

    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
