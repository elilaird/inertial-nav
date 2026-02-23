"""
Publication-quality diagnostic plots for Evaluation 3 (Z posterior diagnostics).

All functions return a matplotlib.Figure with no side effects (no plt.show()).
Save with fig.savefig(...).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ------------------------------------------------------------------ #
# Shared style
# ------------------------------------------------------------------ #

_STYLE = {
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ------------------------------------------------------------------ #
# 3a: Posterior width timeseries
# ------------------------------------------------------------------ #


def plot_posterior_width_timeseries(posterior_width, t, seq_name=""):
    """
    Plot mean(sigma_z) over time with healthy range [0.1, 0.9] shaded.

    Args:
        posterior_width: (N,) array from posterior_diagnostics.compute_posterior_width().
        t:               (N,) timestamps in seconds.
        seq_name:        Sequence label for the title.

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 3.5))

    t_rel = t - t[0]
    ax.plot(t_rel, posterior_width, color="#2271B5", linewidth=0.8, label=r"$\overline{\sigma}_z$")
    ax.axhspan(0.1, 0.9, color="green", alpha=0.08, label="Healthy range [0.1, 0.9]")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Prior (collapse)")
    ax.axhline(0.0, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="Collapsed posterior")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\overline{\sigma}_z = \mathrm{mean}(\sigma_z)$")
    ax.set_title(f"Posterior Width — {seq_name}")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", framealpha=0.85)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 3c: PCA latent space colored by |omega_z|
# ------------------------------------------------------------------ #


def plot_pca_latent_space(z_2d, omega_z, t=None, seq_name="", explained_variance=None):
    """
    Scatter plot of PCA-projected mu_z colored by |omega_z|.

    Args:
        z_2d:              (N, 2) PCA projections.
        omega_z:           (N,) |omega_z| values for coloring.
        t:                 (N,) timestamps (unused, kept for API symmetry).
        seq_name:          Sequence label for the title.
        explained_variance: (2,) PCA explained variance ratios (for axis labels).

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    norm = Normalize(vmin=np.percentile(omega_z, 5), vmax=np.percentile(omega_z, 95))
    sc = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=omega_z, cmap="plasma", norm=norm,
        s=2, alpha=0.5, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$|\omega_z|$ (rad/s)")

    pc1_label = f"PC1 ({100*explained_variance[0]:.1f}%)" if explained_variance is not None else "PC1"
    pc2_label = f"PC2 ({100*explained_variance[1]:.1f}%)" if explained_variance is not None else "PC2"
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_title(f"Latent Space PCA — {seq_name}")

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 3d: N_n diagonal values vs time + |omega_z| overlay
# ------------------------------------------------------------------ #


def plot_nn_vs_omega(n_lat, n_up, omega_z, t, seq_name=""):
    """
    Dual-axis plot: N_n lateral and vertical std-dev vs time, with |omega_z| overlay.

    Args:
        n_lat:    (N,) sqrt(N_n_lat) per timestep.
        n_up:     (N,) sqrt(N_n_up) per timestep.
        omega_z:  (N,) |omega_z| for right-axis overlay.
        t:        (N,) timestamps in seconds.
        seq_name: Sequence label for the title.

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    t_rel = t - t[0]
    ax1.plot(t_rel, n_lat, color="#2271B5", linewidth=0.9, label=r"$\sqrt{N_n^{lat}}$ (m/s)")
    ax1.plot(t_rel, n_up, color="#E69F00", linewidth=0.9, linestyle="--", label=r"$\sqrt{N_n^{up}}$ (m/s)")
    ax2.plot(t_rel, omega_z, color="#CC79A7", linewidth=0.6, alpha=0.5, label=r"$|\omega_z|$ (rad/s)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Measurement noise std (m/s)")
    ax2.set_ylabel(r"$|\omega_z|$ (rad/s)", color="#CC79A7")
    ax2.tick_params(axis="y", labelcolor="#CC79A7")
    ax1.set_title(f"Measurement Decoder Output vs Yaw Rate — {seq_name}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 3e: Bias correction magnitude vs time + |omega_z| overlay
# ------------------------------------------------------------------ #


def plot_bias_corrections_vs_omega(correction_mag, omega_z, t, seq_name=""):
    """
    Dual-axis plot: total bias correction magnitude vs time, with |omega_z| overlay.

    Args:
        correction_mag: (N,) ||delta_b_omega|| + ||delta_b_a|| per timestep.
        omega_z:        (N,) |omega_z| for right-axis overlay.
        t:              (N,) timestamps in seconds.
        seq_name:       Sequence label for the title.

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    t_rel = t - t[0]
    ax1.plot(t_rel, correction_mag, color="#009E73", linewidth=0.9,
             label=r"$\|\Delta b_\omega\| + \|\Delta b_a\|$")
    ax2.plot(t_rel, omega_z, color="#CC79A7", linewidth=0.6, alpha=0.5,
             label=r"$|\omega_z|$ (rad/s)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Bias correction magnitude (rad/s + m/s²)")
    ax2.set_ylabel(r"$|\omega_z|$ (rad/s)", color="#CC79A7")
    ax2.tick_params(axis="y", labelcolor="#CC79A7")
    ax1.set_title(f"Process Decoder Correction Magnitude vs Yaw Rate — {seq_name}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Summary panel: all 5 diagnostics in one figure
# ------------------------------------------------------------------ #


def plot_diagnostic_panel(diag_08, diag_01, seq_name_08="seq08", seq_name_01="seq01"):
    """
    5-subplot panel combining all posterior diagnostics.

    Args:
        diag_08: output of posterior_diagnostics.run_all_diagnostics() for seq 08.
        diag_01: output of posterior_diagnostics.run_all_diagnostics() for seq 01.
        seq_name_08: label for sequence 08.
        seq_name_01: label for sequence 01.

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # 3a: posterior width (seq 08)
    ax_a = fig.add_subplot(gs[0, :])
    t_rel = diag_08["t"] - diag_08["t"][0]
    pw = diag_08["posterior_width"]
    ax_a.plot(t_rel, pw, color="#2271B5", linewidth=0.8, label=r"$\overline{\sigma}_z$")
    ax_a.axhspan(0.1, 0.9, color="green", alpha=0.08, label="Healthy [0.1, 0.9]")
    ax_a.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel(r"$\overline{\sigma}_z$")
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.set_title(f"(3a) Posterior Width — {seq_name_08}")
    ax_a.legend(loc="upper right", framealpha=0.85)
    ax_a.grid(alpha=0.3)

    # 3c: PCA (seq 08)
    ax_c = fig.add_subplot(gs[1, 0])
    z2d = diag_08["z_pca_2d"]
    omega = diag_08["omega_z_abs"]
    norm = Normalize(vmin=np.percentile(omega, 5), vmax=np.percentile(omega, 95))
    sc = ax_c.scatter(z2d[:, 0], z2d[:, 1], c=omega, cmap="plasma",
                      norm=norm, s=2, alpha=0.5, rasterized=True)
    fig.colorbar(sc, ax=ax_c, label=r"$|\omega_z|$")
    evr = diag_08["pca_explained_variance"]
    ax_c.set_xlabel(f"PC1 ({100*evr[0]:.1f}%)")
    ax_c.set_ylabel(f"PC2 ({100*evr[1]:.1f}%)")
    ax_c.set_title(f"(3c) Latent Space PCA — {seq_name_08}")

    # 3e: bias corrections (seq 08)
    ax_e = fig.add_subplot(gs[1, 1])
    t_rel_08 = diag_08["t"] - diag_08["t"][0]
    corr = diag_08["bias_correction_magnitude"]
    if corr is not None:
        ax_e2 = ax_e.twinx()
        ax_e.plot(t_rel_08, corr, color="#009E73", linewidth=0.8)
        ax_e2.plot(t_rel_08, diag_08["omega_z_abs"], color="#CC79A7",
                   linewidth=0.5, alpha=0.5)
        ax_e.set_ylabel("Bias corr. magnitude")
        ax_e2.set_ylabel(r"$|\omega_z|$", color="#CC79A7")
        ax_e2.tick_params(axis="y", labelcolor="#CC79A7")
    else:
        ax_e.text(0.5, 0.5, "Process head disabled", ha="center", va="center",
                  transform=ax_e.transAxes)
    ax_e.set_xlabel("Time (s)")
    ax_e.set_title(f"(3e) Process Corrections — {seq_name_08}")

    # 3d: N_n vs omega (seq 01)
    ax_d = fig.add_subplot(gs[2, :])
    t_rel_01 = diag_01["t"] - diag_01["t"][0]
    n_lat = diag_01["n_lat"]
    n_up = diag_01["n_up"]
    if n_lat is not None:
        ax_d2 = ax_d.twinx()
        ax_d.plot(t_rel_01, n_lat, color="#2271B5", linewidth=0.9, label=r"$\sqrt{N_n^{lat}}$")
        ax_d.plot(t_rel_01, n_up, color="#E69F00", linewidth=0.9, linestyle="--",
                  label=r"$\sqrt{N_n^{up}}$")
        ax_d2.plot(t_rel_01, diag_01["omega_z_abs"], color="#CC79A7",
                   linewidth=0.6, alpha=0.5, label=r"$|\omega_z|$")
        ax_d.set_ylabel("Meas. noise std (m/s)")
        ax_d2.set_ylabel(r"$|\omega_z|$ (rad/s)", color="#CC79A7")
        ax_d2.tick_params(axis="y", labelcolor="#CC79A7")
        lines1, lbl1 = ax_d.get_legend_handles_labels()
        lines2, lbl2 = ax_d2.get_legend_handles_labels()
        ax_d.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", framealpha=0.85)
    else:
        ax_d.text(0.5, 0.5, "Measurement head disabled", ha="center", va="center",
                  transform=ax_d.transAxes)
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title(f"(3d) Measurement Decoder N_n vs Yaw Rate — {seq_name_01}")

    return fig
