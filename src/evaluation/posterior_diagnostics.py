"""
Z posterior diagnostic computations (Evaluation 3).

All functions accept numpy arrays and return numpy arrays or scalars.
They operate on world_model_output fields saved by evaluate_sequence().
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


# ------------------------------------------------------------------ #
# 3a: Posterior width / collapse check
# ------------------------------------------------------------------ #


def compute_posterior_width(log_var_z):
    """
    Compute mean sigma_z per timestep.

    sigma_z = exp(0.5 * log_var_z)   shape (N, latent_dim)
    posterior_width = mean over latent_dim  shape (N,)

    Args:
        log_var_z: (N, latent_dim) log-variance array.

    Returns:
        (N,) posterior width per timestep.
    """
    sigma_z = np.exp(0.5 * np.asarray(log_var_z))
    return sigma_z.mean(axis=1)


def check_collapse(posterior_width):
    """
    Check if posterior width is in the healthy range [0.1, 0.9].

    Args:
        posterior_width: (N,) array from compute_posterior_width().

    Returns:
        dict with stats and boolean flags.
    """
    pw = np.asarray(posterior_width)
    mean_pw = float(np.mean(pw))
    return {
        "min": float(np.min(pw)),
        "max": float(np.max(pw)),
        "mean": mean_pw,
        "std": float(np.std(pw)),
        "collapsed": mean_pw > 0.95,       # encoder outputting prior — KL too large
        "exploded": mean_pw < 0.05,        # encoder ignoring KL — reconstruction dominates
        "healthy": 0.1 <= mean_pw <= 0.9,
    }


# ------------------------------------------------------------------ #
# 3b: Context sensitivity — correlation with yaw rate
# ------------------------------------------------------------------ #


def posterior_omega_correlation(log_var_z, u):
    """
    Pearson correlation between posterior width and |omega_z| (yaw rate).

    Args:
        log_var_z: (N, latent_dim) log-variance from world model output.
        u:         (N, 6) raw IMU [gyro_xyz, acc_xyz]. u[:, 2] = gyro_z.

    Returns:
        dict with "pearson_r" and "p_value".
    """
    pw = compute_posterior_width(log_var_z)
    omega_z = np.abs(np.asarray(u)[:, 2])

    # Align lengths in case of minor mismatch
    n = min(len(pw), len(omega_z))
    r, p = pearsonr(pw[:n], omega_z[:n])
    return {"pearson_r": float(r), "p_value": float(p)}


# ------------------------------------------------------------------ #
# 3c: PCA of mu_z
# ------------------------------------------------------------------ #


def pca_mu_z(mu_z, n_components=2):
    """
    PCA projection of mu_z to 2D.

    Args:
        mu_z: (N, latent_dim) posterior mean array.
        n_components: number of PCA components (default 2 for scatter plot).

    Returns:
        z_2d: (N, n_components) projected array.
        explained_variance_ratio: (n_components,) fraction of variance explained.
    """
    mu_z = np.asarray(mu_z)
    pca = PCA(n_components=n_components)
    z_2d = pca.fit_transform(mu_z)
    return z_2d, pca.explained_variance_ratio_


# ------------------------------------------------------------------ #
# 3d: N_n response to context
# ------------------------------------------------------------------ #


def extract_nn_timeseries(world_model_output):
    """
    Extract N_n diagonal values (lat, up) per timestep.

    Args:
        world_model_output: dict from evaluate_sequence()["world_model_output"].

    Returns:
        n_lat: (N,) lateral measurement noise standard deviation (sqrt of cov).
        n_up:  (N,) vertical measurement noise standard deviation.
        Or (None, None) if measurement_covs not available.
    """
    meas_covs = world_model_output.get("measurement_covs")
    if meas_covs is None:
        return None, None
    meas_covs = np.asarray(meas_covs)
    # meas_covs is (N, 2): [cov_lat, cov_up]
    return np.sqrt(meas_covs[:, 0]), np.sqrt(meas_covs[:, 1])


# ------------------------------------------------------------------ #
# 3e: Process head response to context
# ------------------------------------------------------------------ #


def compute_bias_correction_magnitude(world_model_output):
    """
    Compute total bias correction magnitude per timestep.

    correction_magnitude = ||delta_b_omega|| + ||delta_b_a||

    Args:
        world_model_output: dict from evaluate_sequence()["world_model_output"].

    Returns:
        (N,) total correction magnitude, or None if corrections not available.
    """
    gyro = world_model_output.get("gyro_bias_corrections")
    acc = world_model_output.get("acc_bias_corrections")
    if gyro is None or acc is None:
        return None
    gyro = np.asarray(gyro)
    acc = np.asarray(acc)
    return np.linalg.norm(gyro, axis=1) + np.linalg.norm(acc, axis=1)


# ------------------------------------------------------------------ #
# Convenience: run all diagnostics on one sequence result
# ------------------------------------------------------------------ #


def run_all_diagnostics(result):
    """
    Run all posterior diagnostics on a single evaluate_sequence() result dict.

    Args:
        result: dict returned by evaluate_sequence().

    Returns:
        dict with all diagnostic outputs, keyed by diagnostic name.
    """
    wm = result.get("world_model_output")
    if wm is None:
        raise ValueError("world_model_output not found in result. "
                         "Run with a world-model checkpoint (Condition D).")

    u = result["u"]
    t = result["t"]
    diag = {}

    # 3a
    pw = compute_posterior_width(wm["log_var_z"])
    diag["posterior_width"] = pw
    diag["collapse_check"] = check_collapse(pw)

    # 3b
    diag["omega_correlation"] = posterior_omega_correlation(wm["log_var_z"], u)

    # 3c
    z_2d, evr = pca_mu_z(wm["mu_z"])
    diag["z_pca_2d"] = z_2d
    diag["pca_explained_variance"] = evr

    # 3d
    n_lat, n_up = extract_nn_timeseries(wm)
    diag["n_lat"] = n_lat
    diag["n_up"] = n_up

    # 3e
    corr_mag = compute_bias_correction_magnitude(wm)
    diag["bias_correction_magnitude"] = corr_mag

    # Yaw rate for overlay plots
    diag["omega_z_abs"] = np.abs(np.asarray(u)[:, 2])
    diag["t"] = t

    return diag
