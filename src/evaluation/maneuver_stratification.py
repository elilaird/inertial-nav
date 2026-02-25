"""
Maneuver-stratified RTE evaluation (Evaluation 2).

Segments timesteps by yaw-rate magnitude (|omega_z| = |u[:, 2]|) into
straight, corner, and transition bins, then computes position error
accumulated in each bin.
"""

import numpy as np


def segment_by_yaw_rate(u, percentiles=(30, 70)):
    """
    Segment timesteps by |omega_z| (gyro z-axis = yaw rate).

    Bins:
      straight   — bottom ``percentiles[0]``% of |omega_z| values
      corner     — top  (100 - ``percentiles[1]``)% of |omega_z| values
      transition — middle band (excluded from analysis per experiment spec)

    Args:
        u: (N, 6) raw IMU tensor or array [gyro_xyz, acc_xyz].
        percentiles: (low_pct, high_pct) defining the threshold cut-points.

    Returns:
        dict with boolean masks of shape (N,):
            "straight", "corner", "transition"
        dict with scalar thresholds:
            "omega_z_low", "omega_z_high"
    """
    u = np.asarray(u)
    omega_z = np.abs(u[:, 2])

    p_low = np.percentile(omega_z, percentiles[0])
    p_high = np.percentile(omega_z, percentiles[1])

    masks = {
        "straight": omega_z <= p_low,
        "corner": omega_z >= p_high,
        "transition": (omega_z > p_low) & (omega_z < p_high),
    }
    thresholds = {"omega_z_low": float(p_low), "omega_z_high": float(p_high)}
    return masks, thresholds


def compute_stratified_position_error(p_pred, p_gt, masks):
    """
    Compute per-timestep position error (Euclidean) stratified by segment type.

    Args:
        p_pred: (N, 3) predicted positions.
        p_gt:   (N, 3) ground-truth positions.
        masks:  dict of boolean arrays from segment_by_yaw_rate().

    Returns:
        dict: {segment_name: {"mean_err": float, "std_err": float,
                              "rmse": float, "n_samples": int}}
    """
    p_pred = np.asarray(p_pred)
    p_gt = np.asarray(p_gt)
    pos_err = np.linalg.norm(p_gt - p_pred, axis=1)  # (N,)

    results = {}
    for name, mask in masks.items():
        errs = pos_err[mask]
        if len(errs) == 0:
            results[name] = {
                "mean_err": float("nan"),
                "std_err": float("nan"),
                "rmse": float("nan"),
                "n_samples": 0,
            }
        else:
            results[name] = {
                "mean_err": float(np.mean(errs)),
                "std_err": float(np.std(errs)),
                "rmse": float(np.sqrt(np.mean(errs**2))),
                "n_samples": int(mask.sum()),
            }
    return results


def stratified_comparison(results_A, results_D, seq_names):
    """
    Compare stratified errors between Condition A and Condition D
    across multiple sequences.

    Args:
        results_A: dict of {drive_name: evaluate_sequence output} for Condition A.
        results_D: dict of {drive_name: evaluate_sequence output} for Condition D.
        seq_names: list of drive names to include.

    Returns:
        dict: per-sequence and per-segment stratified comparison.
    """
    comparison = {}
    for name in seq_names:
        if name not in results_A or name not in results_D:
            continue

        u = results_A[name]["u"]
        masks, thresholds = segment_by_yaw_rate(u)

        err_A = compute_stratified_position_error(
            results_A[name]["p"], results_A[name]["p_gt"], masks
        )
        err_D = compute_stratified_position_error(
            results_D[name]["p"], results_D[name]["p_gt"], masks
        )

        comparison[name] = {
            "thresholds": thresholds,
            "condition_A": err_A,
            "condition_D": err_D,
        }
        for seg in ["straight", "corner"]:
            mean_A = err_A[seg]["mean_err"]
            mean_D = err_D[seg]["mean_err"]
            if not (np.isnan(mean_A) or np.isnan(mean_D) or mean_A == 0):
                improvement_pct = 100.0 * (mean_A - mean_D) / mean_A
            else:
                improvement_pct = float("nan")
            comparison[name][f"improvement_{seg}_pct"] = improvement_pct

    return comparison


def print_stratified_table(comparison):
    """Print a human-readable stratified comparison table."""
    print(f"\n{'='*70}")
    print("Maneuver-Stratified Position Error Comparison (A vs D)")
    print(f"{'='*70}")
    header = f"{'Sequence':>35} | {'Straight':>10} | {'Corner':>10}"
    print(header)
    print(f"{'-'*35}-+-{'-'*10}-+-{'-'*10}")

    for name, data in comparison.items():
        err_A_s = data["condition_A"]["straight"]["mean_err"]
        err_D_s = data["condition_D"]["straight"]["mean_err"]
        err_A_c = data["condition_A"]["corner"]["mean_err"]
        err_D_c = data["condition_D"]["corner"]["mean_err"]

        imp_s = data.get("improvement_straight_pct", float("nan"))
        imp_c = data.get("improvement_corner_pct", float("nan"))

        short_name = name[-30:] if len(name) > 30 else name
        print(f"  {short_name:>33} | A:{err_A_s:5.2f}m D:{err_D_s:5.2f}m ({imp_s:+.1f}%) | A:{err_A_c:5.2f}m D:{err_D_c:5.2f}m ({imp_c:+.1f}%)")
