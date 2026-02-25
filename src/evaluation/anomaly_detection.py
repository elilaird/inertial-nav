"""
Anomalous sequence evaluation utilities (Evaluation 4).

Tests whether the world model encoder produces wider posteriors at timestamps
where data gaps or artifacts occur in sequences 00, 02, 05.
"""

import numpy as np


def detect_data_gaps(t, threshold_s=0.02):
    """
    Find timestep indices where dt > threshold (data gaps at 100 Hz).

    At 100 Hz the nominal dt = 0.01s. Gaps larger than threshold_s indicate
    missing frames or logging artifacts.

    Args:
        t:           (N,) timestamps in seconds.
        threshold_s: dt threshold for gap detection (default 0.02 = 2× nominal).

    Returns:
        gap_indices: array of indices i where t[i+1] - t[i] > threshold_s.
        gap_sizes_s: corresponding gap durations in seconds.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    mask = dt > threshold_s
    gap_indices = np.where(mask)[0]
    gap_sizes_s = dt[mask]
    return gap_indices, gap_sizes_s


def sigma_z_at_gaps(posterior_width, gap_indices, window=50):
    """
    Check if posterior_width (sigma_z) spikes around data gap timestamps.

    For each gap, computes the local maximum within ±window timesteps and
    reports it as a z-score relative to the global distribution.

    Args:
        posterior_width: (N,) from posterior_diagnostics.compute_posterior_width().
        gap_indices:     array of gap start indices from detect_data_gaps().
        window:          half-window in timesteps around each gap (default 50 = 0.5s).

    Returns:
        list of dicts, one per gap:
            gap_index, gap_local_max, global_mean, global_std, z_score
    """
    pw = np.asarray(posterior_width)
    global_mean = float(np.mean(pw))
    global_std = float(np.std(pw))

    results = []
    for idx in gap_indices:
        lo = max(0, idx - window)
        hi = min(len(pw) - 1, idx + window)
        local_pw = pw[lo : hi + 1]
        local_max = float(np.max(local_pw))
        z_score = (local_max - global_mean) / max(global_std, 1e-8)
        results.append(
            {
                "gap_index": int(idx),
                "gap_local_max": local_max,
                "global_mean": global_mean,
                "global_std": global_std,
                "z_score": float(z_score),
            }
        )
    return results


def summarize_anomaly_response(result, seq_name=""):
    """
    Full anomaly diagnostic for a single sequence result dict.

    Args:
        result:   dict from evaluate_sequence() with world_model_output.
        seq_name: label for logging.

    Returns:
        dict with keys: gap_indices, gap_sizes_s, spike_stats, mean_sigma_z_at_gaps
    """
    from src.evaluation.posterior_diagnostics import compute_posterior_width

    t = result["t"]
    wm = result.get("world_model_output")

    gap_indices, gap_sizes = detect_data_gaps(t)

    if wm is None or wm.get("log_var_z") is None:
        return {
            "gap_indices": gap_indices,
            "gap_sizes_s": gap_sizes,
            "n_gaps": len(gap_indices),
            "spike_stats": None,
            "mean_sigma_z_at_gaps": None,
        }

    pw = compute_posterior_width(wm["log_var_z"])
    spike_stats = sigma_z_at_gaps(pw, gap_indices)

    mean_at_gaps = float(
        np.mean([s["gap_local_max"] for s in spike_stats])
    ) if spike_stats else float("nan")

    return {
        "gap_indices": gap_indices,
        "gap_sizes_s": gap_sizes,
        "n_gaps": len(gap_indices),
        "spike_stats": spike_stats,
        "mean_sigma_z_at_gaps": mean_at_gaps,
        "global_mean_sigma_z": spike_stats[0]["global_mean"] if spike_stats else float("nan"),
        "mean_z_score": float(np.mean([s["z_score"] for s in spike_stats])) if spike_stats else float("nan"),
        "posterior_width": pw,
    }
