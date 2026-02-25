#!/usr/bin/env python3
"""
Anomalous Sequence Evaluation (Evaluation 4).

Checks if the world model encoder (sigma_z) spikes at timestamps where
data gaps occur in sequences 00, 02, 05.

Usage:
    python scripts/eval_anomalous_sequences.py \
        --results_D path/to/results_D \
        --output_dir results/anomalous_sequences

Results directory must contain <drive_name>_filter.p files evaluated with
a world-model (Condition D) checkpoint using the kitti_anomaly_test dataset.
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.kitti_sequences import ODOM_TO_DRIVE, DRIVE_TO_ODOM
from src.evaluation.anomaly_detection import (
    detect_data_gaps,
    summarize_anomaly_response,
)
from src.evaluation.posterior_plots import _apply_style

# Anomalous sequences (odometry numbers)
ANOMALY_SEQ_NUMS = [0, 2, 5]
ANOMALY_DRIVES = [ODOM_TO_DRIVE[n] for n in ANOMALY_SEQ_NUMS]


def load_results(results_dir):
    results = {}
    if results_dir and os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith("_filter.p"):
                drive = fname[: -len("_filter.p")]
                with open(os.path.join(results_dir, fname), "rb") as f:
                    results[drive] = pickle.load(f)
    return results


def plot_sigma_z_with_gaps(posterior_width, t, gap_indices, gap_sizes, seq_name=""):
    """
    Plot sigma_z timeseries with vertical lines at data gap timestamps.

    Args:
        posterior_width: (N,) posterior width per timestep.
        t:               (N,) timestamps.
        gap_indices:     array of gap start indices.
        gap_sizes:       array of gap durations in seconds.
        seq_name:        label for title.

    Returns:
        matplotlib.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 3.5))

    t_rel = t - t[0]
    ax.plot(t_rel, posterior_width, color="#2271B5", linewidth=0.7,
            label=r"$\overline{\sigma}_z$", zorder=2)
    ax.axhspan(0.1, 0.9, color="green", alpha=0.07, label="Healthy [0.1, 0.9]")

    for i, (idx, gap_s) in enumerate(zip(gap_indices, gap_sizes)):
        gap_t = t_rel[idx] if idx < len(t_rel) else t_rel[-1]
        ax.axvline(gap_t, color="red", linestyle="--", linewidth=1.0,
                   alpha=0.8, label="Data gap" if i == 0 else None, zorder=3)
        ax.text(gap_t, ax.get_ylim()[1] * 0.95, f"+{gap_s:.2f}s",
                fontsize=7, color="red", ha="center", va="top", rotation=90)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\overline{\sigma}_z$")
    ax.set_title(f"Posterior Width at Data Gaps — {seq_name}")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_D", required=True,
                        help="Results dir with anomalous sequence results (Condition D)")
    parser.add_argument("--output_dir", default="results/anomalous_sequences")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_D}")
    results = load_results(args.results_D)
    print(f"  Found: {sorted(results.keys())}")

    all_summaries = {}

    print("\n=== Evaluation 4: Anomalous Sequence Sigma_z Response ===")
    for seq_num, drive in zip(ANOMALY_SEQ_NUMS, ANOMALY_DRIVES):
        seq_label = f"seq{seq_num:02d}"
        if drive not in results:
            print(f"  {seq_label}: NOT FOUND in results dir")
            continue

        result = results[drive]
        summary = summarize_anomaly_response(result, seq_name=seq_label)
        all_summaries[seq_label] = {
            "n_gaps": summary["n_gaps"],
            "gap_sizes_s": summary["gap_sizes_s"].tolist(),
            "mean_sigma_z_at_gaps": summary["mean_sigma_z_at_gaps"],
            "global_mean_sigma_z": summary.get("global_mean_sigma_z"),
            "mean_z_score": summary.get("mean_z_score"),
        }

        n_gaps = summary["n_gaps"]
        mean_z = summary.get("mean_z_score", float("nan"))
        sigma_at = summary.get("mean_sigma_z_at_gaps", float("nan"))
        sigma_global = summary.get("global_mean_sigma_z", float("nan"))

        print(f"  {seq_label} ({drive}):")
        print(f"    Data gaps found: {n_gaps}")
        if n_gaps > 0:
            for gs in summary["gap_sizes_s"]:
                print(f"      gap: {gs:.3f}s")
        if summary["spike_stats"] is not None:
            print(f"    mean sigma_z (global):  {sigma_global:.3f}")
            print(f"    mean sigma_z (at gaps): {sigma_at:.3f}")
            print(f"    mean z-score at gaps:   {mean_z:.2f}  "
                  f"({'SPIKE detected' if mean_z > 2.0 else 'no spike'})")
        else:
            print(f"    (no world model output available)")

        # Plot
        if summary.get("posterior_width") is not None:
            fig = plot_sigma_z_with_gaps(
                summary["posterior_width"],
                result["t"],
                summary["gap_indices"],
                summary["gap_sizes_s"],
                seq_name=seq_label,
            )
            out_path = os.path.join(args.output_dir, f"{seq_label}_sigma_z_gaps.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"    Saved: {out_path}")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "anomaly_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Decision criterion printout
    print("\n=== Decision Criterion (per spec) ===")
    print("Elevated sigma_z (z-score > 2) at gap timestamps → encoder detects OOD signal")
    for seq_label, s in all_summaries.items():
        z = s.get("mean_z_score")
        if z is not None and not (isinstance(z, float) and z != z):
            flag = "SPIKE" if z > 2.0 else "flat"
            print(f"  {seq_label}: mean z-score = {z:.2f}  [{flag}]")
        else:
            print(f"  {seq_label}: no data")


if __name__ == "__main__":
    main()
