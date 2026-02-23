#!/usr/bin/env python3
"""
Maneuver-Stratified RTE evaluation (Evaluation 2).

Segments test sequences by yaw-rate magnitude and computes position error
in straight vs corner segments for Conditions A and D.

Usage:
    python scripts/eval_maneuver_stratified.py \
        --results_A path/to/results_A \
        --results_D path/to/results_D \
        --output_dir results/maneuver_stratified

Results directories contain per-sequence pickles <drive_name>_filter.p
as saved by src/test.py.
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.kitti_sequences import ODOM_TO_DRIVE, DRIVE_TO_ODOM
from src.evaluation.maneuver_stratification import (
    segment_by_yaw_rate,
    compute_stratified_position_error,
    stratified_comparison,
    print_stratified_table,
)

TEST_SEQ_NUMS = [1, 4, 6, 7, 8, 9, 10]
TEST_DRIVES = [ODOM_TO_DRIVE[n] for n in TEST_SEQ_NUMS]


def load_results(results_dir):
    results = {}
    if results_dir and os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith("_filter.p"):
                drive = fname[: -len("_filter.p")]
                with open(os.path.join(results_dir, fname), "rb") as f:
                    results[drive] = pickle.load(f)
    return results


def plot_stratified_comparison(comparison, output_dir):
    """Bar chart: straight vs corner mean error for A and D."""
    seqs = [n for n in TEST_DRIVES if n in comparison]
    short_names = [f"seq{DRIVE_TO_ODOM[s]:02d}" for s in seqs]

    segments = ["straight", "corner"]
    n_seqs = len(seqs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax_idx, seg in enumerate(segments):
        ax = axes[ax_idx]
        x = np.arange(n_seqs)
        width = 0.35

        vals_A = [comparison[s]["condition_A"][seg]["mean_err"] for s in seqs]
        vals_D = [comparison[s]["condition_D"][seg]["mean_err"] for s in seqs]

        bars_A = ax.bar(x - width / 2, vals_A, width, label="A (AI-IMU)", color="#4C72B0", alpha=0.85)
        bars_D = ax.bar(x + width / 2, vals_D, width, label="D (Both heads)", color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_ylabel("Mean position error (m)")
        ax.set_title(f"{seg.capitalize()} segments")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Maneuver-Stratified Position Error: A vs D", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "maneuver_stratified_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_A", required=True)
    parser.add_argument("--results_D", required=True)
    parser.add_argument("--output_dir", default="results/maneuver_stratified")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_A = load_results(args.results_A)
    results_D = load_results(args.results_D)

    print(f"Loaded {len(results_A)} Condition A sequences, {len(results_D)} Condition D sequences")

    comparison = stratified_comparison(results_A, results_D, TEST_DRIVES)
    print_stratified_table(comparison)

    plot_stratified_comparison(comparison, args.output_dir)

    # Save comparison data
    out_pickle = os.path.join(args.output_dir, "maneuver_stratified_results.pkl")
    with open(out_pickle, "wb") as f:
        pickle.dump(comparison, f)
    print(f"Saved: {out_pickle}")


if __name__ == "__main__":
    main()
