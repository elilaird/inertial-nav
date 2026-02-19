#!/usr/bin/env python3
"""
Compare results from ablation study (baseline vs process_only vs full).

Loads evaluation results from each condition and generates a comparison
table showing how each component (learned covariances, bias correction)
affects performance metrics.
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(result_dir):
    """
    Load evaluation results from directory.

    Looks for pickle files or JSON files from evaluate_sequence outputs.
    """
    results_dir = Path(result_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {result_dir}")

    results = {}

    # Try to load pickle files first (from evaluate_sequence)
    for pkl_file in results_dir.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            seq_name = pkl_file.stem
            results[seq_name] = pickle.load(f)

    # If no pickle files, try JSON
    if not results:
        for json_file in results_dir.glob("*.json"):
            with open(json_file, "r") as f:
                seq_name = json_file.stem
                data = json.load(f)
                # Extract metrics if they exist
                if "metrics" in data:
                    results[seq_name] = data["metrics"]

    return results


def extract_metrics(results_by_condition):
    """
    Extract summary statistics from results.

    Args:
        results_by_condition: Dict[condition_name, Dict[seq_name, metrics]]

    Returns:
        Dict mapping condition -> Dict mapping metric -> value
    """
    summary = {}

    for condition, seq_results in results_by_condition.items():
        condition_metrics = {
            "t_rel": [],
            "r_rel": [],
            "ate_rmse": [],
            "orient_error": [],
        }

        for seq_name, seq_data in seq_results.items():
            # Handle both direct metrics and nested format
            if isinstance(seq_data, dict) and "metrics" in seq_data:
                m = seq_data["metrics"]
            else:
                m = seq_data

            # Extract each metric
            if "rpe" in m and m["rpe"]:
                condition_metrics["t_rel"].append(
                    m["rpe"].get("t_rel", np.nan)
                )
                condition_metrics["r_rel"].append(
                    m["rpe"].get("r_rel", np.nan)
                )

            if "ate" in m and m["ate"]:
                condition_metrics["ate_rmse"].append(
                    m["ate"].get("rmse", np.nan)
                )

            if "orientation_error" in m and m["orientation_error"]:
                condition_metrics["orient_error"].append(
                    m["orientation_error"].get("mean_deg", np.nan)
                )

        # Compute means (ignoring NaN values)
        summary[condition] = {
            "t_rel_mean": float(np.nanmean(condition_metrics["t_rel"])),
            "t_rel_std": float(np.nanstd(condition_metrics["t_rel"])),
            "r_rel_mean": float(np.nanmean(condition_metrics["r_rel"])),
            "r_rel_std": float(np.nanstd(condition_metrics["r_rel"])),
            "ate_rmse_mean": float(np.nanmean(condition_metrics["ate_rmse"])),
            "ate_rmse_std": float(np.nanstd(condition_metrics["ate_rmse"])),
            "orient_error_mean": float(
                np.nanmean(condition_metrics["orient_error"])
            ),
            "orient_error_std": float(
                np.nanstd(condition_metrics["orient_error"])
            ),
        }

    return summary


def compute_improvements(summary):
    """
    Compute improvements from baseline for each component.

    Returns:
        Dict with improvement percentages
    """
    baseline_metrics = summary.get("baseline", {})
    learned_cov_metrics = summary.get("learned_cov", {})
    learned_dynamics_metrics = summary.get("learned_dynamics", {})

    improvements = {
        "learned_cov_vs_baseline": {},
        "learned_dynamics_vs_learned_cov": {},
        "learned_dynamics_vs_baseline": {},
    }

    # Compute improvements (lower is better for all metrics)
    if baseline_metrics and learned_cov_metrics:
        for key in baseline_metrics:
            baseline_val = baseline_metrics[key]
            learned_cov_val = learned_cov_metrics[key]
            if baseline_val > 0:
                improvement = (
                    (baseline_val - learned_cov_val) / baseline_val * 100
                )
                improvements["learned_cov_vs_baseline"][key] = improvement

    if learned_cov_metrics and learned_dynamics_metrics:
        for key in learned_cov_metrics:
            learned_cov_val = learned_cov_metrics[key]
            learned_dynamics_val = learned_dynamics_metrics[key]
            if learned_cov_val > 0:
                improvement = (
                    (learned_cov_val - learned_dynamics_val)
                    / learned_cov_val
                    * 100
                )
                improvements["learned_dynamics_vs_learned_cov"][
                    key
                ] = improvement

    if baseline_metrics and learned_dynamics_metrics:
        for key in baseline_metrics:
            baseline_val = baseline_metrics[key]
            learned_dynamics_val = learned_dynamics_metrics[key]
            if baseline_val > 0:
                improvement = (
                    (baseline_val - learned_dynamics_val) / baseline_val * 100
                )
                improvements["learned_dynamics_vs_baseline"][key] = improvement

    return improvements


def format_table(summary, improvements):
    """Format results as a nicely aligned table."""
    lines = []

    lines.append("=" * 100)
    lines.append("Ablation Study Results: Learned Bias Correction Network")
    lines.append("=" * 100)
    lines.append("")

    # Metrics table
    lines.append("Mean Metrics by Condition:")
    lines.append("-" * 100)
    lines.append(
        f"{'Metric':<20} {'Baseline':<20} {'Learned Cov':<20} {'Learned Dynamics':<20} {'Units':<10}"
    )
    lines.append("-" * 100)

    metric_labels = [
        ("t_rel_mean", "t_rel_std", "Translational Error", "%"),
        ("r_rel_mean", "r_rel_std", "Rotational Error", "deg/m"),
        ("ate_rmse_mean", "ate_rmse_std", "ATE RMSE", "m"),
        ("orient_error_mean", "orient_error_std", "Orientation Error", "deg"),
    ]

    for mean_key, std_key, label, unit in metric_labels:
        baseline_val = summary.get("baseline", {}).get(mean_key, np.nan)
        baseline_std = summary.get("baseline", {}).get(std_key, np.nan)
        learned_cov_val = summary.get("learned_cov", {}).get(mean_key, np.nan)
        learned_cov_std = summary.get("learned_cov", {}).get(std_key, np.nan)
        learned_dynamics_val = summary.get("learned_dynamics", {}).get(
            mean_key, np.nan
        )
        learned_dynamics_std = summary.get("learned_dynamics", {}).get(
            std_key, np.nan
        )

        baseline_str = (
            f"{baseline_val:.3f}±{baseline_std:.3f}"
            if not np.isnan(baseline_val)
            else "N/A"
        )
        learned_cov_str = (
            f"{learned_cov_val:.3f}±{learned_cov_std:.3f}"
            if not np.isnan(learned_cov_val)
            else "N/A"
        )
        learned_dynamics_str = (
            f"{learned_dynamics_val:.3f}±{learned_dynamics_std:.3f}"
            if not np.isnan(learned_dynamics_val)
            else "N/A"
        )

        lines.append(
            f"{label:<20} {baseline_str:<20} {learned_cov_str:<20} {learned_dynamics_str:<20} {unit:<10}"
        )

    lines.append("")
    lines.append("Improvements (positive = better):")
    lines.append("-" * 100)

    # Learned_cov vs Baseline
    lines.append("Learned Cov vs Baseline (effect of adaptive covariances):")
    for key in [
        "t_rel_mean",
        "r_rel_mean",
        "ate_rmse_mean",
        "orient_error_mean",
    ]:
        improv = improvements["learned_cov_vs_baseline"].get(key, np.nan)
        if not np.isnan(improv):
            lines.append(f"  {key:<25} {improv:+.2f}%")

    lines.append("")

    # Learned_dynamics vs Learned_cov
    lines.append(
        "Learned Dynamics vs Learned Cov (effect of bias correction net):"
    )
    for key in [
        "t_rel_mean",
        "r_rel_mean",
        "ate_rmse_mean",
        "orient_error_mean",
    ]:
        improv = improvements["learned_dynamics_vs_learned_cov"].get(
            key, np.nan
        )
        if not np.isnan(improv):
            lines.append(f"  {key:<25} {improv:+.2f}%")

    lines.append("")

    # Learned_dynamics vs Baseline
    lines.append("Learned Dynamics vs Baseline (cumulative improvement):")
    for key in [
        "t_rel_mean",
        "r_rel_mean",
        "ate_rmse_mean",
        "orient_error_mean",
    ]:
        improv = improvements["learned_dynamics_vs_baseline"].get(key, np.nan)
        if not np.isnan(improv):
            lines.append(f"  {key:<25} {improv:+.2f}%")

    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ablation study results"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Results directory for baseline condition",
    )
    parser.add_argument(
        "--learned_cov",
        required=True,
        help="Results directory for learned-covariances condition (iekf_learned_cov)",
    )
    parser.add_argument(
        "--learned_dynamics",
        required=True,
        help="Results directory for full system condition (iekf_learned_dynamics)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated list of sequences to include (optional filter)",
    )

    args = parser.parse_args()

    # Load results from all conditions
    print("Loading results...")
    results_by_condition = {
        "baseline": load_results(args.baseline),
        "learned_cov": load_results(args.learned_cov),
        "learned_dynamics": load_results(args.learned_dynamics),
    }

    # Filter by sequences if specified
    if args.sequences:
        seq_filter = set(
            f"seq_{s:02d}" for s in map(int, args.sequences.split(","))
        )
        seq_filter.update(
            f"2011_09_30_drive_{int(s):04d}_extract"
            for s in args.sequences.split(",")
        )
        for condition in results_by_condition:
            results_by_condition[condition] = {
                k: v
                for k, v in results_by_condition[condition].items()
                if k in seq_filter
                or any(s in k for s in args.sequences.split(","))
            }

    # Extract and compute metrics
    print("Computing metrics...")
    summary = extract_metrics(results_by_condition)
    improvements = compute_improvements(summary)

    # Format and output
    table_str = format_table(summary, improvements)
    print(table_str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table_str)
        print(f"\nComparison table saved to: {args.output}")


if __name__ == "__main__":
    main()
