#!/usr/bin/env python3
"""
Z Posterior Diagnostic evaluation (Evaluation 3).

Loads saved result pickles from Condition D, runs all 5 diagnostics (3a-3e),
and saves figures to output_dir.

Usage:
    python scripts/eval_posterior_diagnostics.py \
        --results_D path/to/results_D \
        --output_dir results/posterior_diagnostics

Results directory must contain <drive_name>_filter.p files as saved by
src/test.py with a world-model checkpoint (Condition D).
"""

import argparse
import os
import pickle
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.kitti_sequences import ODOM_TO_DRIVE, DRIVE_TO_ODOM
from src.evaluation.posterior_diagnostics import (
    run_all_diagnostics,
    posterior_omega_correlation,
    check_collapse,
    compute_posterior_width,
)
from src.evaluation.posterior_plots import (
    plot_posterior_width_timeseries,
    plot_pca_latent_space,
    plot_nn_vs_omega,
    plot_bias_corrections_vs_omega,
    plot_diagnostic_panel,
)

# Sequences for each diagnostic
SEQ_08 = ODOM_TO_DRIVE[8]  # longest sequence — posterior width, PCA, bias corrections
SEQ_01 = ODOM_TO_DRIVE[1]  # characteristic bend at t=90-110s — N_n response
TEST_SEQ_NUMS = [1, 4, 6, 7, 8, 9, 10]


def load_results(results_dir):
    results = {}
    if results_dir and os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith("_filter.p"):
                drive = fname[: -len("_filter.p")]
                with open(os.path.join(results_dir, fname), "rb") as f:
                    results[drive] = pickle.load(f)
    return results


def check_world_model_output(result, name):
    wm = result.get("world_model_output")
    if wm is None:
        print(f"  WARNING: {name} has no world_model_output — "
              f"was it evaluated with a world-model checkpoint?")
        return False
    if wm.get("mu_z") is None:
        print(f"  WARNING: {name} world_model_output.mu_z is None")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_D", required=True,
                        help="Results dir for Condition D (world model, both heads)")
    parser.add_argument("--output_dir", default="results/posterior_diagnostics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Condition D results from: {args.results_D}")
    results = load_results(args.results_D)
    print(f"  Found {len(results)} sequences: {sorted(results.keys())}")

    # ---- 3a + 3b: sigma_z stats and omega correlation per test sequence ----
    print("\n=== Diagnostics 3a + 3b: Posterior Width & Context Sensitivity ===")
    collapse_summary = {}
    correlation_summary = {}

    for seq_num in TEST_SEQ_NUMS:
        drive = ODOM_TO_DRIVE[seq_num]
        if drive not in results:
            print(f"  seq {seq_num:02d} ({drive}): NOT FOUND")
            continue
        result = results[drive]
        if not check_world_model_output(result, drive):
            continue

        wm = result["world_model_output"]
        pw = compute_posterior_width(wm["log_var_z"])
        collapse = check_collapse(pw)
        collapse_summary[f"seq{seq_num:02d}"] = collapse

        corr = posterior_omega_correlation(wm["log_var_z"], result["u"])
        correlation_summary[f"seq{seq_num:02d}"] = corr

        status = "HEALTHY" if collapse["healthy"] else ("COLLAPSED" if collapse["collapsed"] else "EXPLODED")
        print(f"  seq{seq_num:02d}: mean_sigma_z={collapse['mean']:.3f} [{status}]  "
              f"pearson_r={corr['pearson_r']:.3f} (p={corr['p_value']:.3e})")

    # Save summary JSON
    summary = {"collapse": collapse_summary, "correlation": correlation_summary}
    with open(os.path.join(args.output_dir, "diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved diagnostics summary to {args.output_dir}/diagnostics_summary.json")

    # ---- Individual figures ----

    # 3a: posterior width timeseries for seq 08
    if SEQ_08 in results and check_world_model_output(results[SEQ_08], SEQ_08):
        print(f"\nPlotting 3a: posterior width for seq08...")
        diag_08 = run_all_diagnostics(results[SEQ_08])
        fig = plot_posterior_width_timeseries(
            diag_08["posterior_width"], diag_08["t"], seq_name="seq08"
        )
        fig.savefig(os.path.join(args.output_dir, "3a_posterior_width_seq08.png"), dpi=300)
        plt.close(fig)
        print("  Saved.")
    else:
        diag_08 = None
        print(f"  seq08 not available, skipping 3a/3c/3e individual plots")

    # 3c: PCA latent space for seq 08
    if diag_08 is not None:
        print("Plotting 3c: PCA latent space for seq08...")
        fig = plot_pca_latent_space(
            diag_08["z_pca_2d"], diag_08["omega_z_abs"],
            seq_name="seq08", explained_variance=diag_08["pca_explained_variance"]
        )
        fig.savefig(os.path.join(args.output_dir, "3c_pca_latent_space_seq08.png"), dpi=300)
        plt.close(fig)
        print("  Saved.")

    # 3e: bias corrections for seq 08
    if diag_08 is not None and diag_08["bias_correction_magnitude"] is not None:
        print("Plotting 3e: bias correction magnitude for seq08...")
        fig = plot_bias_corrections_vs_omega(
            diag_08["bias_correction_magnitude"], diag_08["omega_z_abs"],
            diag_08["t"], seq_name="seq08"
        )
        fig.savefig(os.path.join(args.output_dir, "3e_bias_corrections_seq08.png"), dpi=300)
        plt.close(fig)
        print("  Saved.")
    elif diag_08 is not None:
        print("  Process head corrections not available (head disabled or not returned)")

    # 3d: N_n vs omega for seq 01
    if SEQ_01 in results and check_world_model_output(results[SEQ_01], SEQ_01):
        print("Plotting 3d: N_n response for seq01...")
        diag_01 = run_all_diagnostics(results[SEQ_01])
        if diag_01["n_lat"] is not None:
            fig = plot_nn_vs_omega(
                diag_01["n_lat"], diag_01["n_up"], diag_01["omega_z_abs"],
                diag_01["t"], seq_name="seq01"
            )
            fig.savefig(os.path.join(args.output_dir, "3d_nn_vs_omega_seq01.png"), dpi=300)
            plt.close(fig)
            print("  Saved.")
        else:
            print("  Measurement head not available.")
            diag_01 = None
    else:
        diag_01 = None
        print("  seq01 not available, skipping 3d")

    # Combined panel (all 5 diagnostics)
    if diag_08 is not None and diag_01 is not None:
        print("Plotting combined diagnostic panel...")
        fig = plot_diagnostic_panel(diag_08, diag_01, seq_name_08="seq08", seq_name_01="seq01")
        fig.savefig(os.path.join(args.output_dir, "posterior_diagnostic_panel.png"), dpi=300)
        plt.close(fig)
        print("  Saved.")

    print(f"\nAll outputs saved to: {args.output_dir}")

    # Print decision criteria summary
    print("\n=== Decision Criteria (per spec) ===")
    for seq_key, c in collapse_summary.items():
        status = "OK" if c["healthy"] else "FAIL"
        print(f"  [{status}] {seq_key}: mean_sigma_z={c['mean']:.3f} (healthy: 0.1-0.9)")
    for seq_key, corr in correlation_summary.items():
        pos = "positive" if corr["pearson_r"] > 0 else "NEGATIVE/ZERO"
        print(f"  [{pos}] {seq_key}: pearson_r={corr['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
