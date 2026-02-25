#!/usr/bin/env python3
"""
Generate ablation table (Evaluation 1) from saved result pickles.

Usage:
    python scripts/generate_ablation_table.py \
        --results_A path/to/results_A \
        --results_B path/to/results_B \
        --results_C path/to/results_C \
        --results_D path/to/results_D \
        --output ablation_table.md

Results directories should contain per-sequence pickles named
<drive_name>_filter.p, as saved by src/test.py.
"""

import argparse
import os
import pickle
import numpy as np
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.kitti_sequences import ODOM_TO_DRIVE, DRIVE_TO_ODOM

# Test sequences as odometry numbers (AI-IMU Table 1)
TEST_SEQ_NUMS = [1, 4, 6, 7, 8, 9, 10]


def load_results(results_dir):
    """Load all result pickles from a directory. Returns {drive_name: results}."""
    results = {}
    if results_dir is None or not os.path.isdir(results_dir):
        return results
    for fname in os.listdir(results_dir):
        if fname.endswith("_filter.p"):
            drive_name = fname[: -len("_filter.p")]
            with open(os.path.join(results_dir, fname), "rb") as f:
                results[drive_name] = pickle.load(f)
    return results


def extract_metrics(results, drive_name):
    """Return (t_rel, r_rel) for a sequence, or (nan, nan) if missing."""
    if drive_name not in results:
        return float("nan"), float("nan")
    m = results[drive_name]["metrics"]["rpe"]
    return m["t_rel"], m["r_rel"]


def build_table(results_by_condition, fmt="markdown"):
    """
    Build ablation table.

    Args:
        results_by_condition: dict mapping condition name → {drive_name: results}
        fmt: "markdown" or "latex"

    Returns:
        Formatted table string.
    """
    conditions = ["A (AI-IMU)", "B (Meas only)", "C (Proc only)", "D (Both)"]
    cond_keys = ["A", "B", "C", "D"]

    rows = []
    for seq_num in TEST_SEQ_NUMS:
        drive = ODOM_TO_DRIVE[seq_num]
        row = {"seq": seq_num}
        for ckey in cond_keys:
            res = results_by_condition.get(ckey, {})
            t_rel, r_rel = extract_metrics(res, drive)
            row[f"{ckey}_t"] = t_rel
            row[f"{ckey}_r"] = r_rel
        rows.append(row)

    # Averages (ignoring nan)
    avg_row = {"seq": "Avg"}
    for ckey in cond_keys:
        t_vals = [r[f"{ckey}_t"] for r in rows if not np.isnan(r[f"{ckey}_t"])]
        r_vals = [r[f"{ckey}_r"] for r in rows if not np.isnan(r[f"{ckey}_r"])]
        avg_row[f"{ckey}_t"] = np.mean(t_vals) if t_vals else float("nan")
        avg_row[f"{ckey}_r"] = np.mean(r_vals) if r_vals else float("nan")
    rows.append(avg_row)

    if fmt == "markdown":
        return _markdown_table(rows, conditions, cond_keys)
    elif fmt == "latex":
        return _latex_table(rows, conditions, cond_keys)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _markdown_table(rows, conditions, cond_keys):
    lines = []
    # Header
    header = "| Seq |"
    for c in conditions:
        header += f" {c} t_rel% | {c} r_rel |"
    lines.append(header)

    sep = "|-----|"
    for _ in conditions:
        sep += "---------|---------|"
    lines.append(sep)

    for row in rows:
        seq = row["seq"]
        line = f"| {seq:>3} |"
        for ckey in cond_keys:
            t = row[f"{ckey}_t"]
            r = row[f"{ckey}_r"]
            t_str = f"{t:.2f}" if not np.isnan(t) else "—"
            r_str = f"{r:.4f}" if not np.isnan(r) else "—"
            line += f" {t_str:>7} | {r_str:>7} |"
        lines.append(line)

    return "\n".join(lines)


def _latex_table(rows, conditions, cond_keys):
    n_cond = len(conditions)
    col_spec = "c" + "cc" * n_cond
    lines = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Seq & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in conditions)
        + r" \\",
        " & "
        + " & ".join(r"$t_{rel}$\% & $r_{rel}$" for _ in conditions)
        + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if row["seq"] == "Avg":
            lines.append(r"\midrule")
        seq = row["seq"]
        vals = []
        for ckey in cond_keys:
            t = row[f"{ckey}_t"]
            r = row[f"{ckey}_r"]
            vals.append(f"{t:.2f}" if not np.isnan(t) else "---")
            vals.append(f"{r:.4f}" if not np.isnan(r) else "---")
        lines.append(f"{seq} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation table")
    parser.add_argument(
        "--results_A", default=None, help="Results dir for Condition A"
    )
    parser.add_argument(
        "--results_B", default=None, help="Results dir for Condition B"
    )
    parser.add_argument(
        "--results_C", default=None, help="Results dir for Condition C"
    )
    parser.add_argument(
        "--results_D", default=None, help="Results dir for Condition D"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--format", choices=["markdown", "latex"], default="markdown"
    )
    args = parser.parse_args()

    results_by_condition = {
        "A": load_results(args.results_A),
        "B": load_results(args.results_B),
        "C": load_results(args.results_C),
        "D": load_results(args.results_D),
    }

    # Report which sequences are missing per condition
    for ckey, results in results_by_condition.items():
        found = set(
            DRIVE_TO_ODOM.get(d) for d in results if d in DRIVE_TO_ODOM
        )
        missing = [n for n in TEST_SEQ_NUMS if n not in found]
        if missing:
            print(f"Condition {ckey}: missing sequences {missing}")
        else:
            print(
                f"Condition {ckey}: all {len(TEST_SEQ_NUMS)} sequences found"
            )

    table = build_table(results_by_condition, fmt=args.format)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Table written to {args.output}")
    else:
        print("\n" + table)


if __name__ == "__main__":
    main()
