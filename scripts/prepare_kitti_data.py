#!/usr/bin/env python3
"""
Convert KITTI raw IMU/GPS data to pickle format for training.

Usage:
    python scripts/prepare_kitti_data.py --raw-path /path/to/kitti/raw --output-path data/

Or with Hydra config:
    python scripts/prepare_kitti_data.py --config configs/dataset/kitti.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.kitti_dataset import KITTIDataset


def main():
    parser = argparse.ArgumentParser(description="Prepare KITTI data for training")
    parser.add_argument("--raw-path", type=str, required=True,
                        help="Path to KITTI raw data directory")
    parser.add_argument("--output-path", type=str, default="../data",
                        help="Path to save processed pickle files")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    cfg = {
        "path_data_base": args.raw_path,
        "path_data_save": args.output_path,
    }

    print(f"Reading KITTI raw data from: {args.raw_path}")
    print(f"Saving processed data to: {args.output_path}")

    KITTIDataset.read_data(cfg)
    print("Done!")


if __name__ == '__main__':
    main()
