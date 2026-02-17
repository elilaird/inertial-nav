#!/usr/bin/env python3
"""
Convert legacy checkpoint (temp/iekfnets.p) to modern format.

Legacy format: pickle file containing TORCHIEKF state_dict
Modern format: PyTorch checkpoint with model_state_dict, metadata

Usage:
    python scripts/convert_legacy_checkpoint.py --input temp/iekfnets.p --output checkpoints/converted.pth
"""

import argparse
import os
import sys
import pickle
import torch


# Key mapping from legacy state_dict to modern structure
LEGACY_KEY_MAP = {
    # InitProcessCovNet keys
    'initprocesscov_net.': 'initprocesscov_net.',
    # MesNet keys
    'mes_net.': 'mes_net.',
    # Filter parameters (pass through)
    'P0': 'P0',
    'Q': 'Q',
    'cov0_measurement': 'cov0_measurement',
}


def convert_legacy_checkpoint(input_path, output_path):
    """Convert legacy TORCHIEKF checkpoint to modern format."""

    print(f"Loading legacy checkpoint: {input_path}")

    # Legacy checkpoints are pickled TORCHIEKF state dicts
    with open(input_path, 'rb') as f:
        legacy_data = pickle.load(f)

    # Handle different legacy formats
    if isinstance(legacy_data, dict):
        if 'model_state_dict' in legacy_data:
            # Already in modern-ish format
            state_dict = legacy_data['model_state_dict']
        else:
            # Direct state dict
            state_dict = legacy_data
    else:
        print(f"Error: Unexpected checkpoint format: {type(legacy_data)}")
        sys.exit(1)

    # Map legacy keys to modern keys
    modern_state_dict = {}
    unmapped_keys = []

    for key, value in state_dict.items():
        mapped = False
        for legacy_prefix, modern_prefix in LEGACY_KEY_MAP.items():
            if key.startswith(legacy_prefix) or key == legacy_prefix:
                modern_key = key.replace(legacy_prefix, modern_prefix, 1) if legacy_prefix != key else modern_prefix
                modern_state_dict[modern_key] = value
                mapped = True
                break

        if not mapped:
            # Pass through unmapped keys
            modern_state_dict[key] = value
            unmapped_keys.append(key)

    if unmapped_keys:
        print(f"  Unmapped keys (passed through): {unmapped_keys}")

    # Build modern checkpoint
    checkpoint = {
        'model_state_dict': modern_state_dict,
        'epoch': 0,
        'metrics': {},
        'converted_from': os.path.basename(input_path),
    }

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved modern checkpoint: {output_path}")
    print(f"  Keys: {len(modern_state_dict)}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert legacy checkpoint to modern format")
    parser.add_argument("--input", "-i", required=True, help="Path to legacy checkpoint (e.g., temp/iekfnets.p)")
    parser.add_argument("--output", "-o", required=True, help="Path to save modern checkpoint (e.g., checkpoints/converted.pth)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    convert_legacy_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()
