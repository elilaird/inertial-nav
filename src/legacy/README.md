# Legacy Code

This directory contains the original AI-IMU Dead-Reckoning implementation, preserved for reference and backward compatibility.

## Files

- `main_kitti.py` - Original entry point with KITTI dataset class and parameters
- `dataset.py` - Original base dataset class
- `train_torch_filter.py` - Original training loop
- `utils_torch_filter.py` - Original PyTorch IEKF with neural networks
- `utils_numpy_filter.py` - Original NumPy IEKF implementation
- `utils_plot.py` - Original visualization utilities
- `utils.py` - Original data preparation utilities

## Running Legacy Code

```bash
cd src/legacy
python main_kitti.py
```

Note: The legacy code expects data in `../../data/` and writes results to `../../results/`.

## Converting Legacy Checkpoints

Use the conversion script to migrate legacy checkpoints to the modern format:

```bash
python scripts/convert_legacy_checkpoint.py --input temp/iekfnets.p --output checkpoints/converted.pth
```
