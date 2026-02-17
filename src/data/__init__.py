"""
Data loading and preprocessing for IMU dead-reckoning.

Usage:
    from src.data import KITTIDataset, BaseIMUDataset
    from src.data.transforms import AddIMUNoise, NormalizeIMU, Compose
"""

from src.data.base_dataset import BaseIMUDataset
from src.data.kitti_dataset import KITTIDataset
