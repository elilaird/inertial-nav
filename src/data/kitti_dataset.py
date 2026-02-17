"""
KITTI dataset implementation for IMU dead-reckoning.

Handles loading KITTI raw IMU/GPS data, sequence splitting for
train/validation/test, and dataset-specific configurations.
"""

import os
import datetime
import glob
from collections import namedtuple, OrderedDict

import numpy as np
import torch
from termcolor import cprint

from src.data.base_dataset import BaseIMUDataset


class KITTIDataset(BaseIMUDataset):
    """
    KITTI raw dataset for IMU dead-reckoning.

    Provides access to preprocessed KITTI sequences stored as pickle files.
    Sequences are split into train/validation/test based on configuration.

    Args:
        cfg: OmegaConf config with dataset and path settings.
        split: One of "train", "val", "test", or "all".
    """

    OxtsPacket = namedtuple(
        "OxtsPacket",
        "lat, lon, alt, roll, pitch, yaw, "
        "vn, ve, vf, vl, vu, "
        "ax, ay, az, af, al, au, "
        "wx, wy, wz, wf, wl, wu, "
        "pos_accuracy, vel_accuracy, "
        "navstat, numsats, posmode, velmode, orimode",
    )

    OxtsData = namedtuple("OxtsData", "packet, T_w_imu")

    # Minimum sequence length (25s at 100Hz)
    min_seq_dim = 25 * 100

    # Sequences with known data quality issues
    datasets_fake = [
        "2011_09_26_drive_0093_extract",
        "2011_09_28_drive_0039_extract",
        "2011_09_28_drive_0002_extract",
    ]

    # KITTI odometry benchmark frame ranges
    odometry_benchmark = OrderedDict(
        [
            ("2011_10_03_drive_0027_extract", [0, 45692]),
            ("2011_10_03_drive_0042_extract", [0, 12180]),
            ("2011_10_03_drive_0034_extract", [0, 47935]),
            ("2011_09_26_drive_0067_extract", [0, 8000]),
            ("2011_09_30_drive_0016_extract", [0, 2950]),
            ("2011_09_30_drive_0018_extract", [0, 28659]),
            ("2011_09_30_drive_0020_extract", [0, 11347]),
            ("2011_09_30_drive_0027_extract", [0, 11545]),
            ("2011_09_30_drive_0028_extract", [11231, 53650]),
            ("2011_09_30_drive_0033_extract", [0, 16589]),
            ("2011_09_30_drive_0034_extract", [0, 12744]),
        ]
    )

    def __init__(self, cfg, split="all"):
        # Extract test/validation sequence names from config before calling super
        dataset_cfg = cfg.get("dataset", cfg)
        self._test_sequences = dataset_cfg.get(
            "test_sequences", ["2011_09_30_drive_0028_extract"]
        )
        self._validation_sequences = dataset_cfg.get(
            "cross_validation_sequences", ["2011_09_30_drive_0028_extract"]
        )

        # Training filter config (sequences with frame ranges)
        self._train_filter_cfg = dataset_cfg.get("train_filter", None)
        self._validation_filter_cfg = dataset_cfg.get(
            "validation_filter", None
        )

        super().__init__(cfg, split)

    def _load_sequences(self):
        """Load KITTI sequences from pickle files."""
        self.datasets_test = list(self._test_sequences)
        self.datasets_validation = list(self._validation_sequences)

        print(f"Loading KITTI dataset from {self.path_data_save}...")

        # Discover all available sequences from data directory
        if os.path.isdir(self.path_data_save):
            for f in sorted(os.listdir(self.path_data_save)):
                if f.endswith(self.pickle_extension):
                    name = f[: -len(self.pickle_extension)]
                    self.datasets.append(name)

        # Remove known bad sequences
        for fake in self.datasets_fake:
            if fake in self.datasets:
                self.datasets.remove(fake)

        # Split into train (anything not in test or validation)
        for dataset in self.datasets:
            if (
                dataset not in self.datasets_test
                and dataset not in self.datasets_validation
            ):
                self.datasets_train.append(dataset)

        # Set up training filter with frame ranges
        if self._train_filter_cfg:
            for name, frames in self._train_filter_cfg.items():
                self.datasets_train_filter[name] = list(frames)
        else:
            # Default KITTI training filter
            self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [
                0,
                None,
            ]
            self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [
                0,
                15000,
            ]
            self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [
                0,
                None,
            ]
            self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [
                0,
                None,
            ]
            self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [
                0,
                None,
            ]
            self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [
                0,
                18000,
            ]
            self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [
                0,
                31000,
            ]
            self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [
                0,
                None,
            ]

        # Set up validation filter
        if self._validation_filter_cfg:
            for name, frames in self._validation_filter_cfg.items():
                self.datasets_validatation_filter[name] = list(frames)
        else:
            self.datasets_validatation_filter[
                "2011_09_30_drive_0028_extract"
            ] = [11231, 53650]

    # ========== Raw Data Reading (KITTI-specific) ==========

    @staticmethod
    def read_data(cfg):
        """
        Read raw KITTI data and convert to pickle format.

        Args:
            cfg: Config with path_data_base and path_data_save.
        """
        from src.utils.geometry import to_rpy

        path_data_base = cfg.get("path_data_base", cfg.get("path_raw", ""))
        path_data_save = cfg.get("path_data_save", "../data")

        os.makedirs(path_data_save, exist_ok=True)

        print("Start read_data")
        t_tot = 0
        date_dirs = os.listdir(path_data_base)

        for date_dir in date_dirs:
            path1 = os.path.join(path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue

            for date_dir2 in os.listdir(path1):
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue

                oxts_files = sorted(
                    glob.glob(os.path.join(path2, "oxts", "data", "*.txt"))
                )
                oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

                print("\n Sequence name: " + date_dir2)
                if len(oxts) < KITTIDataset.min_seq_dim:
                    cprint(
                        "Dataset is too short ({:.2f} s)".format(
                            len(oxts) / 100
                        ),
                        "yellow",
                    )
                    continue

                N = len(oxts)
                t = KITTIDataset.load_timestamps(path2)
                acc_bis = np.zeros((N, 3))
                gyro_bis = np.zeros((N, 3))
                p_gt = np.zeros((N, 3))
                v_gt = np.zeros((N, 3))
                roll_gt = np.zeros(N)
                pitch_gt = np.zeros(N)
                yaw_gt = np.zeros(N)

                for k in range(N):
                    oxts_k = oxts[k]
                    t[k] = (
                        3600 * t[k].hour
                        + 60 * t[k].minute
                        + t[k].second
                        + t[k].microsecond / 1e6
                    )
                    acc_bis[k] = [oxts_k[0].ax, oxts_k[0].ay, oxts_k[0].az]
                    gyro_bis[k] = [oxts_k[0].wx, oxts_k[0].wy, oxts_k[0].wz]
                    v_gt[k] = [oxts_k[0].ve, oxts_k[0].vn, oxts_k[0].vu]
                    p_gt[k] = oxts_k[1][:3, 3]
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = to_rpy(Rot_gt_k)

                t = np.array(t) - t[0]
                if np.max(t[:-1] - t[1:]) > 0.1:
                    cprint(date_dir2 + " has time problem", "yellow")

                ang_gt = np.stack([roll_gt, pitch_gt, yaw_gt], axis=1)
                u = np.concatenate((gyro_bis, acc_bis), axis=-1)

                # Convert to tensors
                mondict = {
                    "t": torch.from_numpy(t).float(),
                    "p_gt": torch.from_numpy(p_gt).float(),
                    "ang_gt": torch.from_numpy(ang_gt).float(),
                    "v_gt": torch.from_numpy(v_gt).float(),
                    "u": torch.from_numpy(u).float(),
                    "name": date_dir2,
                }

                t_tot += t[-1] - t[0]
                BaseIMUDataset.dump(mondict, path_data_save, date_dir2)

        print("\n Total dataset duration: {:.2f} s".format(t_tot))

    # ========== KITTI OXTS Parsing Utilities ==========

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Read OXTS data files and compute poses in ENU frame."""
        scale = None
        origin = None
        oxts = []

        for filename in oxts_files:
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.split()
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]
                    packet = KITTIDataset.OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.0)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(
                        R, t - origin
                    )
                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))
        return oxts

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Compute SE(3) pose from OXTS packet using Mercator projection."""
        er = 6378137.0  # earth radius in meters
        tx = scale * packet.lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        c_r, s_r = np.cos(packet.roll), np.sin(packet.roll)
        c_p, s_p = np.cos(packet.pitch), np.sin(packet.pitch)
        c_y, s_y = np.cos(packet.yaw), np.sin(packet.yaw)

        Rx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        Ry = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        Rz = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        R = Rz.dot(Ry.dot(Rx))

        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Build 4x4 homogeneous transform from R and t."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from KITTI oxts timestamps file."""
        timestamp_file = os.path.join(data_path, "oxts", "timestamps.txt")
        timestamps = []
        with open(timestamp_file, "r") as f:
            for line in f.readlines():
                t = datetime.datetime.strptime(
                    line[:-4], "%Y-%m-%d %H:%M:%S.%f"
                )
                timestamps.append(t)
        return timestamps
