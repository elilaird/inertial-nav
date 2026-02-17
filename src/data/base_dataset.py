"""
Abstract base class for IMU datasets.

Provides common infrastructure for loading, normalizing, and accessing
IMU sequences with ground truth. Concrete subclasses implement
dataset-specific loading logic.
"""

import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset


class BaseIMUDataset(Dataset, ABC):
    """
    Abstract base dataset for IMU dead-reckoning.

    Handles:
    - Pickle-based data loading and saving
    - Train/validation/test split management
    - Input normalization factor computation
    - Noise injection for data augmentation

    Args:
        cfg: OmegaConf config with dataset parameters.
        split: One of "train", "val", "test", or "all".
    """

    pickle_extension = ".p"
    file_normalize_factor = "normalize_factors.p"

    def __init__(self, cfg, split="all"):
        self.cfg = cfg
        self.split = split

        # Paths
        self.path_data_save = cfg.get("path_data_save", "../data")
        self.path_results = cfg.get("path_results", "../results")
        self.path_temp = cfg.get("path_temp", "../temp")

        # Noise parameters
        noise_cfg = cfg.get("noise", {})
        self.sigma_gyro = noise_cfg.get("sigma_gyro", 1e-4)
        self.sigma_acc = noise_cfg.get("sigma_acc", 1e-4)
        self.sigma_b_gyro = noise_cfg.get("sigma_b_gyro", 1e-5)
        self.sigma_b_acc = noise_cfg.get("sigma_b_acc", 1e-4)

        # Sequence management
        self.datasets = []          # all dataset names
        self.datasets_train = []    # training datasets
        self.datasets_test = []     # test datasets
        self.datasets_validation = []  # validation datasets

        # Training/validation filter dicts: {name: [start_idx, end_idx]}
        self.datasets_train_filter = OrderedDict()
        self.datasets_validatation_filter = OrderedDict()

        # Normalization
        self.normalize_factors = None
        self.num_data = 0

        # RPE data (populated during training)
        self.list_rpe = {}
        self.list_rpe_validation = {}

        # Load datasets and normalization
        self._load_sequences()
        self._compute_normalization()

    @abstractmethod
    def _load_sequences(self):
        """Load sequence names and populate train/val/test splits.

        Must populate:
        - self.datasets (all sequence names)
        - self.datasets_train, self.datasets_test, self.datasets_validation
        - self.datasets_train_filter, self.datasets_validatation_filter
        """
        pass

    def __getitem__(self, i):
        mondict = self.load(self.path_data_save, self.datasets[i])
        return mondict

    def __len__(self):
        return len(self.datasets)

    def dataset_name(self, i):
        """Get dataset name by index."""
        return self.datasets[i]

    def get_data(self, name_or_idx):
        """
        Get a sequence's data by name or index.

        Returns:
            Tuple of (t, ang_gt, p_gt, v_gt, u)
        """
        if isinstance(name_or_idx, int):
            pickle_dict = self[name_or_idx]
        else:
            idx = self.datasets.index(name_or_idx)
            pickle_dict = self[idx]
        return (pickle_dict['t'], pickle_dict['ang_gt'], pickle_dict['p_gt'],
                pickle_dict['v_gt'], pickle_dict['u'])

    def _compute_normalization(self):
        """Compute or load IMU input normalization factors from training data."""
        path_normalize_factor = os.path.join(self.path_temp, self.file_normalize_factor)

        if os.path.isfile(path_normalize_factor):
            pickle_dict = self.load(path_normalize_factor)
            self.normalize_factors = pickle_dict['normalize_factors']
            self.num_data = pickle_dict['num_data']
            return

        if not self.datasets_train:
            return

        # Compute mean
        self.num_data = 0
        u_loc = None
        for i, dataset in enumerate(self.datasets_train):
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if u_loc is None:
                u_loc = u.sum(dim=0)
            else:
                u_loc += u.sum(dim=0)
            self.num_data += u.shape[0]
        u_loc = u_loc / self.num_data

        # Compute standard deviation
        u_std = None
        for dataset in self.datasets_train:
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if u_std is None:
                u_std = ((u - u_loc) ** 2).sum(dim=0)
            else:
                u_std += ((u - u_loc) ** 2).sum(dim=0)
        u_std = (u_std / self.num_data).sqrt()

        self.normalize_factors = {'u_loc': u_loc, 'u_std': u_std}
        pickle_dict = {
            'normalize_factors': self.normalize_factors,
            'num_data': self.num_data,
        }
        self.dump(pickle_dict, path_normalize_factor)

    def normalize(self, u):
        """Normalize IMU inputs using precomputed statistics."""
        u_loc = self.normalize_factors["u_loc"]
        u_std = self.normalize_factors["u_std"]
        return (u - u_loc) / u_std

    def add_noise(self, u):
        """Add random noise to IMU measurements for data augmentation."""
        w = torch.randn_like(u[:, :6])
        w_b = torch.randn_like(u[0, :6])
        w[:, :3] *= self.sigma_gyro
        w[:, 3:6] *= self.sigma_acc
        w_b[:3] *= self.sigma_b_gyro
        w_b[3:6] *= self.sigma_b_acc
        u = u.clone()
        u[:, :6] += w + w_b
        return u

    def init_state_torch_filter(self, iekf):
        """Initialize filter state for training."""
        b_omega0 = torch.zeros(3).double()
        b_acc0 = torch.zeros(3).double()
        Rot_c_i0 = torch.eye(3).double()
        t_c_i0 = torch.zeros(3).double()
        return b_omega0, b_acc0, Rot_c_i0, t_c_i0

    def get_estimates(self, dataset_name):
        """Load saved filter estimates for a sequence."""
        if isinstance(dataset_name, int):
            dataset_name = self.datasets[dataset_name]
        file_name = os.path.join(self.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            return None
        mondict = self.load(file_name)
        return (mondict['Rot'], mondict['v'], mondict['p'],
                mondict['b_omega'], mondict['b_acc'],
                mondict['Rot_c_i'], mondict['t_c_i'],
                mondict['measurements_covs'])

    @classmethod
    def load(cls, *_file_name):
        """Load a pickle file."""
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "rb") as f:
            return pickle.load(f)

    @classmethod
    def dump(cls, mondict, *_file_name):
        """Save a dictionary as a pickle file."""
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "wb") as f:
            pickle.dump(mondict, f)
