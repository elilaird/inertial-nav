"""Tests for data pipeline components (Phase 3)."""

import pytest
import torch
import numpy as np
import os
import sys
import pickle
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.base_dataset import BaseIMUDataset
from src.data.kitti_dataset import KITTIDataset
from src.data.transforms import (
    Compose, AddIMUNoise, NormalizeIMU, RandomSubsequenceSampler, ToTensor
)


# ==================== Helper: Create Fake Dataset ====================

def create_fake_pickle_data(path, name, N=1000):
    """Create a minimal fake IMU sequence pickle file."""
    t = torch.linspace(0, N * 0.01, N).float()
    u = torch.randn(N, 6).float()
    ang_gt = torch.randn(N, 3).float() * 0.1
    p_gt = torch.cumsum(torch.randn(N, 3).float() * 0.01, dim=0)
    v_gt = torch.randn(N, 3).float() * 0.1

    mondict = {'t': t, 'u': u, 'ang_gt': ang_gt, 'p_gt': p_gt, 'v_gt': v_gt, 'name': name}
    file_path = os.path.join(path, name + ".p")
    with open(file_path, "wb") as f:
        pickle.dump(mondict, f)
    return file_path


class ConcreteDataset(BaseIMUDataset):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, cfg, split="all", train_names=None, test_names=None, val_names=None):
        self._train_names = train_names or []
        self._test_names = test_names or []
        self._val_names = val_names or []
        super().__init__(cfg, split)

    def _load_sequences(self):
        # Discover sequences from data dir
        if os.path.isdir(self.path_data_save):
            for f in sorted(os.listdir(self.path_data_save)):
                if f.endswith(self.pickle_extension):
                    name = f[:-len(self.pickle_extension)]
                    self.datasets.append(name)

        self.datasets_test = list(self._test_names)
        self.datasets_validation = list(self._val_names)

        for d in self.datasets:
            if d not in self.datasets_test and d not in self.datasets_validation:
                self.datasets_train.append(d)


# ==================== BaseIMUDataset Tests ====================

class TestBaseIMUDataset:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmpdir, "data")
        self.temp_dir = os.path.join(self.tmpdir, "temp")
        os.makedirs(self.data_dir)
        os.makedirs(self.temp_dir)

        # Create fake sequences
        create_fake_pickle_data(self.data_dir, "seq_a", N=500)
        create_fake_pickle_data(self.data_dir, "seq_b", N=600)
        create_fake_pickle_data(self.data_dir, "seq_c", N=400)

        self.cfg = {
            "path_data_save": self.data_dir,
            "path_results": os.path.join(self.tmpdir, "results"),
            "path_temp": self.temp_dir,
            "noise": {
                "sigma_gyro": 1e-4,
                "sigma_acc": 1e-4,
            }
        }

    def test_load_all_sequences(self):
        ds = ConcreteDataset(self.cfg, train_names=[], test_names=["seq_c"])
        assert len(ds) == 3
        assert "seq_a" in ds.datasets
        assert "seq_b" in ds.datasets
        assert "seq_c" in ds.datasets

    def test_train_test_split(self):
        ds = ConcreteDataset(self.cfg, test_names=["seq_c"])
        assert "seq_c" in ds.datasets_test
        assert "seq_c" not in ds.datasets_train
        assert "seq_a" in ds.datasets_train
        assert "seq_b" in ds.datasets_train

    def test_getitem_returns_dict(self):
        ds = ConcreteDataset(self.cfg)
        item = ds[0]
        assert 't' in item
        assert 'u' in item
        assert 'ang_gt' in item
        assert 'p_gt' in item
        assert 'v_gt' in item

    def test_get_data_by_name(self):
        ds = ConcreteDataset(self.cfg)
        t, ang_gt, p_gt, v_gt, u = ds.get_data("seq_a")
        assert t.shape[0] == 500
        assert u.shape == (500, 6)

    def test_get_data_by_index(self):
        ds = ConcreteDataset(self.cfg)
        t, ang_gt, p_gt, v_gt, u = ds.get_data(0)
        assert t.shape[0] > 0

    def test_normalization_factors_computed(self):
        ds = ConcreteDataset(self.cfg)
        if ds.normalize_factors is not None:
            assert 'u_loc' in ds.normalize_factors
            assert 'u_std' in ds.normalize_factors
            assert ds.normalize_factors['u_loc'].shape == (6,)

    def test_normalization_factors_cached(self):
        ds1 = ConcreteDataset(self.cfg)
        # Second load should use cached file
        ds2 = ConcreteDataset(self.cfg)
        if ds1.normalize_factors is not None:
            assert torch.allclose(ds1.normalize_factors['u_loc'],
                                  ds2.normalize_factors['u_loc'])

    def test_add_noise(self):
        ds = ConcreteDataset(self.cfg)
        u = torch.randn(100, 6).float()
        u_noisy = ds.add_noise(u)
        assert u_noisy.shape == u.shape
        # Should be different due to noise
        assert not torch.allclose(u, u_noisy)

    def test_add_noise_does_not_modify_original(self):
        ds = ConcreteDataset(self.cfg)
        u = torch.randn(100, 6).float()
        u_orig = u.clone()
        ds.add_noise(u)
        assert torch.allclose(u, u_orig)

    def test_pickle_load_dump(self):
        data = {'key': torch.tensor([1.0, 2.0, 3.0])}
        path = os.path.join(self.tmpdir, "test_dump")
        BaseIMUDataset.dump(data, path)
        loaded = BaseIMUDataset.load(path)
        assert torch.allclose(loaded['key'], data['key'])

    def test_dataset_name(self):
        ds = ConcreteDataset(self.cfg)
        name = ds.dataset_name(0)
        assert name in ds.datasets


# ==================== Transform Tests ====================

class TestTransforms:
    def _make_data(self, N=200):
        return {
            't': torch.linspace(0, 2, N).float(),
            'u': torch.randn(N, 6).float(),
            'ang_gt': torch.randn(N, 3).float() * 0.1,
            'p_gt': torch.cumsum(torch.randn(N, 3).float() * 0.01, dim=0),
            'v_gt': torch.randn(N, 3).float() * 0.1,
        }

    def test_add_imu_noise(self):
        data = self._make_data()
        u_orig = data['u'].clone()
        transform = AddIMUNoise(sigma_gyro=0.1, sigma_acc=0.1)
        out = transform(data)
        assert not torch.allclose(out['u'], u_orig)

    def test_normalize_imu(self):
        data = self._make_data()
        u_loc = data['u'].mean(dim=0)
        u_std = data['u'].std(dim=0)
        transform = NormalizeIMU(u_loc, u_std)
        out = transform(data)
        assert 'u_normalized' in out
        u_n = out['u_normalized']
        assert torch.allclose(u_n.mean(dim=0), torch.zeros(6).float(), atol=0.1)

    def test_random_subsequence_sampler(self):
        data = self._make_data(N=500)
        transform = RandomSubsequenceSampler(seq_dim=100)
        out = transform(data)
        assert out['u'].shape[0] == 100
        assert out['t'].shape[0] == 100
        assert out['p_gt'].shape[0] == 100
        assert 'N0' in out

    def test_random_subsequence_sampler_no_trim(self):
        data = self._make_data(N=50)
        transform = RandomSubsequenceSampler(seq_dim=100)
        out = transform(data)
        assert out['u'].shape[0] == 50  # shorter than seq_dim, no trim

    def test_random_subsequence_sampler_none(self):
        data = self._make_data(N=200)
        transform = RandomSubsequenceSampler(seq_dim=None)
        out = transform(data)
        assert out['u'].shape[0] == 200

    def test_to_tensor_numpy(self):
        data = {
            't': np.linspace(0, 1, 50),
            'u': np.random.randn(50, 6),
            'ang_gt': np.random.randn(50, 3),
            'p_gt': np.random.randn(50, 3),
            'v_gt': np.random.randn(50, 3),
        }
        transform = ToTensor(dtype=torch.float32)
        out = transform(data)
        assert isinstance(out['t'], torch.Tensor)
        assert out['u'].dtype == torch.float32

    def test_to_tensor_already_tensor(self):
        data = self._make_data()
        transform = ToTensor(dtype=torch.float32)
        out = transform(data)
        assert out['u'].dtype == torch.float32

    def test_compose(self):
        data = self._make_data(N=500)
        transform = Compose([
            RandomSubsequenceSampler(seq_dim=100),
            AddIMUNoise(sigma_gyro=0.01, sigma_acc=0.01),
        ])
        out = transform(data)
        assert out['u'].shape[0] == 100


# ==================== KITTIDataset Tests ====================

class TestKITTIDatasetParsing:
    def test_pose_from_oxts_packet(self):
        """Test Mercator projection computation."""
        packet = KITTIDataset.OxtsPacket(
            lat=49.0, lon=8.0, alt=100.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            vn=0, ve=0, vf=0, vl=0, vu=0,
            ax=0, ay=0, az=0, af=0, al=0, au=0,
            wx=0, wy=0, wz=0, wf=0, wl=0, wu=0,
            pos_accuracy=0, vel_accuracy=0,
            navstat=0, numsats=0, posmode=0, velmode=0, orimode=0,
        )
        scale = np.cos(49.0 * np.pi / 180.)
        R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        # Identity rotation at zero roll/pitch/yaw
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_transform_from_rot_trans(self):
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = KITTIDataset.transform_from_rot_trans(R, t)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], R)
        assert np.allclose(T[:3, 3], t)
        assert np.allclose(T[3], [0, 0, 0, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
