"""Tests for neural network model components (Phase 2)."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import get_model, list_models, register_model
from src.models.base_covariance_net import BaseCovarianceNet
from src.models.init_process_cov_net import InitProcessCovNet
from src.models.measurement_cov_net import MeasurementCovNet
from src.models.learned_bias_correction_net import LearnedBiasCorrectionNet
from src.core.torch_iekf import TorchIEKF


# ==================== Model Registry Tests ====================


class TestModelRegistry:
    def test_list_models_returns_all(self):
        names = list_models()
        assert "InitProcessCovNet" in names
        assert "MeasurementCovNet" in names
        assert "LearnedBiasCorrectionNet" in names

    def test_get_model_returns_class(self):
        cls = get_model("MeasurementCovNet")
        assert cls is MeasurementCovNet

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_model("NonExistentModel")

    def test_register_custom_model(self):
        class DummyNet(BaseCovarianceNet):
            def forward(self, u, iekf):
                return u

            def get_output_dim(self):
                return 2

        register_model(DummyNet)
        assert get_model("DummyNet") is DummyNet


# ==================== InitProcessCovNet Tests ====================


class TestInitProcessCovNet:
    def setup_method(self):
        self.net = InitProcessCovNet(
            output_dim=6, initial_beta=3.0, weight_scale=10.0
        )
        self.iekf = TorchIEKF()

    def test_init_cov_output_shape(self):
        beta = self.net.init_cov(self.iekf)
        assert beta.shape == (6,)

    def test_init_cov_positive(self):
        beta = self.net.init_cov(self.iekf)
        assert (beta > 0).all()

    def test_init_processcov_output_shape(self):
        beta = self.net.init_processcov(self.iekf)
        assert beta.shape == (6,)

    def test_init_processcov_positive(self):
        beta = self.net.init_processcov(self.iekf)
        assert (beta > 0).all()

    def test_get_output_dim(self):
        assert self.net.get_output_dim() == 6


# ==================== MeasurementCovNet Tests ====================


class TestMeasurementCovNet:
    def setup_method(self):
        self.net = MeasurementCovNet(
            input_channels=6,
            output_dim=2,
            cnn_channels=32,
            kernel_size=5,
            dilation=3,
            dropout=0.5,
            initial_beta=3.0,
        )
        self.iekf = TorchIEKF()

    def test_forward_shape(self):
        seq_len = 100
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.shape == (seq_len, 2)

    def test_forward_positive(self):
        seq_len = 50
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert (out > 0).all()

    def test_get_output_dim(self):
        assert self.net.get_output_dim() == 2

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.net.parameters())
        assert n_params > 0
        assert n_params < 20000  # Should be lightweight


# ==================== LearnedBiasCorrectionNet Tests ====================


class TestLearnedBiasCorrectionNet:
    def setup_method(self):
        self.net = LearnedBiasCorrectionNet(
            input_channels=6,
            output_dim=3,
            max_correction=0.5,
            cnn_channels=32,
            kernel_size=5,
            dilation=3,
            dropout=0.5,
        )
        self.iekf = TorchIEKF()

    def test_forward_shape(self):
        seq_len = 100
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.shape == (seq_len, 3)

    def test_forward_bounded(self):
        seq_len = 50
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.abs().max() <= 0.5 + 1e-6

    def test_near_zero_init(self):
        """Output should be near zero at initialization."""
        seq_len = 50
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.abs().max() < 0.05

    def test_get_output_dim(self):
        assert self.net.get_output_dim() == 3

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.net.parameters())
        assert n > 0
        assert n < 20000  # Should be lightweight


# ==================== TorchIEKF Network Integration Tests ====================


class TestTorchIEKFIntegration:
    def test_build_from_cfg_empty(self):
        """Build with no networks enabled."""
        cfg = {"networks": {}}
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.initprocesscov_net is None
        assert iekf.mes_net is None
        assert iekf.bias_correction_net is None

    def test_build_from_cfg_measurement_only(self):
        cfg = {
            "networks": {
                "measurement_cov": {
                    "enabled": True,
                    "type": "MeasurementCovNet",
                    "architecture": {
                        "input_channels": 6,
                        "output_dim": 2,
                    },
                }
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.mes_net is not None
        assert isinstance(iekf.mes_net, MeasurementCovNet)
        assert iekf.initprocesscov_net is None
        assert iekf.bias_correction_net is None

    def test_build_from_cfg_all_networks(self):
        cfg = {
            "networks": {
                "init_process_cov": {
                    "enabled": True,
                    "type": "InitProcessCovNet",
                    "architecture": {"output_dim": 6},
                },
                "measurement_cov": {
                    "enabled": True,
                    "type": "MeasurementCovNet",
                    "architecture": {
                        "input_channels": 6,
                        "output_dim": 2,
                    },
                },
                "bias_correction": {
                    "enabled": True,
                    "type": "LearnedBiasCorrectionNet",
                    "architecture": {
                        "input_channels": 6,
                        "output_dim": 3,
                        "max_correction": 0.5,
                    },
                },
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert isinstance(iekf.initprocesscov_net, InitProcessCovNet)
        assert isinstance(iekf.mes_net, MeasurementCovNet)
        assert isinstance(iekf.bias_correction_net, LearnedBiasCorrectionNet)

    def test_build_from_cfg_disabled_network(self):
        cfg = {
            "networks": {
                "bias_correction": {
                    "enabled": False,
                    "type": "LearnedBiasCorrectionNet",
                    "architecture": {},
                }
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.bias_correction_net is None

    def test_forward_nets_with_mesnet(self):
        iekf = TorchIEKF()
        iekf.mes_net = MeasurementCovNet(input_channels=6, output_dim=2)
        iekf.u_loc = torch.zeros(6).double()
        iekf.u_std = torch.ones(6).double()

        u = torch.randn(50, 6).double()
        covs = iekf.forward_nets(u)
        assert covs.shape == (50, 2)
        assert (covs > 0).all()

    def test_forward_nets_without_mesnet(self):
        iekf = TorchIEKF()
        u = torch.randn(50, 6).double()
        covs = iekf.forward_nets(u)
        assert covs.shape == (50, 2)

    def test_propagate_with_bias_correction(self):
        iekf = TorchIEKF()

        Rot = torch.eye(3).double()
        v = torch.zeros(3).double()
        p = torch.zeros(3).double()
        b_omega = torch.zeros(3).double()
        b_acc = torch.zeros(3).double()
        Rot_c_i = torch.eye(3).double()
        t_c_i = torch.zeros(3).double()
        P = torch.eye(21).double() * 0.01
        u = torch.randn(6).double()
        dt = torch.tensor(0.01).double()
        bias_corr = torch.randn(3).double() * 0.1

        Rot_n, v_n, p_n, *rest = iekf.propagate(
            Rot,
            v,
            p,
            b_omega,
            b_acc,
            Rot_c_i,
            t_c_i,
            P,
            u,
            dt,
            bias_correction=bias_corr,
        )
        assert v_n.shape == (3,)
        assert p_n.shape == (3,)

    def test_propagate_without_bias_correction(self):
        """Classical propagation with no bias correction."""
        iekf = TorchIEKF()

        Rot = torch.eye(3).double()
        v = torch.zeros(3).double()
        p = torch.zeros(3).double()
        b_omega = torch.zeros(3).double()
        b_acc = torch.zeros(3).double()
        Rot_c_i = torch.eye(3).double()
        t_c_i = torch.zeros(3).double()
        P = torch.eye(21).double() * 0.01
        u = torch.randn(6).double()
        dt = torch.tensor(0.01).double()

        Rot_n, v_n, p_n, *rest = iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt
        )
        assert v_n.shape == (3,)
        assert p_n.shape == (3,)

    def test_forward_bias_net(self):
        iekf = TorchIEKF()
        iekf.bias_correction_net = LearnedBiasCorrectionNet(
            input_channels=6,
            output_dim=3,
            max_correction=0.5,
        )
        iekf.u_loc = torch.zeros(6).double()
        iekf.u_std = torch.ones(6).double()

        u = torch.randn(50, 6).double()
        corrections = iekf.forward_bias_net(u)
        assert corrections.shape == (50, 3)
        assert corrections.abs().max() <= 0.5 + 1e-6

    def test_forward_bias_net_none(self):
        iekf = TorchIEKF()
        u = torch.randn(50, 6).double()
        assert iekf.forward_bias_net(u) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
