"""Tests for neural network model components (Phase 2)."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model, list_models, register_model
from src.models.base_covariance_net import BaseCovarianceNet
from src.models.init_process_cov_net import InitProcessCovNet
from src.models.measurement_cov_net import MeasurementCovNet
from src.models.neural_ode_cov_net import NeuralODEConvCovNet, NeuralODELSTMCovNet
from src.models.neural_ode_dynamics import NeuralODEDynamics, IMUDynamicsFunc
from src.core.torch_iekf import TorchIEKF


# ==================== Model Registry Tests ====================

class TestModelRegistry:
    def test_list_models_returns_all(self):
        names = list_models()
        assert "InitProcessCovNet" in names
        assert "MeasurementCovNet" in names
        assert "NeuralODEConvCovNet" in names
        assert "NeuralODELSTMCovNet" in names
        assert "NeuralODEDynamics" in names

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
        self.net = InitProcessCovNet(output_dim=6, initial_beta=3.0, weight_scale=10.0)
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
            input_channels=6, output_dim=2,
            cnn_channels=32, kernel_size=5,
            dilation=3, dropout=0.5, initial_beta=3.0
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


# ==================== NeuralODEConvCovNet Tests ====================

class TestNeuralODEConvCovNet:
    def setup_method(self):
        self.net = NeuralODEConvCovNet(
            input_channels=6, output_dim=2, hidden_dim=16,
            conv_channels=16, kernel_size=3, initial_beta=3.0,
            ode_steps=2, solver="euler"
        )
        self.iekf = TorchIEKF()

    def test_forward_shape(self):
        seq_len = 50
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.shape == (seq_len, 2)

    def test_forward_positive(self):
        seq_len = 30
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert (out > 0).all()

    def test_lighter_than_mesnet(self):
        mesnet = MeasurementCovNet()
        n_node = sum(p.numel() for p in self.net.parameters())
        n_mes = sum(p.numel() for p in mesnet.parameters())
        assert n_node < n_mes

    def test_get_output_dim(self):
        assert self.net.get_output_dim() == 2


# ==================== NeuralODELSTMCovNet Tests ====================

class TestNeuralODELSTMCovNet:
    def setup_method(self):
        self.net = NeuralODELSTMCovNet(
            input_channels=6, output_dim=2, hidden_dim=16,
            lstm_hidden=16, initial_beta=3.0,
            ode_steps=2, solver="euler"
        )
        self.iekf = TorchIEKF()

    def test_forward_shape(self):
        seq_len = 50
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert out.shape == (seq_len, 2)

    def test_forward_positive(self):
        seq_len = 30
        u = torch.randn(1, 6, seq_len).double()
        out = self.net(u, self.iekf)
        assert (out > 0).all()

    def test_lighter_than_mesnet(self):
        mesnet = MeasurementCovNet()
        n_node = sum(p.numel() for p in self.net.parameters())
        n_mes = sum(p.numel() for p in mesnet.parameters())
        assert n_node < n_mes

    def test_get_output_dim(self):
        assert self.net.get_output_dim() == 2


# ==================== NeuralODEDynamics Tests ====================

class TestIMUDynamicsFunc:
    def setup_method(self):
        self.func = IMUDynamicsFunc(state_dim=6, imu_dim=6, hidden_dim=32)

    def test_forward_1d(self):
        state = torch.randn(6).double()
        self.func.set_imu_input(torch.randn(6).double())
        out = self.func(torch.tensor(0.0), state)
        assert out.shape == (6,)

    def test_forward_batch(self):
        state = torch.randn(4, 6).double()
        self.func.set_imu_input(torch.randn(6).double())
        out = self.func(torch.tensor(0.0), state)
        assert out.shape == (4, 6)

    def test_near_zero_init(self):
        """Output should be near zero at initialization (residual-like)."""
        state = torch.randn(6).double()
        self.func.set_imu_input(torch.randn(6).double())
        out = self.func(torch.tensor(0.0), state)
        assert out.abs().max() < 1.0


class TestNeuralODEDynamics:
    def setup_method(self):
        self.model = NeuralODEDynamics(
            imu_dim=6, hidden_dim=32, solver="euler",
            use_classical_residual=True
        )

    def test_forward_shape(self):
        v = torch.randn(3).double()
        p = torch.randn(3).double()
        Rot = torch.eye(3).double()
        u = torch.randn(6).double()
        b_acc = torch.zeros(3).double()
        g = torch.tensor([0., 0., -9.81]).double()
        dt = torch.tensor(0.01).double()

        v_next, p_next = self.model(v, p, Rot, u, b_acc, g, dt)
        assert v_next.shape == (3,)
        assert p_next.shape == (3,)

    def test_forward_no_residual(self):
        model = NeuralODEDynamics(use_classical_residual=False)
        v = torch.zeros(3).double()
        p = torch.zeros(3).double()
        Rot = torch.eye(3).double()
        u = torch.zeros(6).double()
        b_acc = torch.zeros(3).double()
        g = torch.tensor([0., 0., -9.81]).double()
        dt = torch.tensor(0.01).double()

        v_next, p_next = model(v, p, Rot, u, b_acc, g, dt)
        assert v_next.shape == (3,)
        assert p_next.shape == (3,)

    def test_gradients_flow(self):
        v = torch.randn(3).double().requires_grad_(True)
        p = torch.randn(3).double()
        Rot = torch.eye(3).double()
        u = torch.randn(6).double()
        b_acc = torch.zeros(3).double()
        g = torch.tensor([0., 0., -9.81]).double()
        dt = torch.tensor(0.01).double()

        v_next, p_next = self.model(v, p, Rot, u, b_acc, g, dt)
        loss = v_next.sum() + p_next.sum()
        loss.backward()
        assert v.grad is not None

    def test_count_parameters(self):
        n = self.model.count_parameters()
        assert n > 0
        assert n < 10000  # Should be lightweight


# ==================== TorchIEKF Network Integration Tests ====================

class TestTorchIEKFIntegration:
    def test_build_from_cfg_empty(self):
        """Build with no networks enabled."""
        cfg = {"networks": {}}
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.initprocesscov_net is None
        assert iekf.mes_net is None
        assert iekf.dynamics_net is None

    def test_build_from_cfg_measurement_only(self):
        cfg = {
            "networks": {
                "measurement_cov": {
                    "enabled": True,
                    "type": "MeasurementCovNet",
                    "architecture": {
                        "input_channels": 6,
                        "output_dim": 2,
                    }
                }
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.mes_net is not None
        assert isinstance(iekf.mes_net, MeasurementCovNet)
        assert iekf.initprocesscov_net is None
        assert iekf.dynamics_net is None

    def test_build_from_cfg_all_networks(self):
        cfg = {
            "networks": {
                "init_process_cov": {
                    "enabled": True,
                    "type": "InitProcessCovNet",
                    "architecture": {"output_dim": 6}
                },
                "measurement_cov": {
                    "enabled": True,
                    "type": "NeuralODEConvCovNet",
                    "architecture": {
                        "input_channels": 6, "output_dim": 2,
                        "hidden_dim": 16, "conv_channels": 16
                    }
                },
                "dynamics": {
                    "enabled": True,
                    "type": "NeuralODEDynamics",
                    "architecture": {
                        "imu_dim": 6, "hidden_dim": 32, "solver": "euler"
                    }
                }
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert isinstance(iekf.initprocesscov_net, InitProcessCovNet)
        assert isinstance(iekf.mes_net, NeuralODEConvCovNet)
        assert isinstance(iekf.dynamics_net, NeuralODEDynamics)

    def test_build_from_cfg_disabled_network(self):
        cfg = {
            "networks": {
                "dynamics": {
                    "enabled": False,
                    "type": "NeuralODEDynamics",
                    "architecture": {}
                }
            }
        }
        iekf = TorchIEKF.build_from_cfg(cfg)
        assert iekf.dynamics_net is None

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

    def test_propagate_with_dynamics_net(self):
        iekf = TorchIEKF()
        iekf.dynamics_net = NeuralODEDynamics(
            imu_dim=6, hidden_dim=32, solver="euler"
        )

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

    def test_propagate_without_dynamics_net(self):
        """Classical propagation when no dynamics net is set."""
        iekf = TorchIEKF()
        assert iekf.dynamics_net is None

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
