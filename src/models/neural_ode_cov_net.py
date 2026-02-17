"""
Neural ODE covariance prediction networks.

Lightweight alternatives to the default CNN-based MesNet. Neural ODEs
model the covariance dynamics as a continuous-time ODE, which is a
natural fit for IMU data sampled at fixed rates.

Two variants:
    - NeuralODEConvCovNet: Conv feature extractor + Neural ODE dynamics
    - NeuralODELSTMCovNet: LSTM feature extractor + Neural ODE dynamics

Both are significantly lighter than MesNet while capturing temporal
dynamics through the ODE formulation rather than large conv filters.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from src.models.base_covariance_net import BaseCovarianceNet


class ODEFunc(nn.Module):
    """
    Lightweight ODE dynamics function: dh/dt = f(h).

    Uses a small MLP to define the vector field.

    Args:
        hidden_dim: Dimension of the hidden state.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        ).double()

        # Initialize near-identity for stable training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, h):
        return self.net(h)


class NeuralODEConvCovNet(BaseCovarianceNet):
    """
    Neural ODE with lightweight 1D conv feature extractor.

    Architecture:
        1. Small Conv1d extracts local IMU features per timestep
        2. Neural ODE evolves features through continuous dynamics
        3. Linear head projects to covariance outputs

    Much lighter than MesNet: ~3.5K params vs ~4.5K for MesNet.

    Args:
        input_channels: Number of IMU input channels (default: 6)
        output_dim: Number of covariance outputs (default: 2)
        hidden_dim: ODE hidden state dimension (default: 16)
        conv_channels: Conv feature channels (default: 16)
        kernel_size: Conv kernel size (default: 3)
        initial_beta: Beta scaling parameter (default: 3.0)
        ode_steps: Number of ODE integration steps (default: 2)
        solver: ODE solver method (default: "euler")
    """

    def __init__(
        self,
        input_channels=6,
        output_dim=2,
        hidden_dim=16,
        conv_channels=16,
        kernel_size=3,
        initial_beta=3.0,
        ode_steps=2,
        solver="euler",
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.solver = solver

        self.register_buffer(
            "beta_measurement",
            initial_beta * torch.ones(output_dim).double(),
        )

        # Lightweight conv feature extractor
        self.feature_net = nn.Sequential(
            nn.Conv1d(
                input_channels,
                conv_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
        ).double()

        # Project conv features to ODE state
        self.to_ode = nn.Linear(conv_channels, hidden_dim).double()

        # ODE dynamics
        self.ode_func = ODEFunc(hidden_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        ).double()

        # Small init for stable output
        self.head[0].weight.data *= 0.01
        self.head[0].bias.data *= 0.01

    def forward(self, u, iekf):
        """
        Predict measurement covariances from normalized IMU input.

        Args:
            u: Normalized IMU input (1, input_channels, seq_len)
            iekf: IEKF filter instance (provides cov0_measurement)

        Returns:
            Measurement covariances (seq_len, output_dim)
        """
        # Extract features: (1, conv_channels, seq_len)
        features = self.feature_net(u)

        # Reshape to (seq_len, conv_channels) and project to ODE state
        h = features.squeeze(0).transpose(0, 1)  # (seq_len, conv_channels)
        h = self.to_ode(h)  # (seq_len, hidden_dim)

        # Run Neural ODE on each timestep's features
        t_span = torch.linspace(
            0, 1, self.ode_steps, dtype=torch.float64, device=h.device
        )
        h_evolved = odeint(self.ode_func, h, t_span, method=self.solver)
        h_out = h_evolved[-1]  # Take final state: (seq_len, hidden_dim)

        # Project to covariance scaling factors
        z_cov = self.head(h_out)  # (seq_len, output_dim)

        # Apply beta scaling and baseline covariance
        z_cov_net = self.beta_measurement.unsqueeze(0) * z_cov
        measurements_covs = iekf.cov0_measurement.unsqueeze(0) * (
            10**z_cov_net
        )

        return measurements_covs

    def get_output_dim(self):
        return self.output_dim


class NeuralODELSTMCovNet(BaseCovarianceNet):
    """
    Neural ODE with lightweight LSTM feature extractor.

    Architecture:
        1. Small single-layer LSTM captures sequential IMU patterns
        2. Neural ODE refines LSTM features through continuous dynamics
        3. Linear head projects to covariance outputs

    The LSTM captures sequential dependencies while the ODE provides
    smooth, continuous-time refinement. ~4K params total.

    Args:
        input_channels: Number of IMU input channels (default: 6)
        output_dim: Number of covariance outputs (default: 2)
        hidden_dim: ODE hidden state dimension (default: 16)
        lstm_hidden: LSTM hidden dimension (default: 16)
        initial_beta: Beta scaling parameter (default: 3.0)
        ode_steps: Number of ODE integration steps (default: 2)
        solver: ODE solver method (default: "euler")
    """

    def __init__(
        self,
        input_channels=6,
        output_dim=2,
        hidden_dim=16,
        lstm_hidden=16,
        initial_beta=3.0,
        ode_steps=2,
        solver="euler",
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.solver = solver

        self.register_buffer(
            "beta_measurement",
            initial_beta * torch.ones(output_dim).double(),
        )

        # Lightweight LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        ).double()

        # Project LSTM output to ODE state
        self.to_ode = nn.Linear(lstm_hidden, hidden_dim).double()

        # ODE dynamics
        self.ode_func = ODEFunc(hidden_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        ).double()

        # Small init for stable output
        self.head[0].weight.data *= 0.01
        self.head[0].bias.data *= 0.01

    def forward(self, u, iekf):
        """
        Predict measurement covariances from normalized IMU input.

        Args:
            u: Normalized IMU input (1, input_channels, seq_len)
            iekf: IEKF filter instance (provides cov0_measurement)

        Returns:
            Measurement covariances (seq_len, output_dim)
        """
        # Reshape for LSTM: (1, seq_len, input_channels)
        x = u.transpose(1, 2)

        # LSTM forward: (1, seq_len, lstm_hidden)
        lstm_out, _ = self.lstm(x)

        # Project to ODE state: (seq_len, hidden_dim)
        h = self.to_ode(lstm_out.squeeze(0))

        # Run Neural ODE
        t_span = torch.linspace(
            0, 1, self.ode_steps, dtype=torch.float64, device=h.device
        )
        h_evolved = odeint(self.ode_func, h, t_span, method=self.solver)
        h_out = h_evolved[-1]  # (seq_len, hidden_dim)

        # Project to covariance scaling factors
        z_cov = self.head(h_out)

        # Apply beta scaling and baseline covariance
        z_cov_net = self.beta_measurement.unsqueeze(0) * z_cov
        measurements_covs = iekf.cov0_measurement.unsqueeze(0) * (
            10**z_cov_net
        )

        return measurements_covs

    def get_output_dim(self):
        return self.output_dim
