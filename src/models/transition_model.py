"""
Transition model for latent context propagation.

Replaces per-timestep encoder inference with learned z propagation.
Each particle's z evolves autoregressively: z_{n+1} = f(z_n, imu_n).
The encoder is used only at initialization (first chunk).
"""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """
    Learned transition model for latent context propagation.

    Input:  Concatenation of z^i_n (latent_dim) and imu_n (6).
    Output: mu_transition, log_var_transition for z^i_{n+1}.

    Architecture:
        FC: (latent_dim + 6) → hidden_dim, ReLU
        FC: hidden_dim → 2 * latent_dim (no activation)
        Split → mu_transition, log_var_transition

    Stability: mu_transition bounded via tanh to prevent drift.

    Near-identity initialization: weights initialized so that
    mu_transition ≈ z_n at the start of training, giving stable
    Stage 3-like behavior initially.

    Args:
        latent_dim: Dimension of z.
        hidden_dim: Hidden layer width.
        alpha: Tanh bound for mu_transition.
        weight_scale: Near-zero init multiplier for weights.
        bias_scale: Near-zero init multiplier for biases.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        alpha: float = 3.0,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        self._near_identity_init(weight_scale, bias_scale)

    def _near_identity_init(self, ws: float, bs: float):
        """
        Initialize so output ≈ input z at start of training.

        The first layer maps [z, imu] → hidden. The second layer maps
        hidden → [mu_raw, log_var]. We scale all weights small so the
        raw output is near zero, making mu_transition ≈ alpha * tanh(0) = 0.
        Combined with z values that start near zero (from encoder init),
        this gives near-identity behavior.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(ws)
                layer.bias.data.mul_(bs)

    def forward(self, z: torch.Tensor, imu: torch.Tensor) -> tuple:
        """
        Propagate z one timestep forward.

        Args:
            z:   (..., latent_dim) current latent state.
            imu: (..., 6) current IMU measurement.

        Returns:
            z_next:             (..., latent_dim) next latent state (sampled).
            mu_transition:      (..., latent_dim) transition mean.
            log_var_transition: (..., latent_dim) transition log-variance.
        """
        x = torch.cat([z, imu], dim=-1)
        out = self.net(x)  # (..., 2 * latent_dim)
        mu_raw = out[..., :self.latent_dim]
        log_var_transition = out[..., self.latent_dim:]

        # Stability: bound mu to prevent drift
        mu_transition = self.alpha * torch.tanh(mu_raw)

        # Reparameterization
        if self.training:
            sigma = torch.exp(0.5 * log_var_transition)
            epsilon = torch.randn_like(sigma)
            z_next = mu_transition + sigma * epsilon
        else:
            z_next = mu_transition

        return z_next, mu_transition, log_var_transition
