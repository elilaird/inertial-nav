"""
Neural ODE dynamics model that can replace classical inertial kinematics.

Instead of the hand-crafted IMU integration in the IEKF propagation step:
    acc = R @ (a_imu - b_acc) + g
    v_next = v + acc * dt
    p_next = p + v * dt + 0.5 * acc * dt^2
    R_next = R @ exp(omega * dt)

This module learns the continuous-time dynamics:
    d[v, p]/dt = f_theta(v, p, u_imu)

where f_theta is a neural network. The rotation dynamics can either
be learned or kept classical (default: classical rotation, learned v/p).

This is the primary use case for Neural ODEs in inertial navigation:
the ODE formulation naturally models the continuous dynamics between
IMU samples, and the adjoint method enables memory-efficient training.

Usage in IEKF:
    The learned dynamics replaces the propagation step. Covariance
    propagation can still use the classical Jacobian or also be learned.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class IMUDynamicsFunc(nn.Module):
    """
    Neural network defining d[state]/dt = f(state, imu_input).

    The IMU input is held constant over each integration interval
    (zero-order hold, matching the IMU sampling model).

    Args:
        state_dim: Dimension of the state being integrated (default: 6 for v,p)
        imu_dim: IMU input dimension (default: 6 for [gyro, acc])
        hidden_dim: Hidden layer dimension (default: 32)
    """

    def __init__(self, state_dim=6, imu_dim=6, hidden_dim=32):
        super().__init__()
        self.imu_input = None  # Set before each integration call

        self.net = nn.Sequential(
            nn.Linear(state_dim + imu_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        ).double()

        # Near-zero init for residual-like behavior at start of training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def set_imu_input(self, u):
        """Set the IMU input for the current integration interval."""
        self.imu_input = u

    def forward(self, t, state):
        """
        Compute state derivative.

        Args:
            t: Current time (scalar, unused but required by odeint)
            state: Current state [v, p] of shape (6,) or (batch, 6)

        Returns:
            State derivative of same shape as state.
        """
        # Concatenate state with IMU input
        if state.dim() == 1:
            x = torch.cat([state, self.imu_input])
        else:
            u_expanded = self.imu_input.unsqueeze(0).expand(state.shape[0], -1)
            x = torch.cat([state, u_expanded], dim=-1)

        return self.net(x)


class NeuralODEDynamics(nn.Module):
    """
    Learned dynamics model for IEKF propagation.

    Replaces the classical inertial kinematics with a Neural ODE that
    learns continuous-time velocity and position dynamics from data.

    The rotation dynamics are kept classical (SO(3) exponential map)
    since they are well-modeled by gyroscope integration. Only the
    translational dynamics (velocity and position) are learned.

    This gives the best of both worlds:
    - Rotation: exact geometric integration (no drift)
    - Translation: learned dynamics that can compensate for
      accelerometer biases, vibrations, and nonlinearities

    Args:
        imu_dim: IMU input dimension (default: 6)
        hidden_dim: Hidden layer dimension for ODE func (default: 32)
        solver: ODE solver method (default: "euler" for speed, "dopri5" for accuracy)
        use_classical_residual: If True, add classical physics as residual (default: True)
    """

    def __init__(self, imu_dim=6, hidden_dim=32, solver="euler",
                 use_classical_residual=True):
        super().__init__()

        self.solver = solver
        self.use_classical_residual = use_classical_residual

        # State = [v(3), p(3)] = 6-dim
        self.state_dim = 6

        # Neural ODE dynamics
        self.ode_func = IMUDynamicsFunc(
            state_dim=self.state_dim,
            imu_dim=imu_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, v, p, Rot, u, b_acc, g, dt):
        """
        Propagate velocity and position using learned dynamics.

        Args:
            v: Current velocity (3,)
            p: Current position (3,)
            Rot: Current rotation matrix (3, 3) — used for classical residual
            u: IMU measurement (6,) — [gyro(3), acc(3)]
            b_acc: Accelerometer bias (3,)
            g: Gravity vector (3,)
            dt: Time step (scalar)

        Returns:
            Tuple of (v_next, p_next)
        """
        # Pack state
        state = torch.cat([v, p])  # (6,)

        # Set IMU input for ODE func
        self.ode_func.set_imu_input(u)

        # Integrate ODE from t=0 to t=dt
        t_span = torch.tensor([0.0, dt.item()]).double()
        state_trajectory = odeint(
            self.ode_func, state, t_span, method=self.solver
        )
        state_next = state_trajectory[-1]  # (6,)

        v_next = state_next[:3]
        p_next = state_next[3:]

        # Optionally add classical physics as residual
        if self.use_classical_residual:
            acc_classical = Rot.mv(u[3:6] - b_acc) + g
            v_next = v_next + acc_classical * dt
            p_next = p_next + v.clone() * dt + 0.5 * acc_classical * dt ** 2

        return v_next, p_next

    def count_parameters(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
