"""
PyTorch implementation of Invariant Extended Kalman Filter (IEKF).

This is the single IEKF implementation used for both training (with autograd)
and inference (with torch.no_grad).  All computation is done in PyTorch which
can target CPU or CUDA.
"""

import torch
import os
from termcolor import cprint

from src.models import get_model


def isclose(mat1, mat2, tol=1e-10):
    """Check if two tensors are close within tolerance."""
    return (mat1 - mat2).abs().lt(tol)


class TorchIEKF(torch.nn.Module):
    """
    PyTorch IEKF with optional learned covariance networks.

    Supports automatic differentiation for training neural networks that
    predict adaptive measurement and process noise covariances.

    State vector: [Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i]
    Covariance is 21-dimensional on the manifold tangent space.
    """

    # ------------------------------------------------------------------
    # Default filter parameters
    # ------------------------------------------------------------------
    class Parameters:
        """Default IEKF filter parameters."""

        g = [0, 0, -9.80665]
        """Gravity vector (m/s^2)."""

        P_dim = 21
        """Covariance dimension."""

        Q_dim = 18
        """Process noise covariance dimension."""

        # Process noise covariances
        cov_omega = 1e-3
        cov_acc = 1e-2
        cov_b_omega = 6e-9
        cov_b_acc = 2e-4
        cov_Rot_c_i = 1e-9
        cov_t_c_i = 1e-9

        # Measurement noise covariances
        cov_lat = 0.2
        cov_up = 300

        # Initial state covariances
        cov_Rot0 = 1e-3
        cov_b_omega0 = 6e-3
        cov_b_acc0 = 4e-3
        cov_v0 = 1e-1
        cov_Rot_c_i0 = 1e-6
        cov_t_c_i0 = 5e-3

        # Numerical parameters
        n_normalize_rot = 100
        n_normalize_rot_c_i = 1000
        verbose = False

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parameter_class=None):
        """
        Initialize PyTorch IEKF.

        Args:
            parameter_class: Parameter class (default: TorchIEKF.Parameters)
        """
        super().__init__()

        if parameter_class is None:
            params = self.Parameters()
        else:
            params = parameter_class()

        # Copy scalar parameters as plain attributes
        self._copy_scalar_params(params)

        # Register tensor-valued parameters as buffers
        self.register_buffer("g", torch.tensor(params.g, dtype=torch.float32))
        self.register_buffer(
            "cov0_measurement",
            torch.tensor([params.cov_lat, params.cov_up], dtype=torch.float32),
        )

        # Identity matrix buffers (moved automatically by model.to(device))
        self.register_buffer("Id2", torch.eye(2, dtype=torch.float32))
        self.register_buffer("Id3", torch.eye(3, dtype=torch.float32))
        self.register_buffer("Id6", torch.eye(6, dtype=torch.float32))
        self.register_buffer("IdP", torch.eye(self.P_dim, dtype=torch.float32))

        # Process noise covariance buffer
        self._register_Q()

        # Input normalization parameters (set via get_normalize_u)
        self.register_buffer("u_loc", None)
        self.register_buffer("u_std", None)

        # Optional neural network components
        self.initprocesscov_net = None
        self.mes_net = None
        self.bias_correction_net = None
        self.world_model = None
        self.transition_model = None

    def _copy_scalar_params(self, params):
        """Copy non-tensor, non-callable attributes from *params* to self."""
        skip = {"g"}  # handled as buffers
        for name in dir(params):
            if name.startswith("_") or name in skip:
                continue
            val = getattr(params, name)
            if callable(val):
                continue
            setattr(self, name, val)

    def _register_Q(self):
        """Build and register the process noise covariance matrix."""
        q_vals = torch.tensor(
            [
                self.cov_omega,
                self.cov_omega,
                self.cov_omega,
                self.cov_acc,
                self.cov_acc,
                self.cov_acc,
                self.cov_b_omega,
                self.cov_b_omega,
                self.cov_b_omega,
                self.cov_b_acc,
                self.cov_b_acc,
                self.cov_b_acc,
                self.cov_Rot_c_i,
                self.cov_Rot_c_i,
                self.cov_Rot_c_i,
                self.cov_t_c_i,
                self.cov_t_c_i,
                self.cov_t_c_i,
            ],
            dtype=torch.float32,
        )
        self.register_buffer("Q", torch.diag(q_vals))

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the model lives on (inferred from registered buffers)."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_cfg(cls, cfg, parameter_class=None):
        """
        Build TorchIEKF with networks from a Hydra config.

        Args:
            cfg: OmegaConf config with 'networks' key containing network specs.
            parameter_class: Optional parameter class override.

        Returns:
            Configured TorchIEKF instance with networks attached.
        """
        iekf = cls(parameter_class=parameter_class)

        networks_cfg = cfg.get("networks", {})

        # Default type names when config omits an explicit 'type' key
        _DEFAULT_TYPES = {
            "init_process_cov": "InitProcessCovNet",
            "measurement_cov": "MeasurementCovNet",
            "bias_correction": "LearnedBiasCorrectionNet",
            "world_model": "LatentWorldModel",
        }

        for key, attr in [
            ("init_process_cov", "initprocesscov_net"),
            ("measurement_cov", "mes_net"),
            ("bias_correction", "bias_correction_net"),
            ("world_model", "world_model"),
        ]:
            net_cfg = networks_cfg.get(key, {})
            if net_cfg.get("enabled", False):
                type_name = net_cfg.get("type", _DEFAULT_TYPES[key])
                net = cls._build_network(
                    type_name, net_cfg.get("architecture", {})
                )
                setattr(iekf, attr, net)

        # Transition model (optional, for Stage 4 particle filter)
        tm_cfg = cfg.get("transition_model", {})
        if tm_cfg.get("enabled", False):
            type_name = tm_cfg.get("type", "TransitionModel")
            net = cls._build_network(
                type_name, tm_cfg.get("architecture", {})
            )
            iekf.transition_model = net

        return iekf

    @staticmethod
    def _build_network(type_name, architecture_cfg):
        """
        Instantiate a network from the model registry.

        Args:
            type_name: Registered model class name (e.g., "MeasurementCovNet")
            architecture_cfg: Dict/DictConfig of constructor kwargs.

        Returns:
            Instantiated network module.
        """
        import inspect
        import warnings

        model_cls = get_model(type_name)
        kwargs = dict(architecture_cfg) if architecture_cfg else {}

        # Filter kwargs to only those accepted by the constructor,
        # so extra config keys (e.g. 'hidden_features') don't cause errors.
        sig = inspect.signature(model_cls.__init__)
        params = sig.parameters
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            valid_keys = set(params.keys()) - {"self"}
            dropped = set(kwargs.keys()) - valid_keys
            if dropped:
                warnings.warn(
                    f"_build_network({type_name}): config keys {sorted(dropped)} "
                    f"are not accepted by the constructor and will be ignored. "
                    f"Valid keys: {sorted(valid_keys)}",
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return model_cls(**kwargs)

    # ------------------------------------------------------------------
    # Filter loop
    # ------------------------------------------------------------------

    def run(
        self,
        t,
        u,
        measurements_covs,
        v_mes,
        p_mes,
        N,
        ang0,
        bias_corrections=None,
        gyro_corrections=None,
        process_noise_scaling=None,
        bias_noise_scaling=None,
    ):
        """
        Run the filter on a sequence of IMU measurements.

        Args:
            t: Timestamps (N,) - PyTorch tensor
            u: IMU measurements (N, 6) - PyTorch tensor
            measurements_covs: Measurement covariances (N, 2) - PyTorch tensor
            v_mes: Ground truth velocities (N, 3) - PyTorch tensor
            p_mes: Ground truth positions (N, 3) - PyTorch tensor
            N: Number of timesteps (None = all)
            ang0: Initial orientation [roll, pitch, yaw] (3,)
            bias_corrections: Optional per-timestep accelerometer bias
                corrections (N, 3) from LearnedBiasCorrectionNet.
            gyro_corrections: Optional per-timestep gyroscope bias
                corrections (N, 3) from WorldModel.
            process_noise_scaling: Optional Q scaling factors — either
                (N, 6) per-timestep or (1, 6) per-chunk.
            bias_noise_scaling: Optional per-timestep per-axis bias noise
                scaling (N, 3) applied to both b_omega and b_acc Q diagonals.

        Returns:
            Tuple of (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
        """
        dt = t[1:] - t[:-1]
        if N is None:
            N = u.shape[0]

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(
            dt, u, p_mes, v_mes, N, ang0
        )

        for i in range(1, N):
            bc_i = (
                bias_corrections[i] if bias_corrections is not None else None
            )
            gc_i = (
                gyro_corrections[i] if gyro_corrections is not None else None
            )
            # Per-timestep or per-chunk Q scaling
            if process_noise_scaling is not None:
                if process_noise_scaling.shape[0] == 1:
                    pns_i = process_noise_scaling[0]
                else:
                    pns_i = process_noise_scaling[i]
            else:
                pns_i = None
            bns_i = bias_noise_scaling[i] if bias_noise_scaling is not None else None

            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = (
                self.propagate(
                    Rot[i - 1],
                    v[i - 1],
                    p[i - 1],
                    b_omega[i - 1],
                    b_acc[i - 1],
                    Rot_c_i[i - 1],
                    t_c_i[i - 1],
                    P,
                    u[i],
                    dt[i - 1],
                    bias_correction=bc_i,
                    gyro_correction=gc_i,
                    process_noise_scaling=pns_i,
                    bias_noise_scaling=bns_i,
                )
            )

            (
                Rot[i],
                v[i],
                p[i],
                b_omega[i],
                b_acc[i],
                Rot_c_i[i],
                t_c_i[i],
                P,
            ) = self.update(
                Rot_i,
                v_i,
                p_i,
                b_omega_i,
                b_acc_i,
                Rot_c_i_i,
                t_c_i_i,
                P_i,
                u[i],
                i,
                measurements_covs[i],
            )

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_state(self, t, u, v_mes, ang0):
        """
        Build the initial filter state for BPTT chunk-wise execution.

        Unlike ``init_run``, this returns a plain state tuple rather than
        pre-allocated trajectory arrays, so it can be threaded between chunks.

        Args:
            t:      Timestamps for the full sequence (used to get dtype/device).
            u:      IMU measurements (for device reference).
            v_mes:  Ground-truth velocities (N, 3); v_mes[0] used for init.
            ang0:   Initial Euler angles [roll, pitch, yaw].

        Returns:
            State dict with keys: Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P.
        """
        P = self.init_covariance()
        Rot0 = self.from_rpy_torch(ang0[0], ang0[1], ang0[2])
        v0 = v_mes[0].clone()
        p0 = t.new_zeros(3).float()
        b_omega0 = t.new_zeros(3).float()
        b_acc0 = t.new_zeros(3).float()
        Rot_c_i0 = torch.eye(3).float()
        t_c_i0 = t.new_zeros(3).float()
        return dict(
            Rot=Rot0,
            v=v0,
            p=p0,
            b_omega=b_omega0,
            b_acc=b_acc0,
            Rot_c_i=Rot_c_i0,
            t_c_i=t_c_i0,
            P=P,
        )

    @staticmethod
    def detach_state(state):
        """Detach all state tensors from the autograd graph (for BPTT)."""
        return {k: v.detach() for k, v in state.items()}

    def run_chunk(
        self,
        state,
        t_chunk,
        u_chunk,
        measurements_covs_chunk,
        bias_corrections_chunk=None,
        gyro_corrections_chunk=None,
        process_noise_scaling_chunk=None,
        bias_noise_scaling_chunk=None,
    ):
        """
        Run the filter for one chunk of timesteps, starting from ``state``.

        Used for Truncated BPTT: caller detaches ``state`` between chunks
        to limit gradient flow to ``chunk_size`` timesteps.

        Args:
            state:               State dict from ``init_state`` or previous chunk.
            t_chunk:             Timestamps for this chunk (K,).
            u_chunk:             IMU measurements for this chunk (K, 6).
            measurements_covs_chunk: Measurement covariances for this chunk (K, 2).
            bias_corrections_chunk: Optional per-timestep accelerometer bias
                corrections (K, 3).
            gyro_corrections_chunk: Optional per-timestep gyroscope bias
                corrections (K, 3).
            process_noise_scaling_chunk: Optional Q scaling — (K, 6) or (1, 6).
            bias_noise_scaling_chunk: Optional per-timestep per-axis bias noise
                scaling (K, 3) applied to both b_omega and b_acc Q diagonals.

        Returns:
            traj:       Tuple (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i),
                        each of shape (K, ...).  Positions are *absolute*
                        (continuing from whatever ``state`` held).
            new_state:  State dict at the end of the chunk (still on graph).
        """
        K = t_chunk.shape[0]
        dt_chunk = t_chunk[1:] - t_chunk[:-1]  # (K-1,)

        # Allocate chunk trajectory tensors
        Rot = t_chunk.new_zeros(K, 3, 3).float()
        v = t_chunk.new_zeros(K, 3).float()
        p = t_chunk.new_zeros(K, 3).float()
        b_omega = t_chunk.new_zeros(K, 3).float()
        b_acc = t_chunk.new_zeros(K, 3).float()
        Rot_c_i = t_chunk.new_zeros(K, 3, 3).float()
        t_c_i = t_chunk.new_zeros(K, 3).float()

        # Seed first timestep from incoming state
        Rot[0] = state["Rot"]
        v[0] = state["v"]
        p[0] = state["p"]
        b_omega[0] = state["b_omega"]
        b_acc[0] = state["b_acc"]
        Rot_c_i[0] = state["Rot_c_i"]
        t_c_i[0] = state["t_c_i"]
        P = state["P"]

        for i in range(1, K):
            bc_i = (
                bias_corrections_chunk[i]
                if bias_corrections_chunk is not None
                else None
            )
            gc_i = (
                gyro_corrections_chunk[i]
                if gyro_corrections_chunk is not None
                else None
            )
            if process_noise_scaling_chunk is not None:
                if process_noise_scaling_chunk.shape[0] == 1:
                    pns_i = process_noise_scaling_chunk[0]
                else:
                    pns_i = process_noise_scaling_chunk[i]
            else:
                pns_i = None
            bns_i = (
                bias_noise_scaling_chunk[i]
                if bias_noise_scaling_chunk is not None
                else None
            )

            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = (
                self.propagate(
                    Rot[i - 1],
                    v[i - 1],
                    p[i - 1],
                    b_omega[i - 1],
                    b_acc[i - 1],
                    Rot_c_i[i - 1],
                    t_c_i[i - 1],
                    P,
                    u_chunk[i],
                    dt_chunk[i - 1],
                    bias_correction=bc_i,
                    gyro_correction=gc_i,
                    process_noise_scaling=pns_i,
                    bias_noise_scaling=bns_i,
                )
            )
            (
                Rot[i],
                v[i],
                p[i],
                b_omega[i],
                b_acc[i],
                Rot_c_i[i],
                t_c_i[i],
                P,
            ) = self.update(
                Rot_i,
                v_i,
                p_i,
                b_omega_i,
                b_acc_i,
                Rot_c_i_i,
                t_c_i_i,
                P_i,
                u_chunk[i],
                i,
                measurements_covs_chunk[i],
            )

        traj = (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
        new_state = dict(
            Rot=Rot[-1],
            v=v[-1],
            p=p[-1],
            b_omega=b_omega[-1],
            b_acc=b_acc[-1],
            Rot_c_i=Rot_c_i[-1],
            t_c_i=t_c_i[-1],
            P=P,
        )
        return traj, new_state

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
        """Initialize filter state arrays and covariance."""
        Rot = dt.new_zeros(N, 3, 3)
        v = dt.new_zeros(N, 3)
        p = dt.new_zeros(N, 3)
        b_omega = dt.new_zeros(N, 3)
        b_acc = dt.new_zeros(N, 3)
        Rot_c_i = dt.new_zeros(N, 3, 3)
        t_c_i = dt.new_zeros(N, 3)

        Rot_c_i[0] = self.Id3
        Rot[0] = self.from_rpy_torch(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]

        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        """
        Initialize state covariance matrix.

        Uses learned scaling factors when ``initprocesscov_net`` is set.
        """
        P = self.IdP.new_zeros(self.P_dim, self.P_dim)

        if self.initprocesscov_net is not None:
            beta = self.initprocesscov_net.init_cov(self)
        else:
            beta = self.IdP.new_ones(6)

        P[:2, :2] = self.cov_Rot0 * beta[0] * self.Id2
        P[3:5, 3:5] = self.cov_v0 * beta[1] * self.Id2
        P[9:12, 9:12] = self.cov_b_omega0 * beta[2] * self.Id3
        P[12:15, 12:15] = self.cov_b_acc0 * beta[3] * self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0 * beta[4] * self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0 * beta[5] * self.Id3
        return P

    # ------------------------------------------------------------------
    # Propagation (prediction)
    # ------------------------------------------------------------------

    def propagate(
        self,
        Rot_prev,
        v_prev,
        p_prev,
        b_omega_prev,
        b_acc_prev,
        Rot_c_i_prev,
        t_c_i_prev,
        P_prev,
        u,
        dt,
        bias_correction=None,
        gyro_correction=None,
        process_noise_scaling=None,
        bias_noise_scaling=None,
    ):
        """
        Propagate state forward one timestep using classical inertial
        kinematics, optionally with learned corrections.

        Args:
            bias_correction: Optional (3,) — additive Δb_acc(t).
            gyro_correction: Optional (3,) — additive Δb_ω(t).
            process_noise_scaling: Optional (6,) — multiplicative Q scaling.
            bias_noise_scaling: Optional (3,) — per-axis scaling for both
                b_omega and b_acc Q diagonal elements.
        """
        Rot_prev = Rot_prev.clone()

        # Classical inertial kinematics
        acc_b = u[3:6] - b_acc_prev
        if bias_correction is not None:
            acc_b = acc_b - bias_correction
        acc = Rot_prev.mv(acc_b) + self.g
        v = v_prev + acc * dt
        p = p_prev + v_prev.clone() * dt + 0.5 * acc * dt**2

        # Rotation: classical SO(3) integration with optional gyro correction
        omega_corrected = u[:3] - b_omega_prev
        if gyro_correction is not None:
            omega_corrected = omega_corrected - gyro_correction
        omega = omega_corrected * dt
        Rot = Rot_prev.mm(self.so3exp_torch(omega))

        # Biases and extrinsics remain constant
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev.clone()
        t_c_i = t_c_i_prev

        # Propagate covariance
        P = self.propagate_cov_torch(
            P_prev,
            Rot_prev,
            v_prev,
            p_prev,
            b_omega_prev,
            b_acc_prev,
            u,
            dt,
            process_noise_scaling=process_noise_scaling,
            bias_noise_scaling=bias_noise_scaling,
        )

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov_torch(
        self,
        P,
        Rot_prev,
        v_prev,
        p_prev,
        b_omega_prev,
        b_acc_prev,
        u,
        dt,
        process_noise_scaling=None,
        bias_noise_scaling=None,
    ):
        """Propagate covariance matrix one timestep.

        Args:
            process_noise_scaling: Optional (6,) tensor of multiplicative
                scaling factors for Q diagonal blocks.  Order matches
                ``set_Q``: [omega, acc, b_omega, b_acc, Rot_c_i, t_c_i].
            bias_noise_scaling: Optional (3,) tensor of per-axis scaling
                applied elementwise to both b_omega and b_acc Q diagonals.
        """
        F = P.new_zeros(self.P_dim, self.P_dim)
        G = P.new_zeros(self.P_dim, self.Q.shape[0])
        Q = self.Q.clone()

        # Apply per-timestep Q scaling if provided
        if process_noise_scaling is not None:
            for k in range(6):
                Q[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = (
                    Q[3 * k : 3 * k + 3, 3 * k : 3 * k + 3]
                    * process_noise_scaling[k]
                )

        # Apply per-axis bias noise scaling (LatentWorldModel Q_bias_scale).
        # Use torch.cat + torch.diag to avoid in-place ops on a tracked tensor,
        # which would break autograd.
        if bias_noise_scaling is not None:
            Q_dim = Q.shape[0]  # 18
            ones_pre = torch.ones(9, dtype=Q.dtype, device=Q.device)
            ones_post = torch.ones(Q_dim - 15, dtype=Q.dtype, device=Q.device)
            diag_scale = torch.cat(
                [ones_pre, bias_noise_scaling, bias_noise_scaling, ones_post]
            )
            Q = Q * torch.diag(diag_scale)

        v_skew_rot = self.skew_torch(v_prev).mm(Rot_prev)
        p_skew_rot = self.skew_torch(p_prev).mm(Rot_prev)

        # Build state transition matrix F
        F[3:6, :3] = self.skew_torch(self.g)
        F[6:9, 3:6] = self.Id3
        F[3:6, 12:15] = -Rot_prev
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot

        # Build noise matrix G
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        G[3:6, 3:6] = Rot_prev
        G[9:12, 6:9] = self.Id3
        G[12:15, 9:12] = self.Id3
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        # Scale by timestep
        F = F * dt
        G = G * dt

        # Compute state transition matrix
        F_square = F.mm(F)
        Phi = self.IdP + F + 0.5 * F_square + (1.0 / 6.0) * F_square.mm(F)

        # Propagate covariance
        P_new = Phi.mm(P + G.mm(Q).mm(G.t())).mm(Phi.t())

        return P_new

    # ------------------------------------------------------------------
    # Update (correction)
    # ------------------------------------------------------------------

    def update(
        self,
        Rot,
        v,
        p,
        b_omega,
        b_acc,
        Rot_c_i,
        t_c_i,
        P,
        u,
        i,
        measurement_cov,
    ):
        """Update state using zero lateral/vertical velocity constraints."""
        # Orientation of body frame
        Rot_body = Rot.mm(Rot_c_i)

        # Velocity in IMU frame
        v_imu = Rot.t().mv(v)

        # Velocity in body frame
        omega = u[:3] - b_omega
        v_body = Rot_c_i.t().mv(v_imu) + self.skew_torch(t_c_i).mv(omega)
        Omega = self.skew_torch(omega)

        # Measurement Jacobian
        H_v_imu = Rot_c_i.t().mm(self.skew_torch(v_imu))
        H_t_c_i = self.skew_torch(t_c_i)

        H = P.new_zeros(2, self.P_dim)
        H[:, 3:6] = Rot_body.t()[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]

        # Innovation
        r = -v_body[1:]

        # Measurement noise covariance
        R = torch.diag(measurement_cov)

        # Perform update
        (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        ) = self.state_and_cov_update_torch(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
        )

        return (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        )

    @staticmethod
    def state_and_cov_update_torch(
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
    ):
        """Kalman gain, state correction, and covariance update."""
        # Kalman gain
        S = H.mm(P).mm(H.t()) + R
        try:
            # Modern PyTorch
            K = torch.linalg.solve(S, P.mm(H.t()).t()).t()
        except AttributeError:
            # Older PyTorch
            Kt, _ = torch.gesv(P.mm(H.t()).t(), S)
            K = Kt.t()

        # Compute state correction
        dx = K.mv(r.view(-1))

        # Update SE_2(3) state
        dR, dxi = TorchIEKF.se23_exp_torch(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.mm(Rot)
        v_up = dR.mv(v) + dv
        p_up = dR.mv(p) + dp

        # Update biases
        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        # Update extrinsics
        dR = TorchIEKF.so3exp_torch(dx[15:18])
        Rot_c_i_up = dR.mm(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        # Update covariance (Joseph form)
        I_KH = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - K.mm(H)
        P_upprev = I_KH.mm(P).mm(I_KH.t()) + K.mm(R).mm(K.t())
        P_up = (P_upprev + P_upprev.t()) / 2

        return (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        )

    # ==================================================================
    # Geometry utilities (all device-aware via input tensors)
    # ==================================================================

    @staticmethod
    def skew_torch(x):
        """Skew-symmetric matrix from 3-vector."""
        zero = x.new_zeros(1).squeeze()
        return torch.stack(
            [
                torch.stack([zero, -x[2], x[1]]),
                torch.stack([x[2], zero, -x[0]]),
                torch.stack([-x[1], x[0], zero]),
            ]
        )

    @staticmethod
    def so3exp_torch(phi):
        """SO(3) exponential map."""
        angle = phi.norm()
        Id3 = torch.eye(3, dtype=phi.dtype, device=phi.device)

        if isclose(angle, phi.new_tensor(0.0)):
            skew_phi = TorchIEKF.skew_torch(phi)
            return Id3 + skew_phi

        axis = phi / angle
        skew_axis = TorchIEKF.skew_torch(axis)
        c = angle.cos()
        s = angle.sin()

        return c * Id3 + (1 - c) * TorchIEKF.outer(axis, axis) + s * skew_axis

    @staticmethod
    def se23_exp_torch(xi):
        """SE_2(3) exponential map.  Returns (Rot, x) where x is 3x2."""
        phi = xi[:3]
        angle = torch.norm(phi)
        Id3 = torch.eye(3, dtype=xi.dtype, device=xi.device)

        if isclose(angle, xi.new_tensor(0.0)):
            skew_phi = TorchIEKF.skew_torch(phi)
            J = Id3 + 0.5 * skew_phi
            Rot = Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = TorchIEKF.skew_torch(axis)
            s = torch.sin(angle)
            c = torch.cos(angle)

            J = (
                (s / angle) * Id3
                + (1 - s / angle) * TorchIEKF.outer(axis, axis)
                + ((1 - c) / angle) * skew_axis
            )
            Rot = (
                c * Id3 + (1 - c) * TorchIEKF.outer(axis, axis) + s * skew_axis
            )

        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    @staticmethod
    def from_rpy_torch(roll, pitch, yaw):
        """Rotation matrix from roll-pitch-yaw (ZYX convention)."""
        return TorchIEKF.rotz_torch(yaw).mm(
            TorchIEKF.roty_torch(pitch).mm(TorchIEKF.rotx_torch(roll))
        )

    @staticmethod
    def rotx_torch(t):
        """Rotation around x-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        zero = t.new_zeros(())
        one = t.new_ones(())
        return torch.stack(
            [
                torch.stack([one, zero, zero]),
                torch.stack([zero, c, -s]),
                torch.stack([zero, s, c]),
            ]
        )

    @staticmethod
    def roty_torch(t):
        """Rotation around y-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        zero = t.new_zeros(())
        one = t.new_ones(())
        return torch.stack(
            [
                torch.stack([c, zero, s]),
                torch.stack([zero, one, zero]),
                torch.stack([-s, zero, c]),
            ]
        )

    @staticmethod
    def rotz_torch(t):
        """Rotation around z-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        zero = t.new_zeros(())
        one = t.new_ones(())
        return torch.stack(
            [
                torch.stack([c, -s, zero]),
                torch.stack([s, c, zero]),
                torch.stack([zero, zero, one]),
            ]
        )

    @staticmethod
    def outer(a, b):
        """Outer product."""
        return torch.ger(a, b)

    @staticmethod
    def normalize_rot_torch(Rot):
        """Project a near-rotation matrix back onto SO(3) via SVD."""
        U, _, V = torch.svd(Rot)
        S = torch.eye(3, dtype=Rot.dtype, device=Rot.device)
        S[2, 2] = torch.det(U) * torch.det(V)
        return U.mm(S).mm(V.t())

    # ==================================================================
    # Batched geometry utilities  (leading dimension M)
    # ==================================================================

    @staticmethod
    def skew_torch_batched(x):
        """Skew-symmetric matrices from (M, 3) batch of vectors → (M, 3, 3)."""
        M = x.shape[0]
        S = x.new_zeros(M, 3, 3)
        S[:, 0, 1] = -x[:, 2]
        S[:, 0, 2] = x[:, 1]
        S[:, 1, 0] = x[:, 2]
        S[:, 1, 2] = -x[:, 0]
        S[:, 2, 0] = -x[:, 1]
        S[:, 2, 1] = x[:, 0]
        return S

    @staticmethod
    def so3exp_torch_batched(phi):
        """SO(3) exponential map for (M, 3) batch → (M, 3, 3).

        Uses Rodrigues formula with numerically stable small-angle handling
        via ``torch.where`` (no branching, fully differentiable).
        """
        angle = phi.norm(dim=-1, keepdim=True)  # (M, 1)
        # Avoid division by zero: use safe denominator for axis computation
        safe_angle = torch.where(angle > 1e-10, angle, torch.ones_like(angle))
        axis = phi / safe_angle  # (M, 3)

        # Small-angle approximation coefficients
        # sin(a)/a ≈ 1 - a²/6,  (1-cos(a))/a² ≈ 1/2 - a²/24
        a2 = (angle * angle).squeeze(-1)  # (M,)
        sin_a_over_a = torch.where(
            angle.squeeze(-1) > 1e-10,
            torch.sin(angle.squeeze(-1)) / safe_angle.squeeze(-1),
            1.0 - a2 / 6.0,
        )
        one_minus_cos_over_a2 = torch.where(
            angle.squeeze(-1) > 1e-10,
            (1.0 - torch.cos(angle.squeeze(-1))) / (safe_angle.squeeze(-1) ** 2),
            0.5 - a2 / 24.0,
        )

        skew = TorchIEKF.skew_torch_batched(phi)  # (M, 3, 3)
        skew2 = torch.bmm(skew, skew)  # (M, 3, 3)

        Id3 = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
        R = (
            Id3
            + sin_a_over_a.unsqueeze(-1).unsqueeze(-1) * skew
            + one_minus_cos_over_a2.unsqueeze(-1).unsqueeze(-1) * skew2
        )
        return R

    @staticmethod
    def se23_exp_torch_batched(xi):
        """SE_2(3) exponential map for (M, 9) batch.

        Returns:
            Rot: (M, 3, 3)
            x:   (M, 3, 2) — velocity and position corrections
        """
        phi = xi[:, :3]  # (M, 3)
        angle = phi.norm(dim=-1, keepdim=True)  # (M, 1)
        safe_angle = torch.where(angle > 1e-10, angle, torch.ones_like(angle))

        a2 = (angle * angle).squeeze(-1)  # (M,)
        a_sq = safe_angle.squeeze(-1)

        sin_a_over_a = torch.where(
            angle.squeeze(-1) > 1e-10,
            torch.sin(angle.squeeze(-1)) / a_sq,
            1.0 - a2 / 6.0,
        )
        one_minus_cos_over_a2 = torch.where(
            angle.squeeze(-1) > 1e-10,
            (1.0 - torch.cos(angle.squeeze(-1))) / (a_sq ** 2),
            0.5 - a2 / 24.0,
        )
        # (a - sin(a))/a^3 ≈ 1/6 - a²/120 for small angles
        a_minus_sin_over_a3 = torch.where(
            angle.squeeze(-1) > 1e-10,
            (a_sq - torch.sin(angle.squeeze(-1))) / (a_sq ** 3),
            1.0 / 6.0 - a2 / 120.0,
        )

        skew = TorchIEKF.skew_torch_batched(phi)  # (M, 3, 3)
        skew2 = torch.bmm(skew, skew)  # (M, 3, 3)

        Id3 = torch.eye(3, dtype=xi.dtype, device=xi.device).unsqueeze(0)

        Rot = (
            Id3
            + sin_a_over_a.unsqueeze(-1).unsqueeze(-1) * skew
            + one_minus_cos_over_a2.unsqueeze(-1).unsqueeze(-1) * skew2
        )

        J = (
            Id3
            + one_minus_cos_over_a2.unsqueeze(-1).unsqueeze(-1) * skew
            + a_minus_sin_over_a3.unsqueeze(-1).unsqueeze(-1) * skew2
        )

        # xi[3:9] → (M, 2, 3) (two 3-vectors: dv and dp)
        vp = xi[:, 3:].reshape(-1, 2, 3)  # (M, 2, 3)
        # J @ vp^T → (M, 3, 2)
        x = torch.bmm(J, vp.transpose(1, 2))  # (M, 3, 2)
        return Rot, x

    @staticmethod
    def normalize_rot_torch_batched(Rot):
        """Project (M, 3, 3) near-rotation matrices back onto SO(3) via SVD."""
        U, _, Vh = torch.linalg.svd(Rot)
        det_sign = torch.det(U) * torch.det(Vh)  # (M,)
        S = torch.eye(3, dtype=Rot.dtype, device=Rot.device).unsqueeze(0).expand_as(Rot).clone()
        S[:, 2, 2] = det_sign
        return torch.bmm(U, torch.bmm(S, Vh))

    # ==================================================================
    # Batched filter methods  (leading dimension M for particles)
    # ==================================================================

    def init_state_batched(self, t, u, v_mes, ang0, M):
        """
        Build initial filter state replicated for M particles.

        Args:
            t, u, v_mes, ang0: Same as ``init_state``.
            M: Number of particles.

        Returns:
            State dict with all tensors having leading dimension M.
        """
        single = self.init_state(t, u, v_mes, ang0)
        batched = {}
        for k, v in single.items():
            batched[k] = v.unsqueeze(0).expand(M, *v.shape).clone()
        return batched

    def propagate_batched(
        self,
        Rot_prev,   # (M, 3, 3)
        v_prev,     # (M, 3)
        p_prev,     # (M, 3)
        b_omega_prev,  # (M, 3)
        b_acc_prev,    # (M, 3)
        Rot_c_i_prev,  # (M, 3, 3)
        t_c_i_prev,    # (M, 3)
        P_prev,        # (M, 21, 21)
        u,             # (6,) shared across particles
        dt,            # scalar
        bias_correction=None,       # (M, 3) or None
        gyro_correction=None,       # (M, 3) or None
        process_noise_scaling=None, # (M, 6) or None
        bias_noise_scaling=None,    # (M, 3) or None
    ):
        """Propagate M filter instances forward one timestep (batched)."""
        M = Rot_prev.shape[0]
        Rot_prev = Rot_prev.clone()

        # Accelerometer: u[3:6] is shared, biases are per-particle
        acc_b = u[3:6].unsqueeze(0) - b_acc_prev  # (M, 3)
        if bias_correction is not None:
            acc_b = acc_b - bias_correction
        # Rot_prev @ acc_b  →  bmm: (M,3,3) @ (M,3,1) → (M,3)
        acc = torch.bmm(Rot_prev, acc_b.unsqueeze(-1)).squeeze(-1) + self.g  # (M, 3)
        v = v_prev + acc * dt
        p = p_prev + v_prev * dt + 0.5 * acc * dt ** 2

        # Rotation
        omega_corrected = u[:3].unsqueeze(0) - b_omega_prev  # (M, 3)
        if gyro_correction is not None:
            omega_corrected = omega_corrected - gyro_correction
        omega = omega_corrected * dt  # (M, 3)
        Rot = torch.bmm(Rot_prev, self.so3exp_torch_batched(omega))  # (M, 3, 3)

        # Biases / extrinsics constant
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev.clone()
        t_c_i = t_c_i_prev

        # Covariance
        P = self.propagate_cov_torch_batched(
            P_prev, Rot_prev, v_prev, p_prev,
            b_omega_prev, b_acc_prev, u, dt,
            process_noise_scaling=process_noise_scaling,
            bias_noise_scaling=bias_noise_scaling,
        )
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov_torch_batched(
        self, P, Rot_prev, v_prev, p_prev,
        b_omega_prev, b_acc_prev, u, dt,
        process_noise_scaling=None,
        bias_noise_scaling=None,
    ):
        """Propagate covariance for M particles in parallel.

        Args:
            P:                    (M, 21, 21)
            Rot_prev:             (M, 3, 3)
            v_prev, p_prev, ...:  (M, 3)
            u:                    (6,)  shared
            dt:                   scalar
            process_noise_scaling: (M, 6) or None
            bias_noise_scaling:    (M, 3) or None
        """
        M = P.shape[0]
        Q_dim = self.Q.shape[0]  # 18

        # Start from the base Q, expand to (M, 18, 18)
        Q = self.Q.unsqueeze(0).expand(M, -1, -1).clone()  # (M, 18, 18)

        if process_noise_scaling is not None:
            # process_noise_scaling: (M, 6)
            for k in range(6):
                Q[:, 3*k:3*k+3, 3*k:3*k+3] = (
                    Q[:, 3*k:3*k+3, 3*k:3*k+3]
                    * process_noise_scaling[:, k].unsqueeze(-1).unsqueeze(-1)
                )

        if bias_noise_scaling is not None:
            # bias_noise_scaling: (M, 3)
            ones_pre = torch.ones(M, 9, dtype=Q.dtype, device=Q.device)
            ones_post = torch.ones(M, Q_dim - 15, dtype=Q.dtype, device=Q.device)
            diag_scale = torch.cat(
                [ones_pre, bias_noise_scaling, bias_noise_scaling, ones_post], dim=1
            )  # (M, 18)
            Q = Q * torch.diag_embed(diag_scale)  # (M, 18, 18)

        # Batched skew products
        v_skew = self.skew_torch_batched(v_prev)           # (M, 3, 3)
        p_skew = self.skew_torch_batched(p_prev)           # (M, 3, 3)
        v_skew_rot = torch.bmm(v_skew, Rot_prev)           # (M, 3, 3)
        p_skew_rot = torch.bmm(p_skew, Rot_prev)           # (M, 3, 3)

        # Gravity skew is shared across particles
        g_skew = self.skew_torch(self.g)                    # (3, 3)

        # Build F: (M, 21, 21)
        F = P.new_zeros(M, self.P_dim, self.P_dim)
        F[:, 3:6, :3] = g_skew.unsqueeze(0)                # shared
        F[:, 6:9, 3:6] = self.Id3.unsqueeze(0)             # shared
        F[:, 3:6, 12:15] = -Rot_prev
        F[:, :3, 9:12] = -Rot_prev
        F[:, 3:6, 9:12] = -v_skew_rot
        F[:, 6:9, 9:12] = -p_skew_rot

        # Build G: (M, 21, 18)
        G = P.new_zeros(M, self.P_dim, Q_dim)
        G[:, :3, :3] = Rot_prev
        G[:, 3:6, :3] = v_skew_rot
        G[:, 6:9, :3] = p_skew_rot
        G[:, 3:6, 3:6] = Rot_prev
        G[:, 9:12, 6:9] = self.Id3.unsqueeze(0)
        G[:, 12:15, 9:12] = self.Id3.unsqueeze(0)
        G[:, 15:18, 12:15] = self.Id3.unsqueeze(0)
        G[:, 18:21, 15:18] = self.Id3.unsqueeze(0)

        F = F * dt
        G = G * dt

        # Phi = I + F + F²/2 + F³/6
        F2 = torch.bmm(F, F)
        IdP = self.IdP.unsqueeze(0)  # (1, 21, 21)
        Phi = IdP + F + 0.5 * F2 + (1.0 / 6.0) * torch.bmm(F2, F)

        # P_new = Phi @ (P + G @ Q @ G^T) @ Phi^T
        GQ = torch.bmm(G, Q)                               # (M, 21, 18)
        GQGt = torch.bmm(GQ, G.transpose(1, 2))            # (M, 21, 21)
        inner = P + GQGt
        P_new = torch.bmm(Phi, torch.bmm(inner, Phi.transpose(1, 2)))
        return P_new

    def update_batched(
        self,
        Rot,            # (M, 3, 3)
        v,              # (M, 3)
        p,              # (M, 3)
        b_omega,        # (M, 3)
        b_acc,          # (M, 3)
        Rot_c_i,        # (M, 3, 3)
        t_c_i,          # (M, 3)
        P,              # (M, 21, 21)
        u,              # (6,) shared
        i,              # timestep index (unused, kept for API compat)
        measurement_cov,  # (M, 2)
    ):
        """Update M filter instances using zero velocity constraints (batched)."""
        M = Rot.shape[0]

        # Body frame orientation
        Rot_body = torch.bmm(Rot, Rot_c_i)  # (M, 3, 3)

        # Velocity in IMU frame: Rot^T @ v  →  (M, 3, 3)^T @ (M, 3, 1)
        v_imu = torch.bmm(Rot.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)  # (M, 3)

        # Angular velocity
        omega = u[:3].unsqueeze(0) - b_omega  # (M, 3)

        # Velocity in body frame: Rot_c_i^T @ v_imu + skew(t_c_i) @ omega
        v_imu_in_body = torch.bmm(
            Rot_c_i.transpose(1, 2), v_imu.unsqueeze(-1)
        ).squeeze(-1)  # (M, 3)
        skew_t = self.skew_torch_batched(t_c_i)  # (M, 3, 3)
        v_body = v_imu_in_body + torch.bmm(skew_t, omega.unsqueeze(-1)).squeeze(-1)

        Omega = self.skew_torch_batched(omega)  # (M, 3, 3)

        # Measurement Jacobian pieces
        skew_v_imu = self.skew_torch_batched(v_imu)  # (M, 3, 3)
        H_v_imu = torch.bmm(Rot_c_i.transpose(1, 2), skew_v_imu)  # (M, 3, 3)
        H_t_c_i = skew_t  # (M, 3, 3)

        # Build H: (M, 2, 21) — rows 1,2 of the body-frame velocity Jacobian
        H = P.new_zeros(M, 2, self.P_dim)
        H[:, :, 3:6] = Rot_body.transpose(1, 2)[:, 1:, :]     # rows 1,2
        H[:, :, 15:18] = H_v_imu[:, 1:, :]
        H[:, :, 9:12] = H_t_c_i[:, 1:, :]
        H[:, :, 18:21] = -Omega[:, 1:, :]

        # Innovation
        r = -v_body[:, 1:]  # (M, 2)

        # Measurement noise
        R = torch.diag_embed(measurement_cov)  # (M, 2, 2)

        return self.state_and_cov_update_torch_batched(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
        )

    def state_and_cov_update_torch_batched(
        self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
    ):
        """Batched Kalman gain, state correction, and covariance update.

        All inputs have leading dimension M.
        """
        M = Rot.shape[0]

        # S = H P H^T + R   → (M, 2, 2)
        S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R

        # K = P H^T S^{-1}  → (M, 21, 2)
        PHt = torch.bmm(P, H.transpose(1, 2))  # (M, 21, 2)
        # Solve S^T K^T = (P H^T)^T  ⟹  K = (P H^T) @ S^{-T}
        # torch.linalg.solve(A, B) solves A X = B
        K = torch.linalg.solve(S.transpose(1, 2), PHt.transpose(1, 2)).transpose(1, 2)

        # dx = K @ r  → (M, 21)
        dx = torch.bmm(K, r.unsqueeze(-1)).squeeze(-1)

        # SE_2(3) correction
        dR, dxi = self.se23_exp_torch_batched(dx[:, :9])  # dR: (M,3,3), dxi: (M,3,2)
        dv = dxi[:, :, 0]  # (M, 3)
        dp = dxi[:, :, 1]  # (M, 3)
        Rot_up = torch.bmm(dR, Rot)
        v_up = torch.bmm(dR, v.unsqueeze(-1)).squeeze(-1) + dv
        p_up = torch.bmm(dR, p.unsqueeze(-1)).squeeze(-1) + dp

        # Bias updates
        b_omega_up = b_omega + dx[:, 9:12]
        b_acc_up = b_acc + dx[:, 12:15]

        # Extrinsic updates
        dR_ci = self.so3exp_torch_batched(dx[:, 15:18])
        Rot_c_i_up = torch.bmm(dR_ci, Rot_c_i)
        t_c_i_up = t_c_i + dx[:, 18:21]

        # Joseph form covariance update
        IdP = torch.eye(
            K.shape[1], dtype=K.dtype, device=K.device
        ).unsqueeze(0).expand(M, -1, -1)
        I_KH = IdP - torch.bmm(K, H)  # (M, 21, 21)
        P_up = (
            torch.bmm(I_KH, torch.bmm(P, I_KH.transpose(1, 2)))
            + torch.bmm(K, torch.bmm(R, K.transpose(1, 2)))
        )
        P_up = (P_up + P_up.transpose(1, 2)) / 2  # symmetrize

        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    def run_chunk_batched(
        self,
        state,
        t_chunk,
        u_chunk,
        measurements_covs_chunk,  # (M, K, 2)
        bias_corrections_chunk=None,        # (M, K, 3) or None
        gyro_corrections_chunk=None,        # (M, K, 3) or None
        process_noise_scaling_chunk=None,   # (M, K, 6) or None
        bias_noise_scaling_chunk=None,      # (M, K, 3) or None
    ):
        """
        Run M batched filter instances for one BPTT chunk.

        Args:
            state: Dict with (M, ...) tensors from ``init_state_batched``
                   or previous chunk.
            t_chunk:  (K,)  timestamps.
            u_chunk:  (K, 6) shared IMU.
            measurements_covs_chunk: (M, K, 2) per-particle measurement covs.
            bias_corrections_chunk:  (M, K, 3) or None.
            gyro_corrections_chunk:  (M, K, 3) or None.
            process_noise_scaling_chunk: (M, K, 6) or None.
            bias_noise_scaling_chunk:    (M, K, 3) or None.

        Returns:
            traj:      Tuple (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i),
                       each (M, K, ...).
            new_state: Dict with (M, ...) tensors at end of chunk.
        """
        K = t_chunk.shape[0]
        M = state["Rot"].shape[0]
        dt_chunk = t_chunk[1:] - t_chunk[:-1]

        # Allocate trajectory tensors
        Rot = t_chunk.new_zeros(M, K, 3, 3).float()
        v_traj = t_chunk.new_zeros(M, K, 3).float()
        p_traj = t_chunk.new_zeros(M, K, 3).float()
        b_omega_traj = t_chunk.new_zeros(M, K, 3).float()
        b_acc_traj = t_chunk.new_zeros(M, K, 3).float()
        Rot_c_i_traj = t_chunk.new_zeros(M, K, 3, 3).float()
        t_c_i_traj = t_chunk.new_zeros(M, K, 3).float()

        # Seed first timestep
        Rot[:, 0] = state["Rot"]
        v_traj[:, 0] = state["v"]
        p_traj[:, 0] = state["p"]
        b_omega_traj[:, 0] = state["b_omega"]
        b_acc_traj[:, 0] = state["b_acc"]
        Rot_c_i_traj[:, 0] = state["Rot_c_i"]
        t_c_i_traj[:, 0] = state["t_c_i"]
        P = state["P"]  # (M, 21, 21)

        for i in range(1, K):
            bc_i = bias_corrections_chunk[:, i] if bias_corrections_chunk is not None else None
            gc_i = gyro_corrections_chunk[:, i] if gyro_corrections_chunk is not None else None
            pns_i = process_noise_scaling_chunk[:, i] if process_noise_scaling_chunk is not None else None
            bns_i = bias_noise_scaling_chunk[:, i] if bias_noise_scaling_chunk is not None else None

            (Rot_i, v_i, p_i, b_omega_i, b_acc_i,
             Rot_c_i_i, t_c_i_i, P_i) = self.propagate_batched(
                Rot[:, i - 1],
                v_traj[:, i - 1],
                p_traj[:, i - 1],
                b_omega_traj[:, i - 1],
                b_acc_traj[:, i - 1],
                Rot_c_i_traj[:, i - 1],
                t_c_i_traj[:, i - 1],
                P,
                u_chunk[i],
                dt_chunk[i - 1],
                bias_correction=bc_i,
                gyro_correction=gc_i,
                process_noise_scaling=pns_i,
                bias_noise_scaling=bns_i,
            )

            (Rot[:, i], v_traj[:, i], p_traj[:, i],
             b_omega_traj[:, i], b_acc_traj[:, i],
             Rot_c_i_traj[:, i], t_c_i_traj[:, i],
             P) = self.update_batched(
                Rot_i, v_i, p_i, b_omega_i, b_acc_i,
                Rot_c_i_i, t_c_i_i, P_i,
                u_chunk[i], i,
                measurements_covs_chunk[:, i],
            )

        traj = (Rot, v_traj, p_traj, b_omega_traj, b_acc_traj,
                Rot_c_i_traj, t_c_i_traj)
        new_state = dict(
            Rot=Rot[:, -1],
            v=v_traj[:, -1],
            p=p_traj[:, -1],
            b_omega=b_omega_traj[:, -1],
            b_acc=b_acc_traj[:, -1],
            Rot_c_i=Rot_c_i_traj[:, -1],
            t_c_i=t_c_i_traj[:, -1],
            P=P,
        )
        return traj, new_state

    # ------------------------------------------------------------------
    # Network integration
    # ------------------------------------------------------------------

    def forward_nets(self, u):
        """
        Predict measurement covariances from raw IMU data.

        Args:
            u: IMU measurements (N, 6).

        Returns:
            Measurement covariances (N, 2).
        """
        if self.mes_net is None:
            # Use default covariances if no network
            return self.cov0_measurement.unsqueeze(0).repeat(u.shape[0], 1)

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        measurements_covs = self.mes_net(u_n, self)
        return measurements_covs

    def forward_bias_net(self, u):
        """
        Predict per-timestep accelerometer bias corrections.

        Args:
            u: IMU measurements (N, 6), raw (not normalized).

        Returns:
            Bias corrections (N, 3) in m/s², or None if no
            bias_correction_net is attached.
        """
        if self.bias_correction_net is None:
            return None

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        return self.bias_correction_net(u_n, self)

    def forward_world_model(self, u):
        """
        Run the world model on raw IMU data.

        Args:
            u: IMU measurements (N, 6), raw (not normalized).

        Returns:
            ``WorldModelOutput`` dataclass, or *None* if no world model
            is attached.
        """
        if self.world_model is None:
            return None

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        return self.world_model(u_n, self)

    def forward_world_model_batched(self, u, M):
        """
        Run the world model with M particle samples on raw IMU data.

        Args:
            u: IMU measurements (N, 6), raw (not normalized).
            M: Number of particles.

        Returns:
            ``WorldModelOutput`` with (M, N, dim) shaped tensors, or *None*
            if no world model is attached.
        """
        if self.world_model is None:
            return None

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        return self.world_model.forward_batched(u_n, self, M)

    def set_Q(self):
        """
        Rebuild Q, optionally using learned scaling factors.

        Called once per training step so the computation graph is fresh.
        """
        # Reset to default values
        q_vals = torch.tensor(
            [
                self.cov_omega,
                self.cov_omega,
                self.cov_omega,
                self.cov_acc,
                self.cov_acc,
                self.cov_acc,
                self.cov_b_omega,
                self.cov_b_omega,
                self.cov_b_omega,
                self.cov_b_acc,
                self.cov_b_acc,
                self.cov_b_acc,
                self.cov_Rot_c_i,
                self.cov_Rot_c_i,
                self.cov_Rot_c_i,
                self.cov_t_c_i,
                self.cov_t_c_i,
                self.cov_t_c_i,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.Q.data.copy_(torch.diag(q_vals))

        if self.initprocesscov_net is None:
            return

        # Apply learned scaling (create on same device as current Q buffer)
        beta = self.initprocesscov_net.init_processcov(self)
        Q_new = self.Q.new_zeros(self.Q.shape[0], self.Q.shape[0])
        Q_new[:3, :3] = self.cov_omega * beta[0] * self.Id3
        Q_new[3:6, 3:6] = self.cov_acc * beta[1] * self.Id3
        Q_new[6:9, 6:9] = self.cov_b_omega * beta[2] * self.Id3
        Q_new[9:12, 9:12] = self.cov_b_acc * beta[3] * self.Id3
        Q_new[12:15, 12:15] = self.cov_Rot_c_i * beta[4] * self.Id3
        Q_new[15:18, 15:18] = self.cov_t_c_i * beta[5] * self.Id3
        self.Q.data.copy_(Q_new)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_u(self, u):
        """Normalize IMU inputs using dataset statistics."""
        if self.u_loc is None or self.u_std is None:
            return u
        return (u - self.u_loc) / self.u_std

    def get_normalize_u(self, dataset):
        """Load normalization parameters from dataset and move to model device."""
        self.u_loc = dataset.normalize_factors["u_loc"].float().to(self.device)
        self.u_std = dataset.normalize_factors["u_std"].float().to(self.device)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def load(self, args, dataset):
        """
        Load trained network weights (legacy format).

        Args:
            args: Arguments containing path_temp
            dataset: Dataset containing normalization factors
        """
        path_iekf = os.path.join(args.path_temp, "iekfnets.p")
        if os.path.isfile(path_iekf):
            mondict = torch.load(path_iekf, map_location=self.device)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", "green")
        else:
            cprint("IEKF nets NOT loaded", "yellow")
        self.get_normalize_u(dataset)


# Alias for backward compatibility
TORCHIEKF = TorchIEKF
