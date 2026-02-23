"""
Export trained PyTorch models for the full `TorchIEKF`.

This script exports the combined `TorchIEKF` (including attached
`world_model` or `mes_net`) as a PyTorch `state_dict`. It also
provides a macOS-only helper that loads a checkpoint/state_dict,
traces the IEKF wrapper and converts directly to CoreML using
`coremltools` (run the conversion on your Mac).

Usage examples:
    # Save state_dict (recommended)
    python3 src/export_coreml.py --export_iekf --checkpoint path.pt --out iekf.pth

    # On macOS: trace checkpoint and convert directly to CoreML
    python3 src/export_coreml.py --trace_and_convert_to_coreml --checkpoint path.pt --out iekf.mlmodel --seq_len 100

Requires: `torch` available in the environment that runs this script.
"""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import platform

# ONNX/TorchScript export removed; we only save a `state_dict` here.

# Ensure repository root is on sys.path so `from src.models...` imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_state_dict_into(model: nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # common checkpoint shapes: raw state_dict, or dict with 'state_dict' key
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state" in ckpt:
            sd = ckpt["model_state"]
        else:
            # maybe it's already a state dict
            sd = ckpt
    else:
        sd = ckpt

    # sometimes keys are prefixed with 'module.' from DataParallel
    def _strip_module(k):
        return k[7:] if k.startswith("module.") else k

    new_sd = {_strip_module(k): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)

    # (measurement/world export removed) keep only full IEKF export below


def export_iekf_full(
    checkpoint: Path,
    seq_len: int,
    out: Path,
    input_channels: int = 6,
):
    """
    Export the full TorchIEKF (including attached neural modules) as a single
    model. The exported wrapper accepts `(u, t)` where `u` is (1, C, N)
    and `t` is (N,) timestamps, and returns trajectories produced by
    `TorchIEKF.run_chunk`.
    """
    from src.core.torch_iekf import TorchIEKF

    iekf = TorchIEKF()
    if checkpoint is not None:
        _load_state_dict_into(iekf, checkpoint)
    iekf.eval()

    class IEKFWrapper(nn.Module):
        def __init__(self, iekf_module: TorchIEKF):
            super().__init__()
            self.iekf = iekf_module

        def forward(self, u: torch.Tensor, t: torch.Tensor):
            # u: (1, C, N) -> convert to (N, C)
            u_seq = u.squeeze(0).transpose(0, 1)
            N = u_seq.shape[0]

            # build dummy v_mes and ang0 for init_state (no GT available on device)
            v_mes = torch.zeros(N, 3, dtype=u.dtype, device=u.device)
            ang0 = torch.zeros(3, dtype=u.dtype, device=u.device)

            self.iekf.set_Q()
            state = self.iekf.init_state(t, u_seq, v_mes, ang0)

            # measurement covariances and corrections from world model / nets
            if self.iekf.world_model is not None:
                wm_out = self.iekf.forward_world_model(u_seq)
                meas = (
                    wm_out.measurement_covs
                    if wm_out.measurement_covs is not None
                    else None
                )
                bias_corr = (
                    wm_out.acc_bias_corrections
                    if wm_out.acc_bias_corrections is not None
                    else None
                )
                gyro_corr = (
                    wm_out.gyro_bias_corrections
                    if wm_out.gyro_bias_corrections is not None
                    else None
                )
                pns = (
                    wm_out.process_noise_scaling
                    if wm_out.process_noise_scaling is not None
                    else None
                )
                bns = (
                    wm_out.bias_noise_scaling
                    if wm_out.bias_noise_scaling is not None
                    else None
                )
            elif self.iekf.mes_net is not None:
                # measurement net expects u as (1, C, N)
                meas = self.iekf.mes_net(u, self.iekf)
                bias_corr = None
                gyro_corr = None
                pns = None
                bns = None
            else:
                meas = self.iekf.cov0_measurement.unsqueeze(0).repeat(N, 1)
                bias_corr = None
                gyro_corr = None
                pns = None
                bns = None

            traj, new_state = self.iekf.run_chunk(
                state,
                t,
                u_seq,
                meas,
                bias_corrections_chunk=bias_corr,
                gyro_corrections_chunk=gyro_corr,
                process_noise_scaling_chunk=pns,
                bias_noise_scaling_chunk=bns,
            )

            # return tuple of trajectories (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
            return traj

    wrapper = IEKFWrapper(iekf)

    # Only save the `state_dict` for the IEKF; tracing/conversion is
    # handled by the macOS-only function below.
    _save_path = out if out.suffix in (".pth",) else out.with_suffix(".pth")
    torch.save(iekf.state_dict(), str(_save_path))
    print(f"Saved state_dict to {_save_path}")
    return


def trace_state_dict_and_convert_to_coreml(
    ckpt_path: Path, out: Path, seq_len: int, input_channels: int = 6
):
    """Load a checkpoint/state_dict, build IEKF wrapper, trace it and convert to CoreML.

    This must run on macOS with `coremltools` installed.
    """
    if platform.system() != "Darwin":
        raise SystemExit(
            "CoreML conversion requires macOS (Darwin). Run this on your Mac."
        )

    try:
        import coremltools as ct
    except Exception as e:
        raise SystemExit(
            f"coremltools is required for conversion on macOS: {e}"
        )

    from src.core.torch_iekf import TorchIEKF

    iekf = TorchIEKF()
    # allow passing either a state_dict file or a full checkpoint
    if ckpt_path is not None:
        _load_state_dict_into(iekf, ckpt_path)
    iekf.eval()

    class IEKFWrapper(nn.Module):
        def __init__(self, iekf_module: TorchIEKF):
            super().__init__()
            self.iekf = iekf_module

        def forward(self, u: torch.Tensor, t: torch.Tensor):
            u_seq = u.squeeze(0).transpose(0, 1)
            N = u_seq.shape[0]
            v_mes = torch.zeros(N, 3, dtype=u.dtype, device=u.device)
            ang0 = torch.zeros(3, dtype=u.dtype, device=u.device)
            self.iekf.set_Q()
            state = self.iekf.init_state(t, u_seq, v_mes, ang0)
            if self.iekf.world_model is not None:
                wm_out = self.iekf.forward_world_model(u_seq)
                meas = (
                    wm_out.measurement_covs
                    if wm_out.measurement_covs is not None
                    else None
                )
                bias_corr = (
                    wm_out.acc_bias_corrections
                    if wm_out.acc_bias_corrections is not None
                    else None
                )
                gyro_corr = (
                    wm_out.gyro_bias_corrections
                    if wm_out.gyro_bias_corrections is not None
                    else None
                )
                pns = (
                    wm_out.process_noise_scaling
                    if wm_out.process_noise_scaling is not None
                    else None
                )
                bns = (
                    wm_out.bias_noise_scaling
                    if wm_out.bias_noise_scaling is not None
                    else None
                )
            elif self.iekf.mes_net is not None:
                meas = self.iekf.mes_net(u, self.iekf)
                bias_corr = None
                gyro_corr = None
                pns = None
                bns = None
            else:
                meas = self.iekf.cov0_measurement.unsqueeze(0).repeat(N, 1)
                bias_corr = None
                gyro_corr = None
                pns = None
                bns = None

            traj, new_state = self.iekf.run_chunk(
                state,
                t,
                u_seq,
                meas,
                bias_corrections_chunk=bias_corr,
                gyro_corrections_chunk=gyro_corr,
                process_noise_scaling_chunk=pns,
                bias_noise_scaling_chunk=bns,
            )
            return traj

    wrapper = IEKFWrapper(iekf)
    wrapper.eval()

    example_u = torch.randn(1, input_channels, seq_len)
    example_t = torch.linspace(0.0, float(seq_len - 1), steps=seq_len)

    traced = torch.jit.trace(wrapper, (example_u, example_t), strict=False)

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(shape=tuple(example_u.shape)),
                ct.TensorType(shape=tuple(example_t.shape)),
            ],
        )
        mlmodel.save(str(out))
        print(f"Traced and converted checkpoint {ckpt_path} -> CoreML {out}")
    except Exception as e:
        raise SystemExit(f"Trace+CoreML conversion failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--export_iekf",
        action="store_true",
        help="Export the combined TorchIEKF (with attached nets)",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--seq_len", type=int, default=1000)
    p.add_argument("--input_channels", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    # Exports are now state_dict-only; use --trace_and_convert_to_coreml to
    # produce a CoreML model on macOS by tracing the checkpoint.
    p.add_argument(
        "--trace_and_convert_to_coreml",
        action="store_true",
        help="On macOS: load checkpoint/state_dict, trace IEKF and convert to CoreML",
    )
    p.add_argument(
        "--state_dict_input",
        type=Path,
        default=None,
        help="Optional path to an existing state_dict (.pth) to use for tracing",
    )
    args = p.parse_args()
    # Conversion path: trace a checkpoint/state_dict and convert to CoreML (macOS only)
    if args.trace_and_convert_to_coreml:
        ckpt = (
            args.state_dict_input
            if args.state_dict_input is not None
            else args.checkpoint
        )
        if ckpt is None:
            raise SystemExit(
                "Provide either --state_dict_input or --checkpoint when --trace_and_convert_to_coreml is set"
            )
        trace_state_dict_and_convert_to_coreml(
            ckpt, args.out, args.seq_len, args.input_channels
        )
        return

    if args.export_iekf:
        export_iekf_full(
            args.checkpoint,
            args.seq_len,
            args.out,
            args.input_channels,
        )
        return

    raise SystemExit(
        "No action specified. Use --export_iekf or --trace_and_convert_to_coreml."
    )


if __name__ == "__main__":
    main()
