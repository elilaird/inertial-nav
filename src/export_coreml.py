"""
Export trained PyTorch models for the full `TorchIEKF`.

This script exports the combined `TorchIEKF` (including attached
`world_model` or `mes_net`) as either a TorchScript file or a PyTorch
`state_dict`. ONNX and CoreML conversion paths were removed — convert
to CoreML on a macOS machine from the saved TorchScript or state_dict.

Usage examples:
    # Save TorchScript (recommended)
    python3 src/export_coreml.py --export_iekf --seq_len 100 --checkpoint path.pt --out iekf.pt --export_format torchscript

    # Save state_dict
    python3 src/export_coreml.py --export_iekf --checkpoint path.pt --out iekf.pth --export_format state_dict

Requires: `torch` available in the environment that runs this script.
"""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import platform

# ONNX export removed; we only save TorchScript or a state_dict for macOS CoreML conversion

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
    export_format: str = "torchscript",
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

    example_u = torch.randn(1, input_channels, seq_len)
    example_t = torch.linspace(0.0, float(seq_len - 1), steps=seq_len)

    traced = torch.jit.trace(wrapper, (example_u, example_t), strict=False)

    if export_format == "torchscript":
        _save_path = out if out.suffix == ".pt" else out.with_suffix(".pt")
        traced.save(str(_save_path))
        print(f"Saved TorchScript to {_save_path}")
        return

    if export_format == "state_dict":
        _save_path = (
            out if out.suffix in (".pth", ".pt") else out.with_suffix(".pth")
        )
        torch.save(iekf.state_dict(), str(_save_path))
        print(f"Saved state_dict to {_save_path}")
        return

    # fallback: always save TorchScript
    _fallback = out.with_suffix(".pt")
    traced.save(str(_fallback))
    print(
        f"Unknown export format '{export_format}'. Saved TorchScript fallback to {_fallback}"
    )


def _save_torchscript(traced: torch.jit.ScriptModule, out: Path):
    ts_path = (
        out if out.suffix == ".pt" else out.with_suffix(out.suffix + ".pt")
    )
    traced.save(str(ts_path))
    print(f"Saved TorchScript to {ts_path}")


def convert_torchscript_to_coreml(
    ts_input: Path, out: Path, seq_len: int, input_channels: int = 6
):
    """Load a TorchScript model and convert to CoreML using coremltools.

    This function must be run on macOS where `coremltools` and
    `libcoremlpython` are available.
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

    if not ts_input.exists():
        raise SystemExit(f"TorchScript input not found: {ts_input}")

    m = torch.jit.load(str(ts_input), map_location="cpu")
    example_u = torch.randn(1, input_channels, seq_len)
    example_t = torch.linspace(0.0, float(seq_len - 1), steps=seq_len)

    try:
        mlmodel = ct.convert(
            m,
            inputs=[
                ct.TensorType(shape=tuple(example_u.shape)),
                ct.TensorType(shape=tuple(example_t.shape)),
            ],
        )
        mlmodel.save(str(out))
        print(f"Converted TorchScript {ts_input} -> CoreML {out}")
    except Exception as e:
        raise SystemExit(f"CoreML conversion failed: {e}")


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
    p.add_argument(
        "--export_format",
        choices=["torchscript", "state_dict"],
        default="torchscript",
        help="Format to export: 'torchscript' (default) or 'state_dict'",
    )
    p.add_argument(
        "--convert_torchscript_to_coreml",
        action="store_true",
        help="On macOS: convert a TorchScript file to CoreML (use --ts_input)",
    )
    p.add_argument(
        "--ts_input",
        type=Path,
        default=None,
        help="Input TorchScript file to convert to CoreML",
    )
    args = p.parse_args()
    # Conversion path: TorchScript -> CoreML (macOS only)
    if args.convert_torchscript_to_coreml:
        if args.ts_input is None:
            raise SystemExit(
                "--ts_input is required when --convert_torchscript_to_coreml is set"
            )
        convert_torchscript_to_coreml(
            args.ts_input, args.out, args.seq_len, args.input_channels
        )
        return

    export_format = args.export_format

    if args.export_iekf:
        export_iekf_full(
            args.checkpoint,
            args.seq_len,
            args.out,
            args.input_channels,
            export_format=export_format,
        )
        return

    raise SystemExit(
        "No action specified. Use --export_iekf or --convert_onnx_to_coreml."
    )


if __name__ == "__main__":
    main()
