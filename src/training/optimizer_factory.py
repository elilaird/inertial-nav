"""
Optimizer and learning rate scheduler factory.

Builds optimizers with per-parameter-group learning rates from config,
matching the original training setup where different network components
have distinct learning rates and weight decay.
"""

import torch


def build_optimizer(cfg, model):
    """
    Build optimizer with parameter groups from config.

    Supports per-component learning rates: the config can specify
    different lr/weight_decay for init_process_cov_net, measurement_cov_net
    sub-modules, and bias_correction_net.

    Args:
        cfg: Optimizer config (from configs/optimizer/*.yaml)
        model: TorchIEKF model instance with network attributes.

    Returns:
        Configured optimizer instance.
    """
    param_groups = _build_param_groups(cfg, model)

    if not param_groups:
        # Fallback: optimize all parameters with a single group
        param_groups = [{"params": model.parameters(), "lr": 1e-4}]

    optimizer_type = cfg.get("type", "Adam")
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type == "SGD":
        lr = cfg.get("lr", 1e-4)
        momentum = cfg.get("momentum", 0.9)
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def _build_param_groups(cfg, model):
    """Build parameter group list from config and model."""
    param_groups = []
    groups_cfg = cfg.get("param_groups", {})

    # InitProcessCovNet
    ipc_cfg = groups_cfg.get("init_process_cov_net", None)
    if ipc_cfg and model.initprocesscov_net is not None:
        param_groups.append(
            {
                "params": model.initprocesscov_net.parameters(),
                "lr": ipc_cfg.get("lr", 1e-4),
                "weight_decay": ipc_cfg.get("weight_decay", 0.0),
            }
        )

    # MeasurementCovNet — supports sub-module groups (cov_net, cov_lin)
    mes_cfg = groups_cfg.get("measurement_cov_net", None)
    if mes_cfg and model.mes_net is not None:
        if _has_nested_lr(mes_cfg):
            # Per-submodule learning rates
            for sub_name, sub_cfg in mes_cfg.items():
                if hasattr(model.mes_net, sub_name):
                    param_groups.append(
                        {
                            "params": getattr(
                                model.mes_net, sub_name
                            ).parameters(),
                            "lr": sub_cfg.get("lr", 1e-4),
                            "weight_decay": sub_cfg.get("weight_decay", 0.0),
                        }
                    )
        else:
            # Single group for entire mes_net
            param_groups.append(
                {
                    "params": model.mes_net.parameters(),
                    "lr": mes_cfg.get("lr", 1e-4),
                    "weight_decay": mes_cfg.get("weight_decay", 0.0),
                }
            )

    # BiasCorrection net
    bc_cfg = groups_cfg.get("bias_correction_net", None)
    if bc_cfg and model.bias_correction_net is not None:
        param_groups.append(
            {
                "params": model.bias_correction_net.parameters(),
                "lr": bc_cfg.get("lr", 1e-4),
                "weight_decay": bc_cfg.get("weight_decay", 0.0),
            }
        )

    return param_groups


def _has_nested_lr(cfg):
    """Check if config has nested sub-module configs (dicts with 'lr' keys)."""
    for v in cfg.values():
        if isinstance(v, dict) and "lr" in v:
            return True
    return False


def build_scheduler(cfg, optimizer):
    """
    Build learning rate scheduler from config.

    Args:
        cfg: Scheduler config section (from optimizer config).
        optimizer: The optimizer to schedule.

    Returns:
        Scheduler instance, or None if disabled.
    """
    if not cfg.get("enabled", False):
        return None

    sched_type = cfg.get("type", "ReduceLROnPlateau")

    if sched_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 10),
            min_lr=cfg.get("min_lr", 1e-5),
        )
    elif sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 50),
            gamma=cfg.get("gamma", 0.5),
        )
    elif sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("T_max", 100),
            eta_min=cfg.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
