"""
Loss function registry for trajectory evaluation.

Usage:
    from src.losses import get_loss, list_losses

    loss_fn = get_loss("RPELoss", cfg={"criterion": "mse"})
"""

from src.losses.base_loss import TrajectoryLoss
from src.losses.rpe_loss import RPELoss
from src.losses.ate_loss import ATELoss

_LOSS_REGISTRY = {
    "RPELoss": RPELoss,
    "ATELoss": ATELoss,
}


def get_loss(name, cfg=None):
    """
    Get a loss function instance by name.

    Args:
        name: Registered loss name (e.g., "RPELoss")
        cfg: Optional config dict for the loss.

    Returns:
        Instantiated loss function.
    """
    if name not in _LOSS_REGISTRY:
        available = ", ".join(_LOSS_REGISTRY.keys())
        raise KeyError(f"Loss '{name}' not found. Available: {available}")
    return _LOSS_REGISTRY[name](cfg)


def list_losses():
    """Return list of all registered loss names."""
    return list(_LOSS_REGISTRY.keys())
