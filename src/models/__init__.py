"""
Model registry for covariance and dynamics networks.

Usage:
    from src.models import get_model, list_models

    # Get a model class by name
    model = get_model("MeasurementCovNet")(input_channels=6, output_dim=2)

    # List all registered models
    print(list_models())
"""

from src.models.base_covariance_net import BaseCovarianceNet
from src.models.init_process_cov_net import InitProcessCovNet
from src.models.measurement_cov_net import MeasurementCovNet
from src.models.learned_bias_correction_net import LearnedBiasCorrectionNet
from src.models.world_model import LatentWorldModel

# Model registry
_MODEL_REGISTRY = {}


def register_model(cls):
    """Decorator to register a model class in the registry."""
    _MODEL_REGISTRY[cls.__name__] = cls
    return cls


def get_model(name):
    """
    Get a model class by name.

    Args:
        name: Registered model name (e.g., "MeasurementCovNet")

    Returns:
        The model class (not instantiated).

    Raises:
        KeyError: If model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name]


def list_models():
    """Return list of all registered model names."""
    return list(_MODEL_REGISTRY.keys())


# Register all built-in models
register_model(InitProcessCovNet)
register_model(MeasurementCovNet)
register_model(LearnedBiasCorrectionNet)
register_model(LatentWorldModel)
