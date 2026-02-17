"""
WandB utility functions for logging configs, plots, and metrics.
"""

import io


def log_config(cfg):
    """
    Flatten an OmegaConf config to a plain dict for WandB.

    Args:
        cfg: OmegaConf DictConfig or plain dict.

    Returns:
        Flat dictionary suitable for wandb.init(config=...).
    """
    try:
        from omegaconf import OmegaConf
        if hasattr(cfg, '_metadata'):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    if isinstance(cfg, dict):
        return cfg
    return dict(cfg)


def log_figure(fig, key="plot"):
    """
    Convert a matplotlib figure to a WandB Image.

    Args:
        fig: matplotlib Figure.
        key: Name for the logged image.

    Returns:
        Dict suitable for wandb.log().
    """
    import wandb

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf)
    return {key: wandb.Image(img)}


def log_metrics_table(metrics_dict, table_name="metrics"):
    """
    Log a dictionary of metrics as a WandB Table.

    Args:
        metrics_dict: Dict of {metric_name: value}.
        table_name: Name for the table.

    Returns:
        Dict suitable for wandb.log().
    """
    import wandb

    columns = list(metrics_dict.keys())
    data = [list(metrics_dict.values())]
    table = wandb.Table(columns=columns, data=data)
    return {table_name: table}
