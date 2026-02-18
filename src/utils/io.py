"""
General-purpose I/O utilities shared across entry points.
"""

import os
import shutil


def seed_normalize_factors(paths_cfg):
    """Copy normalize_factors.p from base_temp into the run's temp dir.

    Ensures each Hydra run has its own copy of the pre-computed normalization
    statistics without re-deriving them.  If ``base_temp`` is not configured
    or the source file does not exist, this is a no-op.

    Args:
        paths_cfg: The resolved ``paths`` config node (dict-like).
    """
    src_dir = paths_cfg.get("base_temp")
    dst_dir = paths_cfg.get("temp")
    if not src_dir or not dst_dir:
        return
    src = os.path.join(src_dir, "normalize_factors.p")
    if not os.path.isfile(src):
        return
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "normalize_factors.p")
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print(f"Copied normalize_factors.p: {src} -> {dst}")
