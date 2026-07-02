from __future__ import annotations

"""
Reproducibility utilities

The training scripts call set_seed() before model training so that Python,
NumPy and PyTorch use the same global random seed. This improves the stability
of experiments across runs, although exact reproducibility can still depend on
hardware, CUDA kernels and library versions.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the main random seeds used by the project.

    Parameters
    ----------
    seed:
        Global seed applied to Python's random module, NumPy and PyTorch.

    Notes
    -----
    CuDNN deterministic mode is enabled and benchmarking is disabled to reduce
    nondeterminism in GPU executions. This can make training slightly slower,
    but helps make experiments more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


