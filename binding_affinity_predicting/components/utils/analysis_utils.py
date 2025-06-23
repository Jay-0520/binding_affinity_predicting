import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pymbar

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bootstrap_error(
    data: np.ndarray, n_bootstrap: int = 1_000, statistic_func=np.mean
) -> tuple[float, float]:
    """
    Bootstrap error estimate for any statistic.
    Returns (statistic, bootstrap_std).
    """
    if len(data) < 2:
        return float(statistic_func(data)), 0.0

    orig = statistic_func(data)
    stats = [
        statistic_func(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    return orig, float(np.std(stats))


def block_averaging(
    data: np.ndarray, max_block_size: int | None = None
) -> dict[str, np.ndarray]:
    """
    Perform block-averaging analysis to estimate how error depends on block size.
    Returns a dict with arrays of block_sizes, block_means, and block_errors.
    """
    n = len(data)
    max_block = (n // 4) if max_block_size is None else min(n // 2, max_block_size)
    sizes, means, errors = [], [], []

    for bs in range(1, max_block + 1):
        m = n // bs
        if m < 2:
            break
        blocks = data[: m * bs].reshape(m, bs)
        block_means = np.mean(blocks, axis=1)
        sizes.append(bs)
        means.append(np.mean(block_means))
        errors.append(np.std(block_means) / np.sqrt(m))

    return {
        'block_sizes': np.array(sizes),
        'block_averages': np.array(means),
        'block_errors': np.array(errors),
    }


def _load_potential_energies_from_xvg() -> np.ndarray:
    """
    Placeholder function to load potential energies from GROMACS EDR files.

    This function should be implemented to read the potential energy data
    required for free energy calculations. It is currently a stub.

    Returns:
    --------
    np.ndarray
        Placeholder array for potential energies.

    Raises:
    -------
    NotImplementedError
        If called without implementation.
    """
    raise NotImplementedError(
        "This function should be implemented to load potential energies."
    )
