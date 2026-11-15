"""Vectorised helpers to turn per-cell masks into per-edge cost arrays.

Per-edge arrays follow ``src.models.terrain``: shape ``(8, *z.shape)``,
``a[k, r, c]`` = edge ``(r, c) -> (r, c) + OFFSETS[k]``. Shapes always come
from the inputs.
"""

import numpy as np

from src.models.terrain import neighbor_stack


def zeros_like_edges(z) -> np.ndarray:
    """All-zero per-edge array for the grid of ``z``."""
    return np.zeros((8, *np.shape(z)))


def inf_like_edges(z) -> np.ndarray:
    """All-inf per-edge array (global NO-GO) for the grid of ``z``."""
    return np.full((8, *np.shape(z)), np.inf)


def inside_edges(shape) -> np.ndarray:
    """Bool ``(8, *shape)``: True where the edge stays inside the grid."""
    return np.isfinite(neighbor_stack(np.zeros(shape)))


def target_mask(cell_mask) -> np.ndarray:
    """Bool ``(8, *shape)``: True where the edge ENDS in a masked cell
    (edges leaving the grid are False)."""
    return neighbor_stack(np.asarray(cell_mask, dtype=np.float64)) == 1.0


def entering_inf(cell_mask) -> np.ndarray:
    """0 / inf per edge: inf on the edges that enter a masked cell."""
    return np.where(target_mask(cell_mask), np.inf, 0.0)


def touching_inf(cell_mask) -> np.ndarray:
    """0 / inf per edge: inf on the edges that start or end in a masked cell
    (edges leaving the grid only count their start cell)."""
    m = np.asarray(cell_mask, dtype=bool)
    return np.where(target_mask(m) | m[None, :, :], np.inf, 0.0)


def edge_mean(a) -> np.ndarray:
    """Mean of the two endpoint cells for every edge (NaN off the grid)."""
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a + neighbor_stack(a))
