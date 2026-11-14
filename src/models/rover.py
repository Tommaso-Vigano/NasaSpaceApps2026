"""Traversability and per-edge travel time for rovers.

Per-edge arrays follow the convention of ``src.models.terrain``
(shape (8, rows, cols), inf = forbidden or leaving the grid). SI units.
"""

import numpy as np

from src.models.rocks import p_rock_taller_than
from src.models.terrain import (
    horizontal_lengths,
    horn_slope,
    neighbor_stack,
    slope_mask,
    step_masks,
)


def rover_hard_mask(slope_deg, rover, ti=None, dune_mask=None):
    """Per-cell no-go mask (True = forbidden) for a RoboticRover/CrewedRover.

    Forbidden if slope > rover.max_slope_deg [R1, R7], thermal inertia below
    rover.min_thermal_inertia (loose soil, [R4], if ``ti`` is given), or dune
    (if rover.dunes_nogo [R3] and ``dune_mask`` is given). NaN = forbidden.
    """
    mask = slope_mask(slope_deg, rover.max_slope_deg)
    if ti is not None:
        mask = mask | ~(np.asarray(ti, dtype=np.float64) >= rover.min_thermal_inertia)
    if dune_mask is not None and rover.dunes_nogo:
        mask = mask | np.asarray(dune_mask, dtype=bool)
    return mask


def rover_edge_time(z, dx, dy, rover, k_map=None, steps=None, slope_deg=None,
                    ti=None, dune_mask=None):
    """Travel time (s) of every edge, shape (8, rows, cols).

    t = d3D / rover.speed_ms * (1 + p_rock), where p_rock is the probability
    of meeting a rock taller than rover.max_step_m in the swept area
    (horizontal edge length * rover.width_m), with k the mean of the two cells
    (0 if ``k_map`` is None). The (1 + p_rock) factor is a TEAM ASSUMPTION
    (detour/hazard-avoidance penalty).

    inf when: start or end cell forbidden by ``rover_hard_mask`` (slope from
    ``horn_slope`` if not given, so border cells are forbidden), |step| >
    rover.max_step_m (if ``steps`` is given), NaN in z or k, or leaving the grid.
    """
    z = np.asarray(z, dtype=np.float64)
    dh = horizontal_lengths(dx, dy)[:, None, None]
    dz = neighbor_stack(z) - z

    if slope_deg is None:
        slope_deg = horn_slope(z, dx, dy)
    hard = rover_hard_mask(slope_deg, rover, ti, dune_mask) | np.isnan(z)
    # neighbour outside the grid -> NaN -> not == 0 -> forbidden
    valid = np.isfinite(dz) & ~hard & (neighbor_stack(hard.astype(np.float64)) == 0.0)
    if steps is not None:
        valid &= ~step_masks(steps, rover.max_step_m, rover.max_step_m)

    p_rock = 0.0
    if k_map is not None:
        k = np.broadcast_to(np.asarray(k_map, dtype=np.float64), z.shape)
        k_edge = 0.5 * (k + neighbor_stack(k))
        valid &= np.isfinite(k_edge)
        p_rock = p_rock_taller_than(
            rover.max_step_m, np.where(valid, k_edge, 0.0), dh * rover.width_m
        )

    d3 = np.hypot(dh, np.where(valid, dz, 0.0))
    return np.where(valid, d3 / rover.speed_ms * (1.0 + p_rock), np.inf)
