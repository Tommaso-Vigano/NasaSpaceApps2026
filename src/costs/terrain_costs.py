"""Cost terms from the two terrain layers: the DEM (time, slope, steps) and the
NOAH-H terrain coefficient eta.

Every function returns a ``(8, *ctx.z.shape)`` array in equivalent seconds,
inf = forbidden, with the ``OFFSETS`` convention of ``src.models.terrain``.
"""

import numpy as np

from src.costs._edges import edge_mean, touching_inf, zeros_like_edges
from src.costs.context import CostContext, is_walker
from src.models.metabolic import calibrate_dM_suit, edge_costs
from src.models.rover import rover_edge_time
from src.models.terrain import edge_steps, horizontal_lengths, horn_slope, neighbor_stack


def time_cost(ctx: CostContext) -> np.ndarray:
    """Travel time (s) of every edge; slope, step and (rover) hard limits
    already give inf.

    - walker: ``metabolic.edge_costs`` with ``eta = 1.0`` (eta enters through
      ``eta_cost``, so it is not counted twice; with the Tobler speed the time
      does not depend on eta anyway), steps from ``edge_steps(z)``,
      ``dM_suit = calibrate_dM_suit()``.
    - rover: ``rover.rover_edge_time`` with the steps, ``k_map = k_rocks``,
      ``ti = thermal_inertia`` and ``dune_mask = large_ripple_mask``.
    """
    steps = edge_steps(ctx.z)
    if is_walker(ctx.agent):
        t, _ = edge_costs(ctx.z, ctx.dx, ctx.dy, 1.0, ctx.agent, calibrate_dM_suit(),
                          steps=steps)
        return t
    return rover_edge_time(ctx.z, ctx.dx, ctx.dy, ctx.agent, k_map=ctx.k_rocks,
                           steps=steps, ti=ctx.thermal_inertia,
                           dune_mask=ctx.large_ripple_mask)


def eta_cost(ctx: CostContext, t_edge: np.ndarray) -> np.ndarray:
    """Extra time due to the terrain coefficient eta.

    OPTION B (TEAM ASSUMPTION): eta also slows the walker down, V_eff = V / eta,
    so the edge time is multiplied by eta_edge (mean of the two cells, as in
    ``edge_costs``). Term = t_edge * (eta_edge - 1) where t_edge is finite,
    inf where t_edge is inf.

    Zeros if ``eta_map`` is None or for a rover (large ripples are already
    obstacles in ``rover_edge_time``).
    """
    if ctx.eta_map is None or not is_walker(ctx.agent):
        return zeros_like_edges(ctx.z)
    t = np.asarray(t_edge, dtype=np.float64)
    eta_e = edge_mean(ctx.eta_map)
    finite = np.isfinite(t)
    return np.where(finite, np.where(finite, t, 0.0) * (np.where(finite, eta_e, 1.0) - 1.0),
                    np.inf)


def slope_cutoff(ctx: CostContext) -> np.ndarray:
    """0 / inf: edges forbidden by the agent's ``max_slope_deg``.

    Display term: ``time_cost`` already applies the same limit, so adding it
    to the total changes nothing. Edges leaving the grid are 0 here.
    - walker: edge slope atan(|dz| / dh), as in ``edge_costs``;
    - rover: Horn slope of the start or end cell, as in ``rover_edge_time``.
    """
    max_deg = ctx.agent.max_slope_deg
    if is_walker(ctx.agent):
        dh = horizontal_lengths(ctx.dx, ctx.dy)[:, None, None]
        dz = neighbor_stack(ctx.z) - ctx.z
        with np.errstate(invalid="ignore"):
            slope = np.degrees(np.arctan(np.abs(dz) / dh))
        return np.where(np.isfinite(slope) & (slope > max_deg), np.inf, 0.0)
    slope = horn_slope(ctx.z, ctx.dx, ctx.dy)
    return touching_inf(np.isfinite(slope) & (slope > max_deg))


def step_cutoff(ctx: CostContext) -> np.ndarray:
    """0 / inf: edges whose step (``terrain.edge_steps``) exceeds the agent's
    limits (walker: ``max_step_up_m`` / ``max_step_down_m``; rover:
    ``max_step_m`` both ways). Display term, already in ``time_cost``."""
    steps = edge_steps(ctx.z)
    if is_walker(ctx.agent):
        up, down = ctx.agent.max_step_up_m, ctx.agent.max_step_down_m
    else:
        up = down = ctx.agent.max_step_m
    finite = np.isfinite(steps)
    s = np.where(finite, steps, 0.0)
    return np.where(finite & ((s > up) | (s < -down)), np.inf, 0.0)
