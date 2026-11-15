"""Cost terms from the whole-path models (``src.models.budget``, ``viewshed``).

They need the edge times, so ``combine.build_costs`` computes them in a second
pass. ``t_edge`` is the effective edge time (s), shape ``(8, *z.shape)``.
Spatial NO-GO terms put inf on the edges ENTERING a forbidden cell.
"""

from typing import Optional

import numpy as np

from src.costs._edges import entering_inf, target_mask, zeros_like_edges
from src.costs.context import CostContext, CostWeights, is_walker
from src.models.budget import eva_reach_mask, outbound_time_map, return_time_map
from src.models.viewshed import viewshed


def lander_distance_map(ctx: CostContext) -> np.ndarray:
    """Horizontal distance (m) of every cell from the lander."""
    rows = np.arange(ctx.shape[0])[:, None] - ctx.lander_rc[0]
    cols = np.arange(ctx.shape[1])[None, :] - ctx.lander_rc[1]
    return np.hypot(rows * ctx.dy, cols * ctx.dx)


def lander_distance_nogo(ctx: CostContext) -> np.ndarray:
    """inf on edges entering cells farther than
    ``agent.max_distance_from_lander_m`` from the lander [R11].
    Walker only (zeros for a rover or if the attribute is missing)."""
    limit = getattr(ctx.agent, "max_distance_from_lander_m", None)
    if not is_walker(ctx.agent) or limit is None:
        return zeros_like_edges(ctx.z)
    return entering_inf(lander_distance_map(ctx) > limit)


def sep_isochrone_nogo(ctx: CostContext, t_edge, t_ret: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """inf on edges entering cells whose return time to the lander exceeds
    ``agent.sep_return_min`` minutes [R20]. Walker only.

    ``t_ret`` can be passed to reuse an already computed ``return_time_map``.
    """
    if not is_walker(ctx.agent):
        return zeros_like_edges(ctx.z)
    if t_ret is None:
        t_ret = return_time_map(t_edge, ctx.lander_rc)
    return entering_inf(t_ret > ctx.agent.sep_return_min * 60.0)


def eva_budget_nogo(ctx: CostContext, t_edge, t_ret: Optional[np.ndarray] = None,
                    t_out: Optional[np.ndarray] = None) -> np.ndarray:
    """inf on edges entering cells outside the action radius
    t_out + t_ret <= agent.eva_nominal_h [R11] (``budget.eva_reach_mask``).
    Walker only."""
    if not is_walker(ctx.agent):
        return zeros_like_edges(ctx.z)
    if t_ret is None:
        t_ret = return_time_map(t_edge, ctx.lander_rc)
    if t_out is None:
        t_out = outbound_time_map(t_edge, ctx.lander_rc)
    return entering_inf(~eva_reach_mask(t_out, t_ret, ctx.agent))


def visibility_mask(ctx: CostContext) -> np.ndarray:
    """Cells in line of sight of the lander (``viewshed.viewshed``).

    The range covers the whole grid and the number of rays keeps the ray
    spacing below one cell at that range: both are derived from the grid,
    not fixed. ``target_h = agent.eye_height_m`` if the agent has it,
    otherwise the ``viewshed`` default.
    """
    rows, cols = ctx.shape
    max_range = float(np.hypot(rows * ctx.dy, cols * ctx.dx))
    n_rays = max(int(np.ceil(2.0 * np.pi * max_range / min(ctx.dx, ctx.dy))), 1)
    kwargs = {"max_range_m": max_range, "n_rays": n_rays}
    if hasattr(ctx.agent, "eye_height_m"):
        kwargs["target_h"] = ctx.agent.eye_height_m
    return viewshed(ctx.z, ctx.dx, ctx.dy, ctx.lander_rc, **kwargs)


def visibility_penalty(ctx: CostContext, t_edge, weights: Optional[CostWeights] = None,
                       visible: Optional[np.ndarray] = None) -> np.ndarray:
    """Penalty (not a prohibition): ``weight * t_edge`` on the edges that end
    out of line of sight of the lander, 0 elsewhere (also where t_edge is inf,
    which is already forbidden). Zeros if ``visibility_penalty_weight`` is None.
    """
    w = (weights or CostWeights()).visibility_penalty_weight
    if w is None:
        return zeros_like_edges(ctx.z)
    if visible is None:
        visible = visibility_mask(ctx)
    t = np.asarray(t_edge, dtype=np.float64)
    hit = target_mask(~visible) & np.isfinite(t)
    return np.where(hit, w * np.where(hit, t, 0.0), 0.0)


def rock_cost(ctx: CostContext) -> np.ndarray:
    """Always zeros.

    Rover: rocks are already in ``rover_edge_time`` through ``k_rocks`` (the
    (1 + p_rock) factor). Walker: there is no rock model / data for suited
    walking, so no term is added (counting it here would double the rover's).
    """
    return zeros_like_edges(ctx.z)
