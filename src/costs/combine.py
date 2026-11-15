"""Assemble every cost term into the total edge cost and plan a route.

Pass 1: ``time_cost`` (s). Pass 2: every other term; the ones that need times
use the effective time ``t_eff = time + eta term`` (with OPTION B the eta term
is real walking time). The total is the sum of all terms (inf propagates);
a NO-GO global gate makes the whole total inf.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.costs._edges import inf_like_edges, inside_edges
from src.costs.bridge import lonlat_to_rc, path_to_coordinates, path_to_rc, to_gridgraph
from src.costs.context import CostContext, CostWeights, is_walker
from src.costs.model_costs import (
    eva_budget_nogo,
    lander_distance_nogo,
    rock_cost,
    sep_isochrone_nogo,
    visibility_mask,
    visibility_penalty,
)
from src.costs.scalar_costs import (
    daylight_gate,
    dust_storm_gate,
    temperature_gate,
    thermal_inertia_nogo,
    wind_gate,
)
from src.costs.terrain_costs import eta_cost, slope_cutoff, step_cutoff, time_cost
from src.models.budget import outbound_time_map, return_time_map
from src.models.terrain import OFFSETS

KINDS = ("cost", "penalty", "spatial_nogo", "global_gate")


@dataclass
class CostTerm:
    name: str
    edge_cost: np.ndarray
    kind: str
    reason: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"CostTerm {self.name!r}: kind must be one of {KINDS}, "
                             f"got {self.kind!r}.")


@dataclass
class CostReport:
    total: np.ndarray
    terms: List[CostTerm]
    go: bool
    reasons: List[str] = field(default_factory=list)

    def term(self, name: str) -> CostTerm:
        return next(t for t in self.terms if t.name == name)


def _check_term(term: CostTerm, shape) -> None:
    a = term.edge_cost
    if a.shape != (8, *shape):
        raise ValueError(f"Cost term {term.name!r} has shape {a.shape}, "
                         f"expected {(8, *shape)}.")
    if np.isnan(a).any():
        raise ValueError(f"Cost term {term.name!r} contains {int(np.isnan(a).sum())} NaN.")
    if (a < 0).any():
        raise ValueError(f"Cost term {term.name!r} contains {int((a < 0).sum())} "
                         "negative values.")


def build_costs(ctx: CostContext, weights: Optional[CostWeights] = None) -> CostReport:
    """Compute every term, check it and sum them into the total edge cost."""
    w = weights or CostWeights()
    walker = is_walker(ctx.agent)

    # Pass 1 -------------------------------------------------------------
    t_time = time_cost(ctx)
    t_eta = eta_cost(ctx, t_time)
    t_eff = t_time + t_eta
    terms = [
        CostTerm("time", t_time, "cost", "travel time (s); slope/step/hard limits inf"),
        CostTerm("eta", t_eta, "cost",
                 "terrain eta, OPTION B: t * (eta_edge - 1)" if walker and ctx.eta_map is not None
                 else "zeros (no eta map, or rover: ripples already obstacles)"),
        CostTerm("slope_cutoff", slope_cutoff(ctx), "spatial_nogo",
                 f"slope > {ctx.agent.max_slope_deg} deg (already in time)"),
        CostTerm("step_cutoff", step_cutoff(ctx), "spatial_nogo",
                 "step beyond agent limits (already in time)"),
    ]

    # Pass 2 -------------------------------------------------------------
    if walker:
        t_ret = return_time_map(t_eff, ctx.lander_rc)
        t_out = outbound_time_map(t_eff, ctx.lander_rc)
        terms += [
            CostTerm("lander_distance", lander_distance_nogo(ctx), "spatial_nogo",
                     f"> {getattr(ctx.agent, 'max_distance_from_lander_m', None)} m "
                     "from the lander"),
            CostTerm("sep_isochrone", sep_isochrone_nogo(ctx, t_eff, t_ret=t_ret),
                     "spatial_nogo", f"return time > {ctx.agent.sep_return_min} min"),
            CostTerm("eva_budget", eva_budget_nogo(ctx, t_eff, t_ret=t_ret, t_out=t_out),
                     "spatial_nogo", f"t_out + t_ret > {ctx.agent.eva_nominal_h} h"),
        ]
    else:
        for name in ("lander_distance", "sep_isochrone", "eva_budget"):
            terms.append(CostTerm(name, np.zeros_like(t_time), "spatial_nogo",
                                  "walker only: zeros for a rover"))

    if w.visibility_penalty_weight is None:
        vis, vis_why = visibility_penalty(ctx, t_eff, w), "weight None: disabled"
    else:
        vis = visibility_penalty(ctx, t_eff, w, visible=visibility_mask(ctx))
        vis_why = f"{w.visibility_penalty_weight} * t on edges ending out of lander LOS"
    terms += [
        CostTerm("visibility", vis, "penalty", vis_why),
        CostTerm("rocks", rock_cost(ctx), "cost",
                 "zeros: rover rocks already in time (k_rocks); no walker data"),
        CostTerm("thermal_inertia", thermal_inertia_nogo(ctx), "spatial_nogo",
                 "rover TI < min_thermal_inertia (already in time)"
                 if ctx.thermal_inertia is not None and not walker else "zeros: no data or walker"),
    ]

    gates = [("dust_storm", dust_storm_gate), ("daylight", daylight_gate),
             ("temperature", temperature_gate), ("wind", wind_gate)]
    nogo_reasons = []
    for name, gate in gates:
        arr, why = gate(ctx, w)
        terms.append(CostTerm(name, arr, "global_gate", why))
        if np.isinf(arr).any():
            nogo_reasons.append(f"{name}: {why}")

    total = np.zeros_like(t_time)
    for term in terms:
        _check_term(term, ctx.shape)
        total = total + term.edge_cost  # inf + x = inf; no -inf anywhere

    go = not nogo_reasons
    if not go:
        total = inf_like_edges(ctx.z)
    return CostReport(total=total, terms=terms, go=go, reasons=nogo_reasons)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
@dataclass
class RouteResult:
    """``path_rc`` is None when there is no route (NO-GO, unreachable, start
    or end on an obstacle); ``reasons`` then says why."""

    path_rc: Optional[list]
    coordinates: Optional[list]
    cost: Optional[float]            # GridGraph cost (model cost + CELL_EPS per step)
    model_cost: Optional[float]      # sum of report.total along the path
    removed_edges: int
    reasons: List[str]

    @property
    def found(self) -> bool:
        return self.path_rc is not None


def path_edge_sum(edge_cost, path_rc) -> float:
    """Sum of a per-edge array along a ``[(row, col), ...]`` path."""
    if len(path_rc) < 2:
        return 0.0
    direction = {off: k for k, off in enumerate(OFFSETS)}
    p = np.asarray(path_rc)
    d = p[1:] - p[:-1]
    k = np.array([direction[(int(a), int(b))] for a, b in d])
    return float(np.asarray(edge_cost)[k, p[:-1, 0], p[:-1, 1]].sum())


def plan_route(report: CostReport, top_left_lonlat, step_deg, start_lonlat, end_lonlat
               ) -> RouteResult:
    """Shortest route on ``report.total``.

    NO-GO -> returns immediately with ``path_rc = None`` and the gate reasons.
    Otherwise ``bridge.to_gridgraph`` + ``GridGraph.shortest_path``.
    """
    if not report.go:
        return RouteResult(None, None, None, None, 0, list(report.reasons))
    shape = report.total.shape[1:]
    for p in (start_lonlat, end_lonlat):
        lonlat_to_rc(p, top_left_lonlat, step_deg, shape)  # ValueError if outside
    gg = to_gridgraph(report.total, top_left_lonlat, step_deg)
    try:
        path, cost = gg.shortest_path(tuple(start_lonlat), tuple(end_lonlat))
    except ValueError as exc:  # start or end on an obstacle
        return RouteResult(None, None, None, None, gg.removed_edges, [str(exc)])
    if not path:
        return RouteResult(None, None, None, None, gg.removed_edges,
                           ["no path between start and end with the current constraints"])
    path_rc = path_to_rc(path)
    return RouteResult(path_rc, path_to_coordinates(gg, path), float(cost),
                       path_edge_sum(report.total, path_rc), gg.removed_edges, [])


def summary(report: CostReport) -> dict:
    """Per-term overview for the demo: number of forbidden (inf) edges inside
    the grid, sum of the finite costs, kind and reason."""
    inside = inside_edges(report.total.shape[1:])
    out = {"go": report.go, "reasons": list(report.reasons), "terms": {}}
    for t in report.terms + [CostTerm("TOTAL", report.total, "cost", "")]:
        a = t.edge_cost
        fin = np.isfinite(a) & inside
        out["terms"][t.name] = {
            "kind": t.kind,
            "inf_edges": int((np.isinf(a) & inside).sum()),
            "finite_sum": float(a[fin].sum()),
            "reason": t.reason,
        }
    return out
