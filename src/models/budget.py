"""Whole-path constraints for an EVA: shortest-time maps, action radius,
SEP isochrone and point of no return.

Times are in seconds, energies in J. Per-edge arrays follow the convention of
``src.models.terrain`` (shape (8, rows, cols), inf = forbidden).
"""

import heapq
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.models.constraints import Astronaut
from src.models.terrain import OFFSETS, neighbor_stack


def dijkstra_grid(edge_cost, sources, reverse=False,
                  solver: Optional[Callable] = None) -> np.ndarray:
    """Minimum cost (rows, cols) on the 8-connected grid.

    reverse=False: cost FROM the nearest source TO every cell.
    reverse=True: cost FROM every cell TO the nearest source, following edges
    in their own direction. Costs are asymmetric, so when a cell u is expanded
    its neighbour v = u + OFFSETS[k] is relaxed with the cost of the edge
    v -> u, i.e. edge_cost[(k + 4) % 8, v] (OFFSETS[k + 4] = -OFFSETS[k]).

    Non-finite edge costs are skipped; unreachable cells are inf.

    This is a pure-Python heapq Dijkstra, O(N log N) with N = rows * cols:
    fine up to ~1e5-1e6 cells. For the demo use a crop of a few km around the
    lander at 5-20 m/px. ``solver`` is a hook for an optimised implementation
    with the same contract: ``solver(edge_cost, sources, reverse) -> array``.
    """
    if solver is not None:
        return np.asarray(solver(edge_cost, sources, reverse), dtype=np.float64)

    ec = np.asarray(edge_cost, dtype=np.float64)
    if ec.ndim != 3 or ec.shape[0] != 8:
        raise ValueError("edge_cost must have shape (8, rows, cols).")
    if np.any(ec < 0):
        raise ValueError("Dijkstra needs non-negative edge costs.")
    _, rows, cols = ec.shape
    n = rows * cols

    if reverse:
        # cost[k][u] = cost of the edge (u + OFFSETS[k]) -> u
        costs = np.stack([neighbor_stack(ec[(k + 4) % 8])[k] for k in range(8)])
    else:
        costs = ec
    costs = costs.reshape(8, n).tolist()

    index = np.arange(n, dtype=np.float64).reshape(rows, cols)
    nbr = neighbor_stack(index)
    nbr = np.where(np.isnan(nbr), -1, nbr).astype(np.int64).reshape(8, n).tolist()

    dist = [float("inf")] * n
    heap = []
    for r, c in sources:
        if not (0 <= r < rows and 0 <= c < cols):
            raise IndexError(f"Source {(r, c)} is outside the grid.")
        u = int(r) * cols + int(c)
        dist[u] = 0.0
        heap.append((0.0, u))
    heapq.heapify(heap)

    inf = float("inf")
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for k in range(8):
            v = nbr[k][u]
            if v < 0:
                continue
            w = costs[k][u]
            if not w < inf:  # skips inf and NaN
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return np.array(dist).reshape(rows, cols)


def return_time_map(edge_time, lander_rc, solver=None):
    """Minimum time (s) from every cell back to the lander."""
    return dijkstra_grid(edge_time, [lander_rc], reverse=True, solver=solver)


def outbound_time_map(edge_time, lander_rc, solver=None):
    """Minimum time (s) from the lander to every cell."""
    return dijkstra_grid(edge_time, [lander_rc], solver=solver)


def eva_reach_mask(t_out, t_ret, astronaut: Astronaut, dwell_s=0.0):
    """Action radius: cells reachable and returnable within the nominal EVA,
    t_out + dwell + t_ret <= eva_nominal_h [R11].

    The 2 km walkback limit (astronaut.max_distance_from_lander_m) is NOT
    applied here.
    """
    total = np.asarray(t_out) + dwell_s + np.asarray(t_ret)
    return total <= astronaut.eva_nominal_h * 3600.0


def sep_isochrone_mask(t_ret, astronaut: Astronaut):
    """Cells from which the lander is reached within sep_return_min [R20]."""
    return np.asarray(t_ret) <= astronaut.sep_return_min * 60.0


@dataclass
class EvaReport:
    """Per-point timeline of an EVA path (index k = k-th point of the path)."""

    t_elapsed: np.ndarray  # s, cumulative time at point k (egress included)
    t_ret: np.ndarray  # s, return time to the lander from point k
    nominal_ok: np.ndarray  # t_elapsed + t_ret <= eva_nominal_h
    contingency_ok: np.ndarray  # t_elapsed + f * t_ret <= eva_nominal_h + eva_margin_h
    point_of_no_return: Optional[int]  # last index before the first contingency failure
    energy_cum_J: np.ndarray  # J, cumulative walking energy at point k
    mean_power_W: float  # W, total energy / walking time
    sep_ok: np.ndarray  # t_ret <= sep_return_min
    feasible: bool  # all nominal_ok and total time + ingress <= eva_nominal_h


def check_eva(path_rc, edge_time, edge_energy, t_ret, astronaut: Astronaut,
              degraded_return_factor=1.5, egress_s=0.0, ingress_s=0.0) -> EvaReport:
    """Check a path (list of (row, col), consecutive points adjacent) against
    the EVA time budget [R11] and the SEP return time [R20].

    degraded_return_factor = 1.5 is a TEAM ASSUMPTION: in a contingency the
    walk back is 50% slower. ``point_of_no_return`` is None if every point
    satisfies the contingency rule, -1 if already the first point fails.
    Raises ValueError if two consecutive points are not adjacent.
    """
    path = [(int(r), int(c)) for r, c in path_rc]
    if not path:
        raise ValueError("Empty path.")
    direction = {off: k for k, off in enumerate(OFFSETS)}
    edge_time = np.asarray(edge_time, dtype=np.float64)
    edge_energy = np.asarray(edge_energy, dtype=np.float64)
    t_ret = np.asarray(t_ret, dtype=np.float64)

    n = len(path)
    t_elapsed = np.empty(n)
    energy = np.empty(n)
    t_elapsed[0] = egress_s
    energy[0] = 0.0
    for i in range(1, n):
        (r0, c0), (r1, c1) = path[i - 1], path[i]
        k = direction.get((r1 - r0, c1 - c0))
        if k is None:
            raise ValueError(
                f"Path points {i - 1} {path[i - 1]} and {i} {path[i]} are not adjacent."
            )
        t_elapsed[i] = t_elapsed[i - 1] + edge_time[k, r0, c0]
        energy[i] = energy[i - 1] + edge_energy[k, r0, c0]

    t_ret_path = np.array([t_ret[r, c] for r, c in path])
    t_nominal = astronaut.eva_nominal_h * 3600.0
    t_margin = astronaut.eva_margin_h * 3600.0

    nominal_ok = t_elapsed + t_ret_path <= t_nominal
    contingency_ok = t_elapsed + degraded_return_factor * t_ret_path <= t_nominal + t_margin
    bad = np.flatnonzero(~contingency_ok)
    pnr = None if bad.size == 0 else int(bad[0]) - 1

    walk_s = t_elapsed[-1] - egress_s
    if walk_s > 0 and np.isfinite(walk_s) and np.isfinite(energy[-1]):
        mean_power = float(energy[-1] / walk_s)
    else:
        mean_power = float("nan") if n > 1 else 0.0

    return EvaReport(
        t_elapsed=t_elapsed,
        t_ret=t_ret_path,
        nominal_ok=nominal_ok,
        contingency_ok=contingency_ok,
        point_of_no_return=pnr,
        energy_cum_J=energy,
        mean_power_W=mean_power,
        sep_ok=t_ret_path <= astronaut.sep_return_min * 60.0,
        feasible=bool(nominal_ok.all() and t_elapsed[-1] + ingress_s <= t_nominal),
    )
