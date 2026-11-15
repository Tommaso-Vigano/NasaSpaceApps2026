"""Bridge between per-edge model costs and the routing ``GridGraph``.

Model side (``src.models``): per-edge arrays of shape ``(8, rows, cols)``,
``a[k, r, c]`` = cost of the edge ``(r, c) -> (r + drow_k, c + dcol_k)`` with
``OFFSETS[k] = (drow, dcol)`` from ``src.models.terrain``; inf = forbidden.

Routing side (``src.routing.algorithm.GridGraph``): nodes are ``(x, y) =
(col, row)``, ``DIRECTIONS[k] = (dx, dy)``. The two orders coincide:
``DIRECTIONS[k] == (OFFSETS[k][1], OFFSETS[k][0])`` for every k.

Coordinate orders used in this module:
- ``(lon, lat)``: geographic points (``top_left_lonlat``, GridGraph I/O);
- ``(row, col)``: model arrays and ``src.models.budget``;
- ``(x, y) = (col, row)``: GridGraph nodes and paths.
"""

import csv
import logging
from pathlib import Path

import numpy as np

from src.basic import Coordinate
from src.models.terrain import OFFSETS
from src.routing.algorithm import GridGraph

log = logging.getLogger(__name__)

CELL_EPS = 1e-9
"""Per-cell cost given to GridGraph for passable cells. GridGraph needs a
positive cell cost; the real cost is carried entirely by the slope_list."""

_ORTHOGONAL = [k for k, (dr, dc) in enumerate(OFFSETS) if dr == 0 or dc == 0]


def _validate_edge_cost(edge_cost) -> np.ndarray:
    ec = np.asarray(edge_cost, dtype=np.float64)
    if ec.ndim != 3 or ec.shape[0] != 8:
        raise ValueError(f"edge_cost must have shape (8, rows, cols), got {ec.shape}.")
    if np.isnan(ec).any():
        raise ValueError(
            f"edge_cost contains {int(np.isnan(ec).sum())} NaN values: "
            "use inf for forbidden edges."
        )
    if (ec < 0).any():
        raise ValueError(
            f"edge_cost contains {int((ec < 0).sum())} negative values: "
            "costs must be >= 0 (or inf)."
        )
    return ec


def _shift(a: np.ndarray, drow: int, dcol: int, fill) -> np.ndarray:
    """``out[r, c] = a[r + drow, c + dcol]``, ``fill`` outside the grid."""
    rows, cols = a.shape
    out = np.full_like(a, fill)
    r0, r1 = max(0, -drow), min(rows, rows - drow)
    c0, c1 = max(0, -dcol), min(cols, cols - dcol)
    out[r0:r1, c0:c1] = a[r0 + drow:r1 + drow, c0 + dcol:c1 + dcol]
    return out


def to_gridgraph(edge_cost, top_left_lonlat, step_deg, allow_diagonal=True) -> GridGraph:
    """Build a ``GridGraph`` whose arcs are exactly the finite model edges.

    :param edge_cost: per-edge costs, shape ``(8, rows, cols)``, indexed
        ``[k, row, col]`` with the ``OFFSETS`` convention; values >= 0 or inf
        (forbidden). NaN or negative values raise ``ValueError``.
    :param top_left_lonlat: ``(lon, lat)`` of cell ``(row=0, col=0)``.
    :param step_deg: cell size in degrees (same for lon and lat, as GridGraph).
    :param allow_diagonal: if False the 4 diagonal directions are ignored
        everywhere (node selection, slope_list and arc removal).
    :return: the GridGraph, with two extra attributes: ``removed_edges`` (number
        of arcs created by ``GridGraph._build_graph`` and removed because the
        model forbids them) and ``cell_eps``.

    Construction:
    - ``cost_matrix[row][col] = CELL_EPS`` for cells with at least one finite
      outgoing or incoming edge (among the allowed directions), inf otherwise
      (GridGraph treats inf cells as obstacles, i.e. not nodes);
    - ``slope_list`` = every finite edge as ``((x, y), (x2, y2), cost)`` with
      ``(x, y) = (col, row)``;
    - GridGraph links every pair of neighbouring nodes, so the arcs that are
      not in the slope_list (forbidden by the model) are removed afterwards.

    Path cost: GridGraph's arc weight is ``cell_cost(target) * (sqrt(2) if
    diagonal else 1) + slope_cost``, so the cost returned by
    ``shortest_path`` is the sum of the model edge costs plus ``CELL_EPS`` per
    step (``CELL_EPS * sqrt(2)`` on diagonal steps).
    """
    ec = _validate_edge_cost(edge_cost)
    _, rows, cols = ec.shape
    dirs = range(8) if allow_diagonal else _ORTHOGONAL

    # finite[k] = edge k is usable; edges leaving the grid are always dropped.
    finite = np.zeros(ec.shape, dtype=bool)
    inside = np.ones((rows, cols), dtype=bool)
    for k in dirs:
        drow, dcol = OFFSETS[k]
        finite[k] = np.isfinite(ec[k]) & _shift(inside, drow, dcol, False)

    # Nodes: at least one finite outgoing or incoming edge.
    node = np.zeros((rows, cols), dtype=bool)
    for k in dirs:
        drow, dcol = OFFSETS[k]
        node |= finite[k]                                # outgoing from (r, c)
        node |= _shift(finite[k], -drow, -dcol, False)   # incoming into (r, c)
    cost_matrix = np.where(node, CELL_EPS, np.inf)

    # slope_list and the list of arcs GridGraph will create but must not keep.
    slope_list = []
    to_remove = []
    for k in dirs:
        drow, dcol = OFFSETS[k]
        r, c = np.nonzero(finite[k])
        slope_list.extend(zip(zip(c.tolist(), r.tolist()),
                              zip((c + dcol).tolist(), (r + drow).tolist()),
                              ec[k, r, c].tolist()))
        # GridGraph arc exists iff both endpoints are nodes.
        both = node & _shift(node, drow, dcol, False)
        r, c = np.nonzero(both & ~finite[k])
        to_remove.extend(zip(zip(c.tolist(), r.tolist()),
                             zip((c + dcol).tolist(), (r + drow).tolist())))

    gg = GridGraph(cost_matrix.tolist(), tuple(top_left_lonlat), step_deg,
                   allow_diagonal=allow_diagonal, slope_list=slope_list)
    gg.graph.remove_edges_from(to_remove)
    gg.removed_edges = len(to_remove)
    gg.cell_eps = CELL_EPS
    log.info("to_gridgraph: %d nodes, %d arcs, %d forbidden arcs removed",
             gg.graph.number_of_nodes(), gg.graph.number_of_edges(), len(to_remove))
    return gg


def lonlat_to_rc(lonlat, top_left_lonlat, step_deg, shape) -> tuple:
    """Grid cell ``(row, col)`` of a geographic point.

    :param lonlat: ``(lon, lat)`` of the point.
    :param top_left_lonlat: ``(lon, lat)`` of cell ``(0, 0)``.
    :param shape: ``(rows, cols)`` of the grid.
    :return: ``(row, col)``, rounding to the nearest cell centre as
        ``GridGraph._to_node`` does. ``ValueError`` if outside the grid.
    """
    lon, lat = lonlat
    lon0, lat0 = top_left_lonlat
    rows, cols = shape
    col = int(round((lon - lon0) / step_deg))
    row = int(round((lat0 - lat) / step_deg))
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(
            f"Point (lon={lon}, lat={lat}) maps to cell (row={row}, col={col}), "
            f"outside the {rows}x{cols} grid."
        )
    return row, col


def path_to_rc(path) -> list:
    """Convert a GridGraph path ``[(x, y), ...]`` to ``[(row, col), ...]``
    (``x = col``, ``y = row``), e.g. for ``src.models.budget.check_eva``."""
    return [(int(y), int(x)) for x, y in path]


def path_to_coordinates(graph: GridGraph, path) -> list:
    """Convert a GridGraph path ``[(x, y), ...]`` to ``[Coordinate, ...]``.

    Uses ``graph.convert_idx_to_polar``, which returns ``(lon, lat)``; the
    Coordinates are built by keyword (``lat=``, ``lon=``).
    """
    return [Coordinate(lat=lat, lon=lon) for lon, lat in graph.convert_idx_to_polar(path)]


def write_path_csv(coords, path_csv) -> None:
    """Write ``[Coordinate, ...]`` as a CSV with header ``lat,lon`` (one point
    per row, in path order), the format read by
    ``scripts/demo_map.py --path-csv``."""
    with open(Path(path_csv), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon"])
        for c in coords:
            w.writerow([repr(float(c.lat)), repr(float(c.lon))])
