"""Corrected, parallel version of the PenaltyManager (src/app/penalty/penalty_manager.py).

Same structure and the same public names as Tommaso's version (it subclasses it):
register_penalty, select_cost_function, _construct_cost_pipeline, build_from,
build_slopes_list, select_slope_cost_function, plus build_graph, which builds the
GridGraph ready for shortest_path. The original file is NOT modified: this one is used
for tests and for the demo, so the two can be compared side by side.

It only depends on code that is already on main (penalty_manager, ProgressiveCutoff,
profiles, GridGraph, Map, Datatype) and on src/models/constraints.py.

Unit: "cell equivalents". 1 = crossing one flat cell on rock. Cost of an arc A -> B in
GridGraph = cell_cost(B) * L + slope_cost(A -> B), with L = 1 straight, sqrt(2) diagonal.
Here cell_cost = BASE_COST + sum of the cell penalties, and the slope cost is already
multiplied by L.

Problems found in the original (each one checked by running it on small cases) and fixes:

| #  | Original                                                       | Fix here                                      |
|----|----------------------------------------------------------------|-----------------------------------------------|
| 1  | GridGraph drops cells with cost 0: with all penalties 0 the    | BASE_COST = 1 added to every cell             |
|    | graph is empty (5x5 zeros -> 0 nodes out of 25)                |                                               |
| 2  | Arcs whose slope cost is inf stay in the graph: the path       | arcs not in the finite slope list are removed |
|    | crosses a 63 deg wall and comes out with cost inf              | from the graph after it is built              |
| 3  | A cell with NaN elevation is still a node, slope cost 0:       | NaN elevation -> cell cost inf (obstacle)     |
|    | missing data becomes a shortcut                                |                                               |
| 4  | Slope limit read from HumanProfile.hard_constraints["slope"]   | limits from src/models/constraints.py         |
|    | = 30 deg instead of 20                                         | (Astronaut 20, RoboticRover 30, with sources) |
| 5  | Slope cost independent of the arc length (diagonal = straight) | slope cost * sqrt(2) on diagonal arcs.        |
|    | and the same uphill and downhill                               | Symmetry kept (ProgressiveCutoff): known limit|
| 7  | Every template gives inf on NaN: missing data = obstacle       | per-Datatype NaN fill (nan_fill); default:    |
|    | (e.g. outside the NOAH-H coverage everything is forbidden)     | TERRAIN_ETA -> ETA_TERRAIN["regolith"]        |
| 8  | A Datatype with no registered penalty stops everything         | ignored_datatypes (default ALBEDO, DUST,      |
|    | (ALBEDO -> NotImplementedError)                                | RADIATION); ELEVATION is used by the slope    |
|    |                                                                | list and skipped if it has no cell penalty;   |
|    |                                                                | anything else -> ValueError listing the       |
|    |                                                                | registered ones                               |
| 9  | The sum does not check the grids (2x2 + 1x2 summed by          | same shape, granularity and top_left required |
|    | broadcasting, no error)                                        | (tolerance 1e-9); NaN in a penalty -> error   |
| 11 | Python loops + haversine with R = 3 389 500 m                  | numpy; dx = R*dlon*cos(lat), dy = R*dlat with |
|    |                                                                | R = MARS_RADIUS_M = 3 396 190 m (data sphere) |

(#6 lower cut-off and #10 uniform MCD maps are about the templates and the configuration,
not about this class.)

TEAM ASSUMPTIONS: BASE_COST = 1; slope weights 2.0 (walker: with Tobler the time grows by
85% at 10 deg and by 258% at 20 deg, the line 2*|s|/20 sits in between) and 0.5 (rover:
speed does not depend on the slope). Values in src/constants.py.
"""

import json
import math
from typing import Iterable, Optional

import numpy as np

from src.app.penalty.implementations.penalty import Penalty
from src.app.penalty.implementations.progressive_cutoff import ProgressiveCutoff
from src.app.penalty.penalty_manager import PenaltyComponents, PenaltyManager
from src.constants import (
    BASE_COST,
    DEFAULT_IGNORED_DATATYPE_NAMES,
    DEFAULT_SLOPE_WEIGHTS,
    GRID_ATOL,
    SQRT2,
)
from src.data.map import Map
from src.data.types import Datatype
from src.models.constraints import ETA_TERRAIN, MARS_RADIUS_M, Astronaut, RoboticRover
from src.routing.algorithm import DIRECTIONS, GridGraph
from src.routing.profiles import ExpeditionProfile, HumanProfile, Profile, RoverProfile

def _default_ignored() -> set:
    """Datatypes of DEFAULT_IGNORED_DATATYPE_NAMES (src/constants.py) that exist."""
    return {dt for dt in (getattr(Datatype, n, None) for n in DEFAULT_IGNORED_DATATYPE_NAMES)
            if dt is not None}


def _default_nan_fill() -> dict:
    """TERRAIN_ETA outside the NOAH-H coverage -> regolith (not an obstacle)."""
    eta = getattr(Datatype, "TERRAIN_ETA", None)
    return {} if eta is None else {eta: ETA_TERRAIN["regolith"]}


def _participants(profile: Profile) -> list:
    """The single profiles of a profile (an expedition is expanded)."""
    if isinstance(profile, ExpeditionProfile):
        if not profile.profiles:
            raise ValueError("An expedition must contain at least one participant.")
        out = []
        for member in profile.profiles:
            out.extend(_participants(member))
        return out
    if isinstance(profile, (HumanProfile, RoverProfile)):
        return [profile]
    raise TypeError(f"Unsupported profile: {type(profile).__name__}")


def _kind(profile: Profile) -> str:
    return "walker" if isinstance(profile, HumanProfile) else "rover"


class FixedPenaltyComponents(PenaltyComponents):
    """Component penalty maps on the same grid, summed with BASE_COST."""

    def __init__(self, maps: Iterable[Map], base_cost: float = BASE_COST,
                 reference: Optional[Map] = None):
        super().__init__(maps)
        self.base_cost = float(base_cost)
        # Grid to use when no component map is left (e.g. all of them ignored).
        self.reference = reference

    @staticmethod
    def _check_same_grid(reference: Map, other: Map) -> None:
        if np.shape(other.values) != np.shape(reference.values):
            raise ValueError(
                f"Penalty map {other.title!r} has shape {np.shape(other.values)}, "
                f"expected {np.shape(reference.values)} as {reference.title!r}."
            )
        for name in ("granularity", "top_left", "bottom_right"):
            a = np.asarray(getattr(reference, name), dtype=float)
            b = np.asarray(getattr(other, name), dtype=float)
            if not np.allclose(a, b, rtol=0.0, atol=GRID_ATOL):
                raise ValueError(
                    f"Penalty map {other.title!r} has {name} {tuple(b)}, "
                    f"expected {tuple(a)} as {reference.title!r}: resample them on the "
                    "same grid (WorldModel.make_slice) first."
                )

    def get_total_penalty(self) -> Map:
        """BASE_COST + sum of the component maps, after grid and NaN checks."""
        reference = self.maps[0] if self.maps else self.reference
        if reference is None:
            raise ValueError("No penalty maps to sum and no reference grid.")

        total = np.full(np.shape(reference.values), self.base_cost, dtype=np.float64)
        components = []
        for penalty_map in self.maps:
            self._check_same_grid(reference, penalty_map)
            values = np.asarray(penalty_map.values, dtype=np.float64)
            if np.isnan(values).any():
                raise ValueError(
                    f"Penalty map {penalty_map.title!r} contains "
                    f"{int(np.isnan(values).sum())} NaN values."
                )
            if (values < 0).any():
                raise ValueError(f"Penalty map {penalty_map.title!r} has negative values.")
            total += values
            components.append(penalty_map.title)

        return Map(
            values=total,
            granularity=tuple(reference.granularity),
            top_left=tuple(reference.top_left),
            bottom_right=tuple(reference.bottom_right),
            metadata=json.dumps({
                "coordinate_units": "degrees",
                "value_unit": "cell-equivalents",
                "base_cost": self.base_cost,
                "components": components,
            }),
            value_type=Datatype.PENALTY,
            title="Total penalty",
        )


class FixedPenaltyManager(PenaltyManager):
    """Same interface as PenaltyManager, with the fixes listed in the module docstring."""

    def __init__(self, ignored_datatypes: Optional[Iterable[Datatype]] = None,
                 nan_fill: Optional[dict] = None,
                 slope_weights: Optional[dict] = None,
                 base_cost: float = BASE_COST):
        super().__init__()
        self.ignored_datatypes = set(_default_ignored() if ignored_datatypes is None
                                     else ignored_datatypes)
        self.nan_fill = _default_nan_fill() if nan_fill is None else dict(nan_fill)
        self.slope_weights = dict(DEFAULT_SLOPE_WEIGHTS if slope_weights is None
                                  else slope_weights)
        self.base_cost = float(base_cost)
        self.forbidden_arcs = 0

    # ------------------------------------------------------------------ cells
    def select_cost_function(self, profile: Profile, data_type: Datatype) -> Penalty:
        """As the original, with a clear error instead of NotImplementedError."""
        if data_type not in self.penalties:
            registered = sorted(dt.name for dt in self.penalties)
            raise ValueError(
                f"No penalty registered for {data_type.name}. Registered: {registered}; "
                f"ignored: {sorted(dt.name for dt in self.ignored_datatypes)}."
            )
        return self.penalties[data_type](profile)

    def _construct_cost_pipeline(self, profile: Profile,
                                 world_slice: Iterable[Map]) -> FixedPenaltyComponents:
        """One penalty map per layer; ignored layers skipped, NaN filled per Datatype."""
        maps = list(world_slice)
        penalty_maps = []
        for source_map in maps:
            data_type = source_map.value_type
            if data_type in self.ignored_datatypes:
                continue
            if data_type == Datatype.ELEVATION and data_type not in self.penalties:
                continue  # the elevation is used by the slope list
            penalty = self.select_cost_function(profile, data_type)

            values = np.asarray(source_map.values, dtype=np.float64)
            if data_type in self.nan_fill:
                values = np.where(np.isnan(values), self.nan_fill[data_type], values)
            costs = np.asarray(penalty.compute(values), dtype=np.float64)
            costs = np.broadcast_to(costs, values.shape).copy()

            penalty_maps.append(Map(
                values=costs,
                granularity=tuple(source_map.granularity),
                top_left=tuple(source_map.top_left),
                bottom_right=tuple(source_map.bottom_right),
                metadata=json.dumps({
                    "coordinate_units": "degrees",
                    "value_unit": "cell-equivalents",
                    "penalty": type(penalty).__name__,
                    "source_title": source_map.title,
                    "nan_filled_with": self.nan_fill.get(data_type),
                }),
                value_type=data_type,
                title=f"{source_map.title} - penalty",
            ))
        return FixedPenaltyComponents(penalty_maps, self.base_cost,
                                      reference=maps[0] if maps else None)

    def build_from(self, profile: Profile, world_slice: Iterable[Map]) -> Map:
        """BASE_COST + sum of the component penalty maps (cell-equivalents)."""
        return self._construct_cost_pipeline(profile, world_slice).get_total_penalty()

    # ----------------------------------------------------------------- slopes
    def _get_slope_limit(self, profile: Profile) -> float:
        """Max slope in degrees from src/models/constraints.py (the strictest member
        for an expedition), NOT from the profiles' hard_constraints dicts."""
        limits = [Astronaut().max_slope_deg if _kind(p) == "walker"
                  else RoboticRover().max_slope_deg for p in _participants(profile)]
        return float(min(limits))

    def select_slope_cost_function(self, profile: Profile) -> Penalty:
        """ProgressiveCutoff(limit, weight): weight * |slope| / limit, inf at the limit.
        For an expedition: the strictest limit and the largest weight."""
        weight = max(self.slope_weights[_kind(p)] for p in _participants(profile))
        return ProgressiveCutoff(threshold=self._get_slope_limit(profile), weight=weight)

    def slope_costs(self, profile: Profile, elevation_map: Map) -> np.ndarray:
        """Per-arc slope cost, shape (8, rows, cols), in the order of
        src.routing.algorithm.DIRECTIONS ((dx, dy) = (dcol, drow)).
        NaN = arc not usable (leaves the grid or touches a NaN cell);
        inf = forbidden by the slope limit."""
        z = np.asarray(elevation_map.values, dtype=np.float64)
        rows, cols = z.shape
        dlon, dlat = (float(v) for v in elevation_map.granularity)
        lat_rows = np.radians(elevation_map.top_left[1] - np.arange(rows) * dlat)
        dy = MARS_RADIUS_M * math.radians(dlat)
        dx_rows = MARS_RADIUS_M * math.radians(dlon) * np.cos(lat_rows)  # (rows,)

        penalty = self.select_slope_cost_function(profile)
        out = np.full((8, rows, cols), np.nan)
        for k, (dc, dr) in enumerate(DIRECTIONS):
            r0, r1 = max(0, -dr), min(rows, rows - dr)
            c0, c1 = max(0, -dc), min(cols, cols - dc)
            src = z[r0:r1, c0:c1]
            dst = z[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            # horizontal length with dx at the mean latitude of the two rows
            dx_mid = 0.5 * (dx_rows[r0:r1] + dx_rows[r0 + dr:r1 + dr])[:, None]
            length = np.hypot(dc * dx_mid, dr * dy)
            slope = np.degrees(np.arctan2(dst - src, length))
            valid = np.isfinite(slope)
            cost = np.full(slope.shape, np.nan)
            cost[valid] = np.asarray(penalty.compute(slope[valid]), dtype=np.float64)
            cost *= SQRT2 if (dc and dr) else 1.0
            out[k, r0:r1, c0:c1] = cost
        return out

    def build_slopes_list(self, profile: Profile, ElevationMap: Map) -> list:
        """[(from, to, cost)] with nodes (x=col, y=row), only finite arcs.
        Forbidden arcs (inf) are counted in self.forbidden_arcs and left out."""
        costs = self.slope_costs(profile, ElevationMap)
        self.forbidden_arcs = int(np.isinf(costs).sum())
        out = []
        for k, (dc, dr) in enumerate(DIRECTIONS):
            r, c = np.nonzero(np.isfinite(costs[k]))
            out.extend(zip(zip(c.tolist(), r.tolist()),
                           zip((c + dc).tolist(), (r + dr).tolist()),
                           costs[k, r, c].tolist()))
        return out

    # ------------------------------------------------------------------ graph
    def build_graph(self, profile: Profile, world_slice: Iterable[Map]) -> GridGraph:
        """GridGraph ready for shortest_path: cell costs from build_from (NaN elevation
        = obstacle), arc costs from build_slopes_list, and the arcs that GridGraph
        creates by itself but that are not allowed removed."""
        maps = list(world_slice)
        elevations = [m for m in maps if m.value_type == Datatype.ELEVATION]
        if len(elevations) != 1:
            raise ValueError(f"Expected exactly one ELEVATION map, got {len(elevations)}.")
        elevation = elevations[0]
        dlon, dlat = (float(v) for v in elevation.granularity)
        if not math.isclose(dlon, dlat, rel_tol=0.0, abs_tol=GRID_ATOL):
            raise ValueError(f"GridGraph needs square cells in degrees, got {(dlon, dlat)}.")

        cell = np.array(self.build_from(profile, maps).values, dtype=np.float64)
        cell[~np.isfinite(np.asarray(elevation.values, dtype=np.float64))] = np.inf

        slopes = self.build_slopes_list(profile, elevation)
        graph = GridGraph(cell.tolist(), tuple(elevation.top_left), dlon,
                          slope_list=slopes)

        allowed = {(a, b) for a, b, _ in slopes}
        to_remove = [e for e in graph.graph.edges() if e not in allowed]
        graph.graph.remove_edges_from(to_remove)
        graph.removed_arcs = len(to_remove)
        graph.forbidden_arcs = self.forbidden_arcs
        return graph
