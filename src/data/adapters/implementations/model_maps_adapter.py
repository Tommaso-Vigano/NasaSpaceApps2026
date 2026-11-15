"""Maps computed by the models of src/models on the grid of an ELEVATION Map.

ModelMapsAdapter(lander_lonlat, agent=Astronaut(), eta_map=None, steps_k=5)
    .adapt(elevation_map) -> list[Map]

Every returned Map has exactly the grid of the ELEVATION Map (same corners, same
granularity, same shape), so it can go straight into FixedPenaltyManager together with it.

Walker (Astronaut):
    STEP_HEIGHT      m    max |edge_steps(z, k, compensate=True)| over the 8 arcs of the cell;
                          with step_dem (a fine DEM, e.g. HiRISE 1 m), the max over the fine
                          cells inside each cell. Use step_dem on grids coarser than a few m:
                          a step is a metre-scale feature (see __init__).
    LANDER_DISTANCE  m    horizontal distance from the lander (dx, dy from terrain.from_map)
    RETURN_TIME      s    minimum walking time from the cell back to the lander
    EVA_TIME         s    lander -> cell -> lander (outbound + return), no dwell time
    VISIBILITY       0/1  1 - viewshed from the lander (1 = out of sight)
Rover (RoboticRover, CrewedRover): only STEP_HEIGHT. The walk limits do not apply to a
robotic rover, and its time model (rover.rover_edge_time) is not used for routing yet.

Times: metabolic.edge_costs (Tobler speed, slope and step limits of the astronaut) and
budget.return_time_map / outbound_time_map (Dijkstra on the 8-connected grid). If
eta_map is given, the time of every arc is multiplied by the mean eta of its two cells
(option B, TEAM ASSUMPTION: the terrain slows the walk in proportion to Pandolf's eta),
consistent with the TERRAIN_ETA penalty eta - 1 of penalty_config. Unreachable cells = inf.

Dijkstra is pure Python (budget.dijkstra_grid): a few seconds for ~1e5 cells.
"""

import json
import math
from typing import Optional, Union

import numpy as np

from src.constants import GRID_ATOL, LANDER_ANTENNA_H_M, MAX_VIEWSHED_SAMPLES
from src.data.adapters.adapter import Adapter
from src.data.adapters.registry import AdapterRegistry
from src.data.map import Map
from src.data.types import Datatype
from src.models.budget import outbound_time_map, return_time_map
from src.models.constraints import ETA_TERRAIN, Astronaut
from src.models.metabolic import calibrate_dM_suit, edge_costs
from src.models.terrain import edge_steps, from_map, neighbor_stack
from src.models.viewshed import viewshed

# GRID_ATOL, MAX_VIEWSHED_SAMPLES (viewshed memory bound) and LANDER_ANTENNA_H_M
# (TEAM ASSUMPTION, also the default of viewshed.viewshed) are in src/constants.py.


@AdapterRegistry.register_adapter("Jezero-model-maps")
class ModelMapsAdapter(Adapter):
    """Model-derived Maps (step height, lander distance, return / EVA time,
    visibility) on the grid of an ELEVATION Map. See the module docstring."""

    def __init__(self, lander_lonlat, agent=None,
                 eta_map: Optional[Union[Map, np.ndarray]] = None, steps_k: int = 5,
                 observer_h: float = LANDER_ANTENNA_H_M, step_dem: Optional[Map] = None,
                 chunk_rows: int = 256):
        """step_dem: optional fine ELEVATION Map (e.g. the 1 m HiRISE DEM) covering the
        grid. If given, STEP_HEIGHT is the max |step| of the fine cells inside each
        coarse cell, and the walking times forbid the arcs touching a cell whose
        STEP_HEIGHT is >= agent.max_step_up_m (same rule as the CutOff of
        penalty_config) instead of using coarse per-arc steps. Reason: a step is a
        metre-scale feature. On a 20 m grid, edge_steps measures the undulation of
        the terrain over the 5 x 5 window (100 m), not steps: on the Jezero lander
        DEM, |step| > 0.30 m in 0.02% of the 1 m cells but in 92% of the 20 m cells."""
        lon, lat = (float(v) for v in lander_lonlat)
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError("lander_lonlat must be finite (longitude, latitude) degrees.")
        if int(steps_k) != steps_k or steps_k < 1:
            raise ValueError("steps_k must be a positive integer.")
        self.lander_lonlat = (lon, lat)
        self.agent = Astronaut() if agent is None else agent
        self.eta_map = eta_map
        self.steps_k = int(steps_k)
        self.observer_h = float(observer_h)
        if step_dem is not None and step_dem.value_type != Datatype.ELEVATION:
            raise ValueError("step_dem must be an ELEVATION Map.")
        self.step_dem = step_dem
        self.chunk_rows = max(1, int(chunk_rows))

    @property
    def is_walker(self) -> bool:
        return isinstance(self.agent, Astronaut)

    # --------------------------------------------------------------- helpers
    def lander_rc(self, m: Map) -> tuple:
        """(row, col) of the cell containing the lander; ValueError if outside / NaN."""
        dlon, dlat = (float(v) for v in m.granularity)
        lon, lat = self.lander_lonlat
        row = int(round((m.top_left[1] - lat) / dlat))
        col = int(round((lon - m.top_left[0]) / dlon))
        rows, cols = np.shape(m.values)
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(
                f"Lander (lon={lon}, lat={lat}) maps to cell (row={row}, col={col}), "
                f"outside the {rows}x{cols} elevation grid."
            )
        if not np.isfinite(m.values[row, col]):
            raise ValueError(f"Lander cell (row={row}, col={col}) has no elevation (NaN).")
        return row, col

    def _eta_values(self, m: Map) -> Optional[np.ndarray]:
        """eta on the elevation grid, NaN filled with regolith (TEAM ASSUMPTION, the
        same NaN policy as FixedPenaltyManager for TERRAIN_ETA)."""
        if self.eta_map is None:
            return None
        if isinstance(self.eta_map, Map):
            _check_same_grid(m, self.eta_map)
            eta = np.asarray(self.eta_map.values, dtype=np.float64)
        else:
            eta = np.asarray(self.eta_map, dtype=np.float64)
            if eta.shape != np.shape(m.values):
                raise ValueError(f"eta_map has shape {eta.shape}, "
                                 f"expected {np.shape(m.values)}.")
        return np.where(np.isnan(eta), ETA_TERRAIN["regolith"], eta)

    def _map(self, m: Map, values, data_type: Datatype, unit: str, source: str,
             description: str, extra: Optional[dict] = None) -> Map:
        meta = {
            "coordinate_units": "degrees",
            "value_unit": unit,
            "source": source,
            "description": description,
            "lander_lonlat": list(self.lander_lonlat),
            "agent": type(self.agent).__name__,
            "elevation_title": m.title,
        }
        meta.update(extra or {})
        return Map(
            values=np.asarray(values, dtype=np.float64),
            granularity=tuple(m.granularity),
            top_left=tuple(m.top_left),
            bottom_right=tuple(m.bottom_right),
            metadata=json.dumps(meta),
            value_type=data_type,
            title=f"{data_type.value} ({type(self.agent).__name__})",
        )

    # ----------------------------------------------------------------- layers
    def step_height(self, z: np.ndarray) -> np.ndarray:
        """Max |step| over the 8 arcs of each cell; NaN if the cell is NaN (or has
        no valid arc)."""
        steps = np.abs(edge_steps(z, self.steps_k, compensate=True))
        out = np.where(np.isfinite(steps), steps, -np.inf).max(axis=0)
        return np.where(np.isfinite(out) & np.isfinite(z), out, np.nan)

    def fine_step_height(self, m: Map) -> np.ndarray:
        """Max |step| of the fine cells of self.step_dem inside each cell of ``m``
        (block maximum). Fine cells are assigned to the coarse cell whose centre is
        nearest; the fine DEM is processed in chunks of rows (with a halo of k // 2 + 1
        rows, so the chunks give the same steps as the whole DEM). Coarse cells with no
        valid fine cell, or with a NaN elevation, are NaN."""
        fine = self.step_dem
        zf = np.asarray(fine.values, dtype=np.float64)
        rows, cols = np.shape(m.values)
        gx, gy = (float(v) for v in m.granularity)
        fgx, fgy = (float(v) for v in fine.granularity)
        if fgx > gx or fgy > gy:
            raise ValueError("step_dem must be at least as fine as the elevation grid.")

        # coarse index of every fine row / column (monotonic non-decreasing)
        ci = np.rint((fine.top_left[0] + np.arange(zf.shape[1]) * fgx - m.top_left[0]) / gx)
        ri = np.rint((m.top_left[1] - (fine.top_left[1] - np.arange(zf.shape[0]) * fgy)) / gy)
        ci, ri = ci.astype(np.int64), ri.astype(np.int64)
        jv = np.flatnonzero((ci >= 0) & (ci < cols))
        iv = np.flatnonzero((ri >= 0) & (ri < rows))
        out = np.full((rows, cols), -np.inf)
        if jv.size == 0 or iv.size == 0:
            raise ValueError("step_dem does not overlap the elevation grid.")
        j0, j1 = jv[0], jv[-1] + 1
        col_ids = ci[j0:j1]
        col_starts = np.flatnonzero(np.r_[True, np.diff(col_ids) != 0])

        halo = self.steps_k // 2 + 1
        for a in range(iv[0], iv[-1] + 1, self.chunk_rows):
            b = min(a + self.chunk_rows, iv[-1] + 1)
            lo, hi = max(0, a - halo), min(zf.shape[0], b + halo)
            # columns: the whole valid range plus a halo (edge_steps needs neighbours)
            cl, ch = max(0, j0 - halo), min(zf.shape[1], j1 + halo)
            z = zf[lo:hi, cl:ch]
            s = np.abs(edge_steps(z, self.steps_k, compensate=True))
            s = np.where(np.isfinite(s), s, -np.inf).max(axis=0)
            s = np.where(np.isfinite(z), s, -np.inf)[a - lo:b - lo, j0 - cl:j1 - cl]
            colmax = np.maximum.reduceat(s, col_starts, axis=1)
            row_ids = ri[a:b]
            row_starts = np.flatnonzero(np.r_[True, np.diff(row_ids) != 0])
            block = np.maximum.reduceat(colmax, row_starts, axis=0)
            rr, cc = row_ids[row_starts], col_ids[col_starts]
            out[np.ix_(rr, cc)] = np.maximum(out[np.ix_(rr, cc)], block)

        z_coarse = np.asarray(m.values, dtype=np.float64)
        return np.where(np.isfinite(out) & np.isfinite(z_coarse), out, np.nan)

    @staticmethod
    def lander_distance(shape, dx, dy, rc) -> np.ndarray:
        rows, cols = shape
        dr = (np.arange(rows) - rc[0])[:, None] * dy
        dc = (np.arange(cols) - rc[1])[None, :] * dx
        return np.hypot(dr, dc)

    def edge_times(self, z, dx, dy, eta, step_cell=None) -> np.ndarray:
        """Walking time (s) of every arc, (8, rows, cols); inf = forbidden.

        step_cell None: step limits per arc from edge_steps on this grid.
        step_cell given (fine STEP_HEIGHT per cell, see step_dem): arcs touching a
        cell with step_cell >= max_step_up_m (or NaN) are forbidden."""
        a = self.agent
        dM = calibrate_dM_suit(a.walkback_power_W, a.v_nominal_ms, a.body_mass_kg,
                               a.suit_mass_kg)
        steps = edge_steps(z, self.steps_k, compensate=True) if step_cell is None else None
        time_s, _ = edge_costs(z, dx, dy, 1.0 if eta is None else eta, a, dM, steps=steps)
        if step_cell is not None:
            bad = ~(np.asarray(step_cell, dtype=np.float64) < a.max_step_up_m)
            bad_arc = bad[None] | (neighbor_stack(bad.astype(np.float64)) != 0.0)
            time_s = np.where(bad_arc, np.inf, time_s)
        if eta is not None:  # option B: time x mean eta of the arc
            eta_arc = 0.5 * (eta + neighbor_stack(eta))
            time_s = np.where(np.isfinite(time_s), time_s * eta_arc, np.inf)
        return time_s

    def visibility(self, z, dx, dy, rc) -> tuple:
        """1 - viewshed; range = farthest cell centre + one cell, rays such that
        their spacing at that range is at most half a cell."""
        dist = self.lander_distance(z.shape, dx, dy, rc)
        step = min(dx, dy)
        max_range = float(dist.max()) + step
        n_rays = max(8, int(math.ceil(2.0 * math.pi * max_range / (0.5 * step))))
        n_samples = max(int(max_range // step), 1)
        if n_rays * n_samples > MAX_VIEWSHED_SAMPLES:
            raise ValueError(
                f"Viewshed too large: {n_rays} rays x {n_samples} samples. "
                "Use a coarser grid (e.g. make_slice at 5-20 m)."
            )
        target_h = float(getattr(self.agent, "eye_height_m", 1.8))
        seen = viewshed(z, dx, dy, rc, observer_h=self.observer_h, target_h=target_h,
                        max_range_m=max_range, n_rays=n_rays)
        vis = np.where(np.isfinite(z), 1.0 - seen.astype(np.float64), np.nan)
        return vis, {"max_range_m": max_range, "n_rays": n_rays,
                     "observer_h_m": self.observer_h, "target_h_m": target_h}

    # ------------------------------------------------------------------ adapt
    def adapt(self, data: Map) -> list[Map]:
        m = data
        if m.value_type != Datatype.ELEVATION:
            raise ValueError(f"ModelMapsAdapter needs an ELEVATION Map, got {m.value_type}.")
        z, dx, dy = from_map(m)
        rc = self.lander_rc(m)
        grid = {"dx_m": dx, "dy_m": dy, "lander_rc": list(rc)}

        if self.step_dem is None:
            step_cell = None
            step_values = self.step_height(z)
            step_desc = "max |step| over the 8 arcs of the cell; NaN on nodata"
        else:
            step_cell = step_values = self.fine_step_height(m)
            step_desc = ("max |step| of the step_dem cells inside the cell (each: max "
                         "over its 8 arcs); NaN on nodata")
        maps = [self._map(
            m, step_values, Datatype.STEP_HEIGHT, "m",
            f"src.models.terrain.edge_steps(k={self.steps_k}, compensate=True)"
            + ("" if self.step_dem is None else f" on step_dem {self.step_dem.title!r}"),
            step_desc,
            {**grid, "steps_k": self.steps_k,
             "step_dem_granularity": None if self.step_dem is None
             else list(self.step_dem.granularity)},
        )]
        if not self.is_walker:
            return maps

        maps.append(self._map(
            m, self.lander_distance(z.shape, dx, dy, rc), Datatype.LANDER_DISTANCE, "m",
            "src.models.terrain.from_map (local spherical approximation, MARS_RADIUS_M)",
            "horizontal distance from the lander cell", grid,
        ))

        eta = self._eta_values(m)
        time_s = self.edge_times(z, dx, dy, eta, step_cell)
        self.last_edge_times = time_s  # for the caller (e.g. walking time along a path)
        t_ret = return_time_map(time_s, rc)
        t_out = outbound_time_map(time_s, rc)
        time_note = {
            **grid,
            "eta": "none (rock)" if eta is None else
                   "time x mean eta of the arc (option B, TEAM ASSUMPTION); "
                   "NaN eta = regolith (TEAM ASSUMPTION)",
            "unreachable": "inf",
        }
        maps.append(self._map(
            m, t_ret, Datatype.RETURN_TIME, "s",
            "src.models.metabolic.edge_costs + src.models.budget.return_time_map",
            "minimum walking time from the cell to the lander", time_note,
        ))
        maps.append(self._map(
            m, t_out + t_ret, Datatype.EVA_TIME, "s",
            "src.models.budget.outbound_time_map + return_time_map",
            "lander -> cell -> lander walking time, no dwell", time_note,
        ))

        vis, vis_note = self.visibility(z, dx, dy, rc)
        maps.append(self._map(
            m, vis, Datatype.VISIBILITY, "dimensionless (0 = in view, 1 = out of view)",
            "1 - src.models.viewshed.viewshed from the lander",
            "line of sight from the lander antenna to the astronaut's helmet",
            {**grid, **vis_note,
             "team_assumption": "observer_h (antenna) and target_h (eye height)"},
        ))
        return maps


def _check_same_grid(reference: Map, other: Map) -> None:
    if np.shape(other.values) != np.shape(reference.values):
        raise ValueError(f"{other.title!r} has shape {np.shape(other.values)}, "
                         f"expected {np.shape(reference.values)}.")
    for name in ("granularity", "top_left", "bottom_right"):
        a = np.asarray(getattr(reference, name), dtype=float)
        b = np.asarray(getattr(other, name), dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=GRID_ATOL):
            raise ValueError(f"{other.title!r} has {name} {tuple(b)}, expected {tuple(a)}: "
                             "resample both on the same grid (make_slice) first.")
