"""Single-valued (global) gates and the rover thermal-inertia filter.

Every gate returns ``(edge_cost, reason)``: zeros shaped ``(8, *z.shape)`` if
GO, all inf if NO-GO. A threshold left to None in ``CostWeights`` disables
its gate (GO). Without MCD data every gate is GO with reason "nessun dato MCD".
"""

import math
from typing import Optional

import numpy as np

from src.costs._edges import inf_like_edges, touching_inf, zeros_like_edges
from src.costs.context import CostContext, CostWeights, is_rover
from src.data.adapters.implementations.temperature_adapter import TemperatureAdapter
from src.models.gates import (
    COL_FLUX,
    COL_TAU,
    COL_TEMP,
    COL_TIME,
    COL_TSURF,
    COL_WIND,
    _column,
    dust_storm_nogo,
    eva_go,
)
from src.models.rover import rover_hard_mask

NO_MCD = "nessun dato MCD (no MCD data): GO"
COL_TEMP_AIR = COL_TEMP
COL_TEMP_SURFACE = COL_TSURF


def _find_column(mcd, substring) -> Optional[np.ndarray]:
    """First MCD column matching ``substring`` with the same rule as
    ``gates.load_mcd_diurnal`` (air temperature never matches the surface
    column), or None."""
    try:
        return _column(mcd, substring)
    except KeyError:
        return None


def _result(ctx, nogo: bool, reason: str):
    return (inf_like_edges(ctx.z) if nogo else zeros_like_edges(ctx.z)), reason


def _missing(ctx, *columns):
    """Reason for a GO when MCD data or a needed column is missing, else None."""
    if ctx.mcd is None:
        return NO_MCD
    absent = [c for c in columns if _find_column(ctx.mcd, c) is None]
    if absent:
        return f"MCD column(s) {absent} missing: gate not evaluated, GO"
    return None


def _window_hours(mcd_hours, start_h, duration_h) -> np.ndarray:
    """Local hours to sample the EVA window [start, start + duration]: the two
    ends plus every MCD sample hour falling inside (repeated each sol), so the
    min/max of the piecewise-linear profile in the window are exact."""
    end_h = start_h + duration_h
    h = np.mod(np.asarray(mcd_hours, dtype=np.float64)[np.isfinite(mcd_hours)], 24.0)
    first = start_h + np.mod(h - start_h, 24.0)  # first occurrence >= start
    sols = np.arange(int(duration_h // 24.0) + 1)
    inside = (first[:, None] + 24.0 * sols[None, :]).ravel()
    inside = inside[inside <= end_h]
    return np.unique(np.concatenate(([start_h, end_h], inside)))


def _profile_in_window(ctx, column):
    """Values of an MCD column along the EVA window, interpolated on the
    24 h cycle with ``TemperatureAdapter._interpolate_cyclic``."""
    hours = _find_column(ctx.mcd, COL_TIME)
    values = _find_column(ctx.mcd, column)
    at = _window_hours(hours, ctx.eva_start_h, ctx.eva_duration_h)
    return np.array([TemperatureAdapter._interpolate_cyclic(hours, values, t) for t in at])


def _window_missing(ctx):
    if ctx.eva_start_h is None or ctx.eva_duration_h is None:
        return "EVA window (eva_start_h / eva_duration_h) not set: gate not evaluated, GO"
    return None


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def dust_storm_gate(ctx: CostContext, weights: Optional[CostWeights] = None):
    """NO-GO if the MCD dust optical depth reaches ``weights.tau_max``
    (``gates.dust_storm_nogo``)."""
    w = weights or CostWeights()
    why = _missing(ctx, COL_TAU)
    if why:
        return _result(ctx, False, why)
    tau = float(np.nanmax(_find_column(ctx.mcd, COL_TAU)))
    if dust_storm_nogo(ctx.mcd, w.tau_max):
        return _result(ctx, True, f"dust storm: optical depth {tau:.2f} >= {w.tau_max}")
    return _result(ctx, False, f"optical depth {tau:.2f} < {w.tau_max}: GO")


def daylight_gate(ctx: CostContext, weights: Optional[CostWeights] = None):
    """NO-GO if the EVA window leaves the daylight window (solar flux >=
    ``weights.min_flux``), via ``gates.eva_go``. The dust part of ``eva_go``
    is switched off here (tau_max = inf): it is ``dust_storm_gate``'s job."""
    w = weights or CostWeights()
    why = _missing(ctx, COL_TIME, COL_FLUX) or _window_missing(ctx)
    if why:
        return _result(ctx, False, why)
    go, reasons = eva_go(ctx.mcd, ctx.eva_start_h, ctx.eva_duration_h,
                         min_flux=w.min_flux, tau_max=math.inf)
    if not go:
        return _result(ctx, True, "; ".join(reasons))
    return _result(ctx, False,
                   f"EVA {ctx.eva_start_h:.2f}-{ctx.eva_start_h + ctx.eva_duration_h:.2f} h "
                   f"inside daylight (flux >= {w.min_flux} W/m2): GO")


def _range_gate(ctx, column, label, unit, lo, hi):
    if lo is None and hi is None:
        return _result(ctx, False, f"{label}: thresholds None, gate disabled")
    why = _missing(ctx, COL_TIME, column) or _window_missing(ctx)
    if why:
        return _result(ctx, False, f"{label}: {why}")
    v = _profile_in_window(ctx, column)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    bad = []
    if lo is not None and vmin < lo:
        bad.append(f"min {vmin:.1f} {unit} < {lo} {unit}")
    if hi is not None and vmax > hi:
        bad.append(f"max {vmax:.1f} {unit} > {hi} {unit}")
    if bad:
        return _result(ctx, True, f"{label} during EVA: " + ", ".join(bad))
    return _result(ctx, False, f"{label} during EVA {vmin:.1f}-{vmax:.1f} {unit}: GO")


def temperature_gate(ctx: CostContext, weights: Optional[CostWeights] = None):
    """NO-GO if the MCD air (``"Temperature (K)"``) or surface
    (``"Surface temperature"``) temperature leaves
    [temp_*_min_K, temp_*_max_K] during the EVA window. Each threshold set to
    None is ignored; a field with both thresholds None is not checked."""
    w = weights or CostWeights()
    air, air_why = _range_gate(ctx, COL_TEMP_AIR, "air temperature", "K",
                               w.temp_air_min_K, w.temp_air_max_K)
    srf, srf_why = _range_gate(ctx, COL_TEMP_SURFACE, "surface temperature", "K",
                               w.temp_surface_min_K, w.temp_surface_max_K)
    return air + srf, "; ".join((air_why, srf_why))


def wind_gate(ctx: CostContext, weights: Optional[CostWeights] = None):
    """NO-GO if the MCD wind speed exceeds ``weights.wind_max_ms`` during the
    EVA window (None -> disabled)."""
    w = weights or CostWeights()
    return _range_gate(ctx, COL_WIND, "wind speed", "m/s", None, w.wind_max_ms)


# --------------------------------------------------------------------------
# Spatial
# --------------------------------------------------------------------------
def thermal_inertia_nogo(ctx: CostContext) -> np.ndarray:
    """Rover only: inf on edges starting or ending in a cell with thermal
    inertia below ``agent.min_thermal_inertia`` (or NaN), via
    ``rover.rover_hard_mask`` with the slope part switched off.

    Display term: ``rover_edge_time`` (in ``time_cost``) already applies it, so
    it does not change the total. Zeros for a walker or without data.
    """
    if ctx.thermal_inertia is None or not is_rover(ctx.agent):
        return zeros_like_edges(ctx.z)
    if not hasattr(ctx.agent, "min_thermal_inertia"):
        return zeros_like_edges(ctx.z)
    no_slope = np.zeros(ctx.shape)
    return touching_inf(rover_hard_mask(no_slope, ctx.agent, ti=ctx.thermal_inertia))
