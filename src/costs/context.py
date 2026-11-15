"""Inputs shared by every cost function: the grid, the agent and the weights.

Nothing here depends on the grid size: every array takes its shape from ``z``
(the DEM produced by ``WorldModel.make_slice``) and is only checked against it.

Agent profiles are duck-typed. A *walker* is any object with the attributes
used by ``src.models.constraints.Astronaut`` (detected by ``v_nominal_ms``);
a *rover* is any object with the attributes of ``RoboticRover`` (detected by
``speed_ms``). ``isinstance`` is never used, so the colleagues' profiles work
as long as they expose the same attributes.
"""

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from src.costs.bridge import lonlat_to_rc
from src.data.types import Datatype
from src.models.constraints import ETA_TERRAIN
from src.models.gates import eva_go
from src.models.terrain import from_map

log = logging.getLogger(__name__)


def is_walker(agent) -> bool:
    """True for an astronaut-like profile (has ``v_nominal_ms``)."""
    return hasattr(agent, "v_nominal_ms")


def is_rover(agent) -> bool:
    """True for a rover-like profile (has ``speed_ms``, not a walker)."""
    return hasattr(agent, "speed_ms") and not is_walker(agent)


def _gate_default(name):
    """Default of a keyword argument of ``src.models.gates.eva_go`` (reused,
    not copied, so the two never diverge)."""
    return inspect.signature(eva_go).parameters[name].default


# --------------------------------------------------------------------------
# Weights and thresholds
# --------------------------------------------------------------------------
@dataclass
class CostWeights:
    """Every weight and threshold of the cost functions, in one place.

    ``None`` means: filter or penalty disabled (no source for a value yet).
    """

    # eta used where the NOAH-H map has no data (NaN). TEAM ASSUMPTION:
    # unmapped terrain is generic regolith (ETA_TERRAIN is itself a TEAM
    # ASSUMPTION derived from [R18]).
    eta_default: float = ETA_TERRAIN["regolith"]

    # From src.models.gates (defaults of eva_go; TEAM ASSUMPTIONs documented there).
    tau_max: float = field(default_factory=lambda: _gate_default("tau_max"))
    min_flux: float = field(default_factory=lambda: _gate_default("min_flux"))

    # TEAM ASSUMPTION: no source yet -> disabled.
    temp_air_min_K: Optional[float] = None
    temp_air_max_K: Optional[float] = None
    temp_surface_min_K: Optional[float] = None
    temp_surface_max_K: Optional[float] = None
    wind_max_ms: Optional[float] = None
    # Extra cost = weight * edge time on edges ending out of line of sight
    # of the lander. TEAM ASSUMPTION: no source yet -> disabled.
    visibility_penalty_weight: Optional[float] = None

    def __post_init__(self):
        if not (math.isfinite(self.eta_default) and self.eta_default >= 1.0):
            raise ValueError(f"eta_default must be finite and >= 1, got {self.eta_default}.")
        w = self.visibility_penalty_weight
        if w is not None and not (math.isfinite(w) and w >= 0):
            raise ValueError(f"visibility_penalty_weight must be finite and >= 0, got {w}.")


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------
@dataclass
class CostContext:
    """Grid, agent and environment for the cost functions.

    :param z: DEM (m), 2-D; its shape defines the grid.
    :param dx, dy: cell size (m) along columns / rows.
    :param eta_map: Pandolf terrain coefficient per cell (same shape as z), or
        None. Must not contain NaN: fill it with ``fill_eta`` first
        (``from_slice`` does it).
    :param lander_rc: lander cell ``(row, col)``.
    :param agent: walker or rover profile (duck-typed, see module docstring).
    :param mcd: dict from ``src.models.gates.load_mcd_diurnal``, or None.
    :param eva_start_h: local Martian hour at which the EVA starts, or None.
    :param eva_duration_h: EVA duration (h); None -> ``agent.eva_nominal_h``
        if the agent has it.
    :param k_rocks: rock abundance k (float or array shaped like z).
    :param large_ripple_mask: True on NOAH-H "Large ripples" cells, or None.
    :param thermal_inertia: thermal inertia per cell, or None.
    """

    z: np.ndarray
    dx: float
    dy: float
    eta_map: Optional[np.ndarray]
    lander_rc: tuple
    agent: object
    mcd: Optional[dict] = None
    eva_start_h: Optional[float] = None
    eva_duration_h: Optional[float] = None
    k_rocks: Union[float, np.ndarray] = 0.0
    large_ripple_mask: Optional[np.ndarray] = None
    thermal_inertia: Optional[np.ndarray] = None

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=np.float64)
        if self.z.ndim != 2 or self.z.size == 0:
            raise ValueError(f"z must be a non-empty 2-D array, got shape {self.z.shape}.")
        shape = self.z.shape
        for name in ("dx", "dy"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and positive, got {v}.")
            setattr(self, name, v)

        def check(name, arr, dtype):
            if arr is None:
                return None
            a = np.asarray(arr, dtype=dtype)
            if a.shape != shape:
                raise ValueError(
                    f"CostContext.{name} has shape {a.shape}, expected the shape of z {shape}."
                )
            return a

        self.eta_map = check("eta_map", self.eta_map, np.float64)
        if self.eta_map is not None and np.isnan(self.eta_map).any():
            raise ValueError(
                f"CostContext.eta_map contains {int(np.isnan(self.eta_map).sum())} NaN: "
                "fill them with fill_eta(eta_map, CostWeights().eta_default) "
                "(from_slice does it automatically)."
            )
        self.large_ripple_mask = check("large_ripple_mask", self.large_ripple_mask, bool)
        self.thermal_inertia = check("thermal_inertia", self.thermal_inertia, np.float64)
        if np.ndim(self.k_rocks) == 0:
            self.k_rocks = float(self.k_rocks)
        else:
            self.k_rocks = check("k_rocks", self.k_rocks, np.float64)

        r, c = (int(v) for v in self.lander_rc)
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            raise ValueError(f"lander_rc {self.lander_rc} is outside the grid of shape {shape}.")
        if not np.isfinite(self.z[r, c]):
            raise ValueError(f"lander_rc {self.lander_rc} is on a NaN (nodata) cell of z.")
        self.lander_rc = (r, c)

        if not (is_walker(self.agent) or is_rover(self.agent)):
            raise TypeError(
                "agent must expose the attributes of Astronaut (v_nominal_ms, ...) "
                f"or of RoboticRover (speed_ms, ...); got {type(self.agent).__name__}."
            )
        if self.eva_duration_h is None and hasattr(self.agent, "eva_nominal_h"):
            self.eva_duration_h = float(self.agent.eva_nominal_h)

    @property
    def shape(self) -> tuple:
        return self.z.shape


def fill_eta(eta_map, eta_default: float) -> np.ndarray:
    """Replace NaN (outside NOAH-H coverage) with ``eta_default``: missing
    terrain classes are NOT obstacles."""
    eta = np.asarray(eta_map, dtype=np.float64)
    return np.where(np.isnan(eta), float(eta_default), eta)


def _first_map(world_slice, datatype):
    return next((m for m in world_slice if m.value_type == datatype), None)


def from_slice(world_slice, lander_lonlat, agent, weights: Optional[CostWeights] = None,
               **kwargs) -> CostContext:
    """Build a ``CostContext`` from a ``WorldSlice`` (``WorldModel.make_slice``).

    Uses the ELEVATION Map (required) for z, dx, dy and the lander cell, the
    TERRAIN_ETA Map (NaN -> ``weights.eta_default``; missing layer -> None,
    with a warning) and the THERMAL_INERTIA Map if present. Extra keyword
    arguments (mcd, eva_start_h, k_rocks, large_ripple_mask, ...) are passed
    to ``CostContext``.
    """
    weights = weights or CostWeights()
    elev = _first_map(world_slice, Datatype.ELEVATION)
    if elev is None:
        raise ValueError("The WorldSlice has no ELEVATION map.")
    glon, glat = (float(g) for g in elev.granularity)
    if not math.isclose(glon, glat, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(
            f"ELEVATION granularity must be equal in lon and lat for the GridGraph, "
            f"got {elev.granularity}."
        )
    z, dx, dy = from_map(elev)
    lander_rc = lonlat_to_rc(lander_lonlat, elev.top_left, glon, z.shape)

    eta_m = _first_map(world_slice, Datatype.TERRAIN_ETA)
    if eta_m is None:
        log.warning("from_slice: no TERRAIN_ETA map, eta_map = None (no eta term).")
        eta_map = None
    else:
        eta_map = fill_eta(eta_m.values, weights.eta_default)

    ti_m = _first_map(world_slice, Datatype.THERMAL_INERTIA)
    kwargs.setdefault("thermal_inertia", None if ti_m is None else ti_m.values)
    return CostContext(z=z, dx=dx, dy=dy, eta_map=eta_map, lander_rc=lander_rc,
                       agent=agent, **kwargs)
