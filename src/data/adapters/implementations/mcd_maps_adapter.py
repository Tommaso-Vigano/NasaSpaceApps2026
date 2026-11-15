"""Uniform GO/NO-GO Maps from a Mars Climate Database diurnal profile [R24].

McdMapsAdapter(reference_map, eva_start_h, eva_duration_h).adapt(csv) -> list[Map]

All Maps are uniform (one value everywhere) on the grid of ``reference_map``:
    DUST_TAU    dust optical depth, maximum over the whole day
    SOLAR_FLUX  W/m^2, minimum inside the EVA window [start, start + duration]
    WIND_SPEED  m/s,   maximum inside the EVA window
Inside the window the hourly samples are linearly interpolated at the window ends, so a
window starting at 9.5 h uses the value at 9.5 h, not the one at 9 h or 10 h.

The MCD profile has one value for the whole crater (one grid point of a GCM), hence
uniform maps: penalty_config uses only cut-off templates on them (0 or inf everywhere).
A column missing from the CSV gives no Map (with a warning).
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np

from src.data.adapters.adapter import Adapter
from src.data.adapters.registry import AdapterRegistry
from src.data.map import Map
from src.data.types import Datatype
from src.constants import COL_FLUX, COL_TAU, COL_TIME, COL_WIND
from src.models.gates import _column, load_mcd_diurnal


def window_values(hours, values, start_h, end_h) -> np.ndarray:
    """Samples inside [start_h, end_h] plus the values interpolated at both ends."""
    hours = np.asarray(hours, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(hours) & np.isfinite(values)
    hours, values = hours[ok], values[ok]
    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    if hours.size == 0:
        raise ValueError("No finite MCD samples.")
    if start_h < hours[0] or end_h > hours[-1]:
        raise ValueError(
            f"EVA window {start_h:g}-{end_h:g} h outside the MCD profile "
            f"{hours[0]:g}-{hours[-1]:g} h."
        )
    inside = values[(hours > start_h) & (hours < end_h)]
    ends = np.interp([start_h, end_h], hours, values)
    return np.concatenate([ends, inside])


@AdapterRegistry.register_adapter("MCD-gates")
class McdMapsAdapter(Adapter):
    """DUST_TAU, SOLAR_FLUX, WIND_SPEED uniform Maps on the grid of ``reference_map``."""

    SOURCE = "Mars Climate Database v6 (LMD/ESA/CNES), diurnal profile [R24]"

    def __init__(self, reference_map: Map, eva_start_h: float, eva_duration_h: float):
        start, duration = float(eva_start_h), float(eva_duration_h)
        if not (math.isfinite(start) and math.isfinite(duration) and duration > 0):
            raise ValueError("eva_start_h must be finite and eva_duration_h positive.")
        self.reference = reference_map
        self.eva_start_h = start
        self.eva_duration_h = duration

    @property
    def eva_end_h(self) -> float:
        return self.eva_start_h + self.eva_duration_h

    def _uniform(self, value: float, data_type: Datatype, unit: str, statistic: str,
                 csv_path) -> Map:
        ref = self.reference
        meta = {
            "coordinate_units": "degrees",
            "value_unit": unit,
            "value": value,
            "statistic": statistic,
            "eva_window_h": [self.eva_start_h, self.eva_end_h],
            "source": self.SOURCE,
            "source_file": str(csv_path),
            "description": "uniform over the grid: the MCD profile is one point for the crater",
        }
        return Map(
            values=np.full(np.shape(ref.values), value, dtype=np.float64),
            granularity=tuple(ref.granularity),
            top_left=tuple(ref.top_left),
            bottom_right=tuple(ref.bottom_right),
            metadata=json.dumps(meta),
            value_type=data_type,
            title=f"{data_type.value} (MCD {Path(str(csv_path)).stem})",
        )

    def adapt(self, data) -> list[Map]:
        mcd = load_mcd_diurnal(data)
        hours = _column(mcd, COL_TIME)
        start, end = self.eva_start_h, self.eva_end_h
        specs = [
            (COL_TAU, Datatype.DUST_TAU, "dimensionless", "max over the day",
             lambda v: float(np.nanmax(v))),
            (COL_FLUX, Datatype.SOLAR_FLUX, "W/m^2", "min inside the EVA window",
             lambda v: float(np.min(window_values(hours, v, start, end)))),
            (COL_WIND, Datatype.WIND_SPEED, "m/s", "max inside the EVA window",
             lambda v: float(np.max(window_values(hours, v, start, end)))),
        ]
        maps = []
        for column, data_type, unit, statistic, reduce in specs:
            try:
                values = _column(mcd, column)
            except KeyError:
                warnings.warn(f"{data}: no MCD column containing {column!r}, "
                              f"no {data_type.name} map.", stacklevel=2)
                continue
            maps.append(self._uniform(reduce(values), data_type, unit, statistic, data))
        return maps
