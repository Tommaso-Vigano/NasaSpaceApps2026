"""
Implementation of the adapter interface in order to insert Mars Climate Database (MCD)
diurnal CSV exports to our data structure defined in map.py

The MCD point is a single grid point over Jezero (18.44 N, 77.45 E, 2 m above the
surface), so the resulting Map is spatially uniform over the requested bounds.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.aliases import Point2D
from src.data.adapters.adapter import Adapter
from src.data.map import Map
from src.data.types import Datatype
from src.data.adapters.registry import AdapterRegistry


@AdapterRegistry.register_adapter("MCD-temperature")
class TemperatureAdapter(Adapter):
    """Turn an MCD diurnal CSV into a uniform temperature Map (Kelvin)."""

    TIME_COLUMN = "Local time"
    FIELD_COLUMNS = {
        "air": "Temperature (K)",
        "surface": "Surface temperature",
    }
    CITATION = "Mars Climate Database v6.2 (LMD/ESA/CNES)"
    SPATIAL_NOTE = (
        "single MCD grid point over Jezero, spatially uniform; "
        "elevation correction (~1.5 K/km lapse) not applied"
    )

    def __init__(
        self,
        local_time_h: float = 13.0,
        field: str = "air",
        top_left: Point2D = (77.160, 18.721),
        bottom_right: Point2D = (77.699, 18.211),
        resolution_deg: float = 0.01,
    ) -> None:
        """
        local_time_h: local Martian hour (cyclic, 0-24) at which to sample the day.
        field: "air" (temperature 2 m above the surface) or "surface".
        top_left, bottom_right: bounds (longitude East, latitude) in degrees.
        resolution_deg: grid spacing in degrees along both axes.
        """
        if field not in self.FIELD_COLUMNS:
            raise ValueError(
                f"field must be one of {sorted(self.FIELD_COLUMNS)}, got {field!r}."
            )
        if not math.isfinite(local_time_h):
            raise ValueError("local_time_h must be finite.")
        if not (math.isfinite(resolution_deg) and resolution_deg > 0):
            raise ValueError("resolution_deg must be a finite positive number.")
        left, top = (float(v) for v in top_left)
        right, bottom = (float(v) for v in bottom_right)
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            raise ValueError("Bounds must contain finite coordinates.")
        if right < left or bottom > top:
            raise ValueError("Expected a top-left and bottom-right ordered rectangle.")

        self.local_time_h = float(local_time_h)
        self.field = field
        self.top_left = (left, top)
        self.bottom_right = (right, bottom)
        self.resolution_deg = float(resolution_deg)

    @staticmethod
    def _read_rows(path: str | Path) -> tuple[list[str], list[list[str]]]:
        """Return (header, data rows), skipping blank lines and '#' comments."""
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                row for row in csv.reader(f)
                if row and any(cell.strip() for cell in row)
                and not row[0].lstrip().startswith("#")
            ]
        if not rows:
            raise ValueError(f"{path}: empty file, no header found.")
        return [h.strip() for h in rows[0]], rows[1:]

    @staticmethod
    def _column(header: list[str], rows: list[list[str]], index: int) -> np.ndarray:
        def to_float(cell: str) -> float:
            try:
                return float(cell)
            except ValueError:
                return math.nan

        return np.array(
            [to_float(r[index]) if index < len(r) else math.nan for r in rows],
            dtype=np.float64,
        )

    @staticmethod
    def _interpolate_cyclic(hours: np.ndarray, values: np.ndarray, at: float) -> float:
        """Linear interpolation on a 24 h cycle (hour 24 is hour 0)."""
        valid = np.isfinite(hours) & np.isfinite(values)
        if not np.any(valid):
            raise ValueError("No valid (local time, value) samples in the file.")
        hours = np.mod(hours[valid], 24.0)
        values = values[valid]
        # Keep the first sample for each hour (drops the 24 h duplicate of 0 h).
        hours, first = np.unique(hours, return_index=True)
        values = values[first]
        return float(np.interp(at % 24.0, hours, values, period=24.0))

    @staticmethod
    def _axis(start: float, extent: float, step: float) -> np.ndarray:
        """Inclusive cell-centre axis, same rule as SamplingUtils.window_grid."""
        intervals = extent / step
        # Tolerate rounding of an otherwise integral number of intervals.
        nearest = round(intervals)
        if np.isclose(intervals, nearest, rtol=0, atol=1e-10):
            intervals = nearest
        count = int(np.floor(intervals)) + 1
        return start + np.arange(count, dtype=float) * step

    def adapt(self, data: str | Path) -> Map:
        """
        Read an MCD diurnal CSV and return a uniform temperature Map in Kelvin
        at self.local_time_h, covering the configured bounds.
        """
        header, rows = self._read_rows(data)

        time_index = next(
            (j for j, name in enumerate(header) if self.TIME_COLUMN in name), None,
        )
        if time_index is None:
            raise ValueError(
                f"{data}: not an MCD diurnal file (no column containing "
                f"{self.TIME_COLUMN!r}); columns: {header}"
            )

        wanted = self.FIELD_COLUMNS[self.field]
        value_index = next(
            (j for j, name in enumerate(header)
             if wanted in name and (self.field != "air" or "Surface" not in name)),
            None,
        )
        if value_index is None:
            raise KeyError(
                f"{data}: no column containing {wanted!r} for field "
                f"{self.field!r}; columns: {header}"
            )

        temperature = self._interpolate_cyclic(
            self._column(header, rows, time_index),
            self._column(header, rows, value_index),
            self.local_time_h,
        )

        left, top = self.top_left
        right, bottom = self.bottom_right
        step = self.resolution_deg
        xs = self._axis(left, right - left, step)
        # Latitude decreases as the row index increases.
        ys = top - self._axis(0.0, top - bottom, step)
        top_left = (float(xs[0]), float(ys[0]))
        bottom_right = (float(xs[-1]), float(ys[-1]))

        scenario = Path(data).stem
        if scenario.startswith("diurnal_"):
            scenario = scenario[len("diurnal_"):]

        metadata = json.dumps({
            "source": str(data),
            "coordinate_units": "degrees",
            "value_unit": "K",
            "field": self.field,
            "source_column": header[value_index],
            "local_time_h": self.local_time_h,
            "scenario": scenario,
            "citation": self.CITATION,
            "spatial_note": self.SPATIAL_NOTE,
        })

        return Map(
            values=np.full((ys.size, xs.size), temperature, dtype=np.float64),
            granularity=(step, step),
            top_left=top_left,
            bottom_right=bottom_right,
            metadata=metadata,
            value_type=Datatype.TEMPERATURE,
            title=(
                f"Temperature Map {top_left[0]} {top_left[1]} "
                f"{bottom_right[0]} {bottom_right[1]}"
            ),
        )
