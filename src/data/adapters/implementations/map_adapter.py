"""
Implementation of the adapter interface in order to insert geographical data coming in DEM
format to our data structure defined in map.py
"""

from pathlib import Path

import numpy as np
import rasterio

from ..adapter import Adapter
from ...map import Map
from ...types import SpatialUnit


class MapAdapter(Adapter):
    """Convert a north-up, single-band elevation raster into a Map."""

    def adapt(self, data: str | Path) -> Map:
        """
        Read a DEM file and return its elevation grid and spatial metadata.

        Missing elevations become NaN.
        Coordinates and elevations retain their source units.
        """
        with rasterio.open(data) as source:
            if source.count != 1:
                raise ValueError(
                    "Expected a DEM with exactly one elevation band."
                )

            if source.crs is None:
                raise ValueError(
                    "The DEM must define a coordinate reference system."
                )

            transform = source.transform

            # Map assumes x increases rightwards and y decreases downwards.
            if (
                transform.b != 0
                or transform.d != 0
                or transform.a <= 0
                or transform.e >= 0
            ):
                raise ValueError(
                    "Expected a north-up raster without rotation."
                )

            # Identify the units of horizontal coordinates and spacing.
            if source.crs.is_geographic:
                value_type = SpatialUnit.Coordinate
            elif (
                source.crs.is_projected
                and source.crs.linear_units_factor[1] == 1.0
            ):
                value_type = SpatialUnit.Metre
            else:
                raise ValueError(
                    "Only geographic coordinates or projected metres "
                    "are supported."
                )

            # Read the elevation band, masking missing measurements.
            values = source.read(1, masked=True).astype(np.float64)

            # Apply the file's scale and offset to obtain elevation values.
            values = values * source.scales[0] + source.offsets[0]
            elevation = values.filled(np.nan)

            # Use the centre of the first cell, consistent with Map.
            x, y = source.xy(0, 0, offset="center")

            return Map(
                elevation=elevation,
                granularity=(float(transform.a), float(-transform.e)),
                top_left=(float(x), float(y)),
                crs=source.crs.to_wkt(),
                value_type=value_type,
            )