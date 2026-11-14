"""
Implementation of the adapter interface in order to insert geographical data coming in DEM
format to our data structure defined in map.py
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer

from src.data.adapters.adapter import Adapter
from src.data.map import Map
from src.data.types import Datatype


class MapAdapter(Adapter):
    """Convert a north-up, single-band elevation raster into a Map."""

    def adapt(self, data: str | Path) -> Map:
        """
        Read a DEM file and return its elevation grid and spatial metadata.

        Missing elevations become NaN.
        Coordinates are (longitude, latitude) in the source body's geodetic CRS.
        Angular spacing is approximated from the first and last cell centres;
        the elevation grid is not resampled. Elevations retain their source units.
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

            # Read the elevation band, masking missing measurements.
            values = source.read(1, masked=True).astype(np.float64)

            # Apply the file's scale and offset to obtain elevation values.
            values = values * source.scales[0] + source.offsets[0]
            elevation = values.filled(np.nan)

            source_crs = CRS.from_wkt(source.crs.to_wkt())
            geographic_crs = source_crs.geodetic_crs
            if geographic_crs is None:
                raise ValueError("The DEM CRS must have a geodetic CRS.")
            transformer = Transformer.from_crs(
                source_crs, geographic_crs, always_xy=True,
            )

            def geographic_cell(row: int, column: int) -> tuple[float, float]:
                x, y = source.xy(row, column, offset="center")
                longitude, latitude = transformer.transform(x, y, errcheck=True)
                return float(longitude), float(latitude)

            top_left = geographic_cell(0, 0)
            bottom_right = geographic_cell(source.height - 1, source.width - 1)
            # For a singleton axis, estimate its spacing using a neighbouring
            # position in the source transform, without allocating a mesh.
            right = bottom_right if source.width > 1 else geographic_cell(0, 1)
            bottom = bottom_right if source.height > 1 else geographic_cell(1, 0)
            granularity = (
                (right[0] - top_left[0]) / max(source.width - 1, 1),
                (top_left[1] - bottom[1]) / max(source.height - 1, 1),
            )
            metadata = json.dumps({
                "source": str(data),
                "source_crs": source_crs.to_wkt(),
                "geographic_crs": geographic_crs.to_wkt(),
                "coordinate_units": "degrees",
                "value_unit": source.units[0],
                "tags": source.tags(),
                "band_tags": source.tags(1),
                "description": source.descriptions[0],
                "grid_approximation": "constant angular spacing between cell centres",
            })

            return Map(
                values=elevation,
                granularity=granularity,
                top_left=top_left,
                bottom_right=bottom_right,
                metadata=metadata,
                value_type=Datatype.ELEVATION,
                title=f"Elevation Map {top_left[0]} {top_left[1]} {bottom_right[0]} {bottom_right[1]}",
            )
