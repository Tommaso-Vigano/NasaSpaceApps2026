"""
Implementation of the adapter interface in order to insert the THEMIS-derived thermal
inertia raster of Jezero crater (GeoTIFF) to our data structure defined in map.py

Source: Edwards & Buz (2021), THEMIS-derived Thermal Inertia of Jezero Crater, Mars,
Zenodo, doi:10.5281/zenodo.4931348, CC BY 4.0.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer

from src.data.adapters.adapter import Adapter
from src.data.map import Map
from src.data.types import Datatype
from src.data.adapters.registry import AdapterRegistry


@AdapterRegistry.register_adapter("Jezero-thermal-inertia")
class ThermalInertiaAdapter(Adapter):
    """Convert a north-up, single-band thermal inertia raster into a Map."""

    # Physically plausible thermal inertia range (J m^-2 K^-1 s^-1/2).
    MIN_VALID = 5.0
    MAX_VALID = 5000.0

    VALUE_UNIT = "J m^-2 K^-1 s^-1/2"
    CITATION = (
        "Edwards, C. S. & Buz, J. (2021). THEMIS-derived Thermal Inertia of Jezero "
        "Crater, Mars. Zenodo. doi:10.5281/zenodo.4931348"
    )
    LICENSE = "CC BY 4.0"
    ACCURACY_NOTE = "THEMIS-derived thermal inertia, absolute accuracy ~20%"

    @classmethod
    def _clean(cls, values: np.ndarray) -> np.ndarray:
        """Return a copy where nodata (<= 0, non-finite) and out-of-range values are NaN.

        Out-of-range values are reported with a single warning.
        """
        values = np.array(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            values[~np.isfinite(values) | (values <= 0)] = np.nan
            out_of_range = (values < cls.MIN_VALID) | (values > cls.MAX_VALID)
        count = int(np.count_nonzero(out_of_range))
        if count:
            warnings.warn(
                f"{count} thermal inertia value(s) outside "
                f"[{cls.MIN_VALID:g}, {cls.MAX_VALID:g}] {cls.VALUE_UNIT} set to NaN.",
                stacklevel=3,
            )
            values[out_of_range] = np.nan
        return values

    def adapt(self, data: str | Path) -> Map:
        """
        Read a thermal inertia GeoTIFF and return its grid and spatial metadata.

        Missing values (0 in the source) become NaN, as do values outside the
        plausible range. Coordinates are (longitude, latitude) in the source
        body's geodetic CRS. Angular spacing is approximated from the first and
        last cell centres; the grid is not resampled.
        """
        with rasterio.open(data) as source:
            if source.count != 1:
                raise ValueError(
                    "Expected a thermal inertia raster with exactly one band."
                )

            if source.crs is None:
                raise ValueError(
                    "The thermal inertia raster must define a coordinate reference system."
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

            # Read the band, masking missing measurements.
            values = source.read(1, masked=True).astype(np.float64)

            # Apply the file's scale and offset to obtain physical values.
            values = values * source.scales[0] + source.offsets[0]
            thermal_inertia = self._clean(values.filled(np.nan))

            source_crs = CRS.from_wkt(source.crs.to_wkt())
            geographic_crs = source_crs.geodetic_crs
            if geographic_crs is None:
                raise ValueError("The raster CRS must have a geodetic CRS.")
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
                "value_unit": self.VALUE_UNIT,
                "valid_range": [self.MIN_VALID, self.MAX_VALID],
                "citation": self.CITATION,
                "license": self.LICENSE,
                "accuracy_note": self.ACCURACY_NOTE,
                "tags": source.tags(),
                "band_tags": source.tags(1),
                "description": (
                    "THEMIS-derived thermal inertia of Jezero crater; nodata (0) "
                    "and out-of-range values are NaN"
                ),
                "grid_approximation": "constant angular spacing between cell centres",
            })

            return Map(
                values=thermal_inertia,
                granularity=granularity,
                top_left=top_left,
                bottom_right=bottom_right,
                metadata=metadata,
                value_type=Datatype.THERMAL_INERTIA,
                title=(
                    f"Thermal Inertia Map {top_left[0]} {top_left[1]} "
                    f"{bottom_right[0]} {bottom_right[1]}"
                ),
            )
