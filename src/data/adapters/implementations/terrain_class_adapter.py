"""
Implementation of the adapter interface in order to insert the NOAH-H terrain
classification of Jezero crater (GeoTIFF, interpretative groups) to our data
structure defined in map.py, as a map of the Pandolf terrain coefficient eta.

The output Map is meant to be used as ``eta_map`` in
``src.models.metabolic.edge_costs``.

Source: Wright, J., Balme, M. R., Fawdon, P. et al. (2022), NOAH-H terrain
classifications of Jezero Crater, Mars, The Open University,
doi:10.21954/ou.rd.17121482.v2, CC BY 4.0.
"""

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.aliases import Point2D
from src.data.adapters.adapter import Adapter
from src.data.adapters.registry import AdapterRegistry
from src.data.map import Map
from src.data.types import Datatype
from src.models.constraints import ETA_TERRAIN


@AdapterRegistry.register_adapter("Jezero-NOAH-H-terrain")
class TerrainClassAdapter(Adapter):
    """Convert a window of the NOAH-H interpretative groups raster into an eta Map.

    NOAH-H interpretative groups (values of the source raster):
        0 = no data, 1 = Non-bedrock, 2 = Bedrock, 3 = Boulder fields,
        4 = Small ripples, 5 = Large ripples; 127 is the file's nodata value.

    Default class -> eta mapping (TEAM ASSUMPTION, keys of
    ``src.models.constraints.ETA_TERRAIN``):
        1 Non-bedrock    -> ETA_TERRAIN["regolith"]
        2 Bedrock        -> ETA_TERRAIN["rock"]
        3 Boulder fields -> ETA_TERRAIN["rocky"]
        4 Small ripples  -> ETA_TERRAIN["sand"]
        5 Large ripples  -> ETA_TERRAIN["sand"]

    Only the requested (lon, lat) box is read from the file. Classes are read
    at an intermediate 1 m resolution with nearest-neighbour resampling (they
    are categories, never interpolated), converted to eta, then block-averaged
    to ``resolution_m``, ignoring nodata. Cells with fewer than
    ``min_valid_fraction`` valid 1 m pixels are NaN, as are cells outside the
    raster coverage.

    Distances (``resolution_m``) are metres in the source equirectangular
    projection (eqc, lat_ts = 0), i.e. metres along a meridian; this keeps the
    output grid regular in degrees.
    """

    CLASS_NAMES = {
        1: "Non-bedrock",
        2: "Bedrock",
        3: "Boulder fields",
        4: "Small ripples",
        5: "Large ripples",
    }
    DEFAULT_CLASS_TO_ETA = {
        1: ETA_TERRAIN["regolith"],
        2: ETA_TERRAIN["rock"],
        3: ETA_TERRAIN["rocky"],
        4: ETA_TERRAIN["sand"],
        5: ETA_TERRAIN["sand"],
    }
    NODATA_CLASSES = (0, 127)

    INTERMEDIATE_RESOLUTION_M = 1.0
    # Upper bound on the 1 m rows read at once, to bound memory use.
    MAX_CHUNK_ROWS = 2048

    VALUE_UNIT = "dimensionless (Pandolf terrain coefficient eta)"
    CITATION = (
        "Wright, J., Balme, M. R., Fawdon, P. et al. (2022). NOAH-H terrain "
        "classifications of Jezero Crater, Mars. The Open University. "
        "doi:10.21954/ou.rd.17121482.v2"
    )
    LICENSE = "CC BY 4.0"

    def __init__(
        self,
        top_left: Point2D,
        bottom_right: Point2D,
        resolution_m: float = 5.0,
        class_to_eta: dict[int, float] | None = None,
        min_valid_fraction: float = 0.5,
    ):
        """
        top_left and bottom_right are the (longitude, latitude) outer edges of
        the requested box, in degrees. resolution_m must be a whole number of
        metres (>= 1) because it is reached by averaging 1 m blocks.
        """
        left, top = map(float, top_left)
        right, bottom = map(float, bottom_right)
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            raise ValueError("Box corners must be finite (longitude, latitude) degrees.")
        if not (left < right and bottom < top):
            raise ValueError(
                "Expected top_left west and north of bottom_right "
                "(corners are (longitude, latitude) in degrees)."
            )

        factor = resolution_m / self.INTERMEDIATE_RESOLUTION_M
        if not (math.isfinite(factor) and factor >= 1 and abs(factor - round(factor)) < 1e-9):
            raise ValueError(
                "resolution_m must be a whole multiple of the 1 m intermediate resolution."
            )

        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1].")

        mapping = dict(self.DEFAULT_CLASS_TO_ETA if class_to_eta is None else class_to_eta)
        for key, eta in mapping.items():
            if int(key) != key or not 0 <= key <= 255:
                raise ValueError(f"Class {key!r} is not a valid uint8 class value.")
            if key in self.NODATA_CLASSES:
                raise ValueError(f"Class {key} is nodata and cannot be mapped to eta.")
            if not (math.isfinite(eta) and eta > 0):
                raise ValueError(f"eta for class {key} must be finite and positive.")

        self.top_left = (left, top)
        self.bottom_right = (right, bottom)
        self.resolution_m = float(resolution_m)
        self.factor = int(round(factor))
        self.class_to_eta = {int(k): float(v) for k, v in mapping.items()}
        self.min_valid_fraction = float(min_valid_fraction)

    def _lookup_table(self, extra_nodata: float | None) -> np.ndarray:
        """uint8 class -> eta; unmapped classes and nodata are NaN."""
        lut = np.full(256, np.nan, dtype=np.float64)
        for key, eta in self.class_to_eta.items():
            lut[key] = eta
        nodata = list(self.NODATA_CLASSES)
        if extra_nodata is not None and float(extra_nodata).is_integer() \
                and 0 <= extra_nodata <= 255:
            nodata.append(int(extra_nodata))
        lut[nodata] = np.nan
        return lut

    @staticmethod
    def _sphere_radius(source_crs: CRS) -> float:
        """Radius of an eqc (lat_ts = 0) spherical CRS; otherwise raise.

        Only for this projection is a regular metre grid also regular in degrees.
        """
        operation = source_crs.coordinate_operation
        ellipsoid = source_crs.ellipsoid
        params = {} if operation is None else {p.name: p.value for p in operation.params}
        if (
            operation is None
            or not operation.method_name.startswith("Equidistant Cylindrical")
            or params.get("Latitude of 1st standard parallel", 0.0) != 0.0
            or ellipsoid is None
            or not math.isclose(ellipsoid.semi_major_metre, ellipsoid.semi_minor_metre)
        ):
            raise ValueError(
                "Expected an equirectangular (eqc, lat_ts = 0) CRS on a sphere."
            )
        return float(ellipsoid.semi_major_metre)

    def adapt(self, data: str | Path) -> Map:
        """
        Read the requested box of a NOAH-H interpretative groups GeoTIFF and
        return the block-averaged eta grid with its spatial metadata.
        """
        with rasterio.open(data) as source:
            if source.count != 1:
                raise ValueError("Expected a terrain class raster with exactly one band.")

            if source.crs is None:
                raise ValueError(
                    "The terrain class raster must define a coordinate reference system."
                )

            transform = source.transform

            # Map assumes x increases rightwards and y decreases downwards.
            if (
                transform.b != 0
                or transform.d != 0
                or transform.a <= 0
                or transform.e >= 0
            ):
                raise ValueError("Expected a north-up raster without rotation.")

            source_crs = CRS.from_wkt(source.crs.to_wkt())
            radius = self._sphere_radius(source_crs)
            geographic_crs = source_crs.geodetic_crs
            if geographic_crs is None:
                raise ValueError("The raster CRS must have a geodetic CRS.")
            to_source = Transformer.from_crs(geographic_crs, source_crs, always_xy=True)
            to_geographic = Transformer.from_crs(source_crs, geographic_crs, always_xy=True)

            # Requested box in projected metres, snapped to the source pixel grid.
            x_left, y_top = to_source.transform(*self.top_left, errcheck=True)
            x_right, y_bottom = to_source.transform(*self.bottom_right, errcheck=True)
            x_left = transform.c + round((x_left - transform.c) / transform.a) * transform.a
            y_top = transform.f + round((y_top - transform.f) / transform.e) * transform.e

            res = self.resolution_m
            n_cols = max(1, int(round((x_right - x_left) / res)))
            n_rows = max(1, int(round((y_top - y_bottom) / res)))
            f = self.factor
            step = self.INTERMEDIATE_RESOLUTION_M

            # Intermediate (1 m) cells fully inside the raster coverage.
            bounds = source.bounds
            eps = 1e-6
            k_lo = max(0, math.ceil((bounds.left - x_left) / step - eps))
            k_hi = min(n_cols * f, math.floor((bounds.right - x_left) / step + eps))
            m_lo = max(0, math.ceil((y_top - bounds.top) / step - eps))
            m_hi = min(n_rows * f, math.floor((y_top - bounds.bottom) / step + eps))
            if k_lo >= k_hi or m_lo >= m_hi:
                raise ValueError(
                    "The requested box lies entirely outside the terrain class raster."
                )

            lut = self._lookup_table(source.nodata)
            eta = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
            min_count = self.min_valid_fraction * f * f
            col_off = (x_left + k_lo * step - transform.c) / transform.a
            col_len = (k_hi - k_lo) * step / transform.a

            chunk_out_rows = max(1, self.MAX_CHUNK_ROWS // f)
            for r0 in range(0, n_rows, chunk_out_rows):
                r1 = min(n_rows, r0 + chunk_out_rows)
                a, b = max(m_lo, r0 * f), min(m_hi, r1 * f)
                if a >= b:
                    continue
                # Classes at 1 m; cells outside coverage stay nodata.
                classes = np.full(((r1 - r0) * f, n_cols * f),
                                  self.NODATA_CLASSES[1], dtype=np.uint8)
                row_off = (y_top - a * step - transform.f) / transform.e
                window = Window(
                    round(col_off), round(row_off),
                    round(col_len), round((b - a) * step / -transform.e),
                )
                classes[a - r0 * f:b - r0 * f, k_lo:k_hi] = source.read(
                    1, window=window, out_shape=(b - a, k_hi - k_lo),
                    resampling=Resampling.nearest,
                )

                fine = lut[classes].reshape(r1 - r0, f, n_cols, f)
                valid = np.isfinite(fine)
                count = valid.sum(axis=(1, 3))
                total = np.where(valid, fine, 0.0).sum(axis=(1, 3))
                keep = (count > 0) & (count >= min_count - 1e-9)
                with np.errstate(invalid="ignore", divide="ignore"):
                    eta[r0:r1] = np.where(keep, total / count, np.nan)

            # Output cell centres in (lon, lat) degrees.
            def geographic(x: float, y: float) -> tuple[float, float]:
                longitude, latitude = to_geographic.transform(x, y, errcheck=True)
                return float(longitude), float(latitude)

            top_left = geographic(x_left + res / 2, y_top - res / 2)
            bottom_right = geographic(
                x_left + (n_cols - 0.5) * res, y_top - (n_rows - 0.5) * res,
            )
            # eqc with lat_ts = 0: x = R * lon, y = R * lat, so this is exact.
            step_deg = math.degrees(res / radius)
            granularity = (step_deg, step_deg)

            metadata = json.dumps({
                "source": str(data),
                "source_crs": source_crs.to_wkt(),
                "geographic_crs": geographic_crs.to_wkt(),
                "coordinate_units": "degrees",
                "value_unit": self.VALUE_UNIT,
                "class_to_eta": {str(k): v for k, v in sorted(self.class_to_eta.items())},
                "class_names": {str(k): v for k, v in self.CLASS_NAMES.items()},
                "nodata_classes": list(self.NODATA_CLASSES),
                "resolution_m": res,
                "intermediate_resolution_m": step,
                "min_valid_fraction": self.min_valid_fraction,
                "citation": self.CITATION,
                "license": self.LICENSE,
                "note": (
                    "TEAM ASSUMPTION mapping from NOAH-H interpretative groups to "
                    "Pandolf eta (keys of src.models.constraints.ETA_TERRAIN)"
                ),
                "description": (
                    "Pandolf terrain coefficient eta from NOAH-H terrain classes of "
                    "Jezero crater, block-averaged from 1 m nearest-neighbour classes; "
                    "nodata, insufficient coverage and out-of-raster cells are NaN"
                ),
            })

            return Map(
                values=eta,
                granularity=granularity,
                top_left=top_left,
                bottom_right=bottom_right,
                metadata=metadata,
                value_type=Datatype.TERRAIN_ETA,
                title=(
                    f"Terrain Eta Map {top_left[0]} {top_left[1]} "
                    f"{bottom_right[0]} {bottom_right[1]}"
                ),
            )
