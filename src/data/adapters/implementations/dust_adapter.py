import colorsys
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from src.data.adapters.adapter import Adapter
from src.data.map import Map
from src.data.types import Datatype
from src.data.adapters.registry import AdapterRegistry


@AdapterRegistry.register_adapter("Global-dust")
class DustAdapter(Adapter):
    """
    Estimate dust indices from the original RGB dust-map TIFF.
    Calibrated using https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2001JE001580
    Returns approximate band midpoints, not original measurements.
    Coordinates remain in degrees without reprojection. Out-of-range latitude
    rows are removed; longitude columns are wrapped and reordered if needed.
    """

    # Colour families: red, orange, yellow, green, cyan, blue, magenta.
    REFERENCE_HUES = np.array(
        [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    )

    DUST_INDICES = np.array(
        [0.925, 0.935, 0.945, 0.955, 0.965, 0.975, 0.985],
        dtype=np.float64,
    )

    @classmethod
    def _rgb_to_dust_index(cls, rgb: np.ndarray) -> np.ndarray:
        """Convert an RGB array shaped (3, rows, columns) into indices."""

        # Encode each RGB colour as one integer.
        # This lets us convert each distinct colour only once.
        packed = (
            (rgb[0].astype(np.uint32) << 16)
            | (rgb[1].astype(np.uint32) << 8)
            | rgb[2].astype(np.uint32)
        )

        unique, inverse = np.unique(packed, return_inverse=True)

        colours = np.column_stack(
            (
                (unique >> 16) & 255,
                (unique >> 8) & 255,
                unique & 255,
            )
        ).astype(np.float64)

        # HSV separates colour family (hue) from saturation and brightness.
        hsv = np.array([
            colorsys.rgb_to_hsv(*colour)
            for colour in colours / 255.0
        ])

        hue = hsv[:, 0] * 360.0
        saturation = hsv[:, 1]
        chroma = colours.max(axis=1) - colours.min(axis=1)

        # Hue is circular: 359 degrees is close to 0 degrees.
        difference = np.abs(
            hue[:, None] - cls.REFERENCE_HUES[None, :]
        )
        distance = np.minimum(difference, 360.0 - difference)

        nearest_colour = distance.argmin(axis=1)

        # Conservative matching: exclude grey and ambiguous colours.
        accepted = (
            (chroma >= 30.0)
            & (saturation >= 0.15)
            & (distance.min(axis=1) <= 10.0)
        )

        lookup = np.full(unique.size, np.nan, dtype=np.float64)
        lookup[accepted] = cls.DUST_INDICES[
            nearest_colour[accepted]
        ]

        return lookup[inverse].reshape(packed.shape)

    def adapt(self, data: str | Path) -> Map:
        with rasterio.open(data) as source:
            if (
                source.count != 3
                or any(dtype != "uint8" for dtype in source.dtypes)
                or source.colorinterp != (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                )
            ):
                raise ValueError(
                    "Expected an 8-bit, three-band RGB dust map."
                )

            if source.crs is None:
                raise ValueError("The raster has no coordinate reference system.")

            transform = source.transform

            if (
                transform.b != 0
                or transform.d != 0
                or transform.a <= 0
                or transform.e >= 0
            ):
                raise ValueError(
                    "Expected a north-up raster without rotation."
                )

            if not source.crs.is_geographic:
                raise ValueError(
                    "Expected a dust map already in geographic coordinates."
                )

            values = self._rgb_to_dust_index(source.read())

            # A pixel is valid only if all three RGB bands are valid.
            valid = np.all(source.read_masks() > 0, axis=0)
            values[~valid] = np.nan

            # Corners represent cell centres, consistent with MapAdapter.
            x, y = source.xy(0, 0, offset="center")
            dx, dy = float(transform.a), float(-transform.e)
            longitudes = float(x) + np.arange(source.width) * dx
            latitudes = float(y) - np.arange(source.height) * dy

            # Do not clamp invalid rows onto the poles: that would break spacing.
            latitude_rows = np.flatnonzero((latitudes >= -90) & (latitudes <= 90))
            if latitude_rows.size == 0:
                raise ValueError("The dust map has no latitude centres within [-90, 90].")
            values = values[latitude_rows, :]
            latitudes = latitudes[latitude_rows]
            if np.any((longitudes < -180) | (longitudes > 180)):
                longitudes = (longitudes + 180) % 360 - 180
                order = np.argsort(longitudes)
                longitudes = longitudes[order]
                values = values[:, order]
                if not np.allclose(np.diff(longitudes), dx, rtol=0, atol=1e-8):
                    raise ValueError(
                        "Wrapped dust longitudes do not form one regular grid; "
                        "split patches crossing the antimeridian before adapting."
                    )
            top_left = (float(longitudes[0]), float(latitudes[0]))
            bottom_right = (float(longitudes[-1]), float(latitudes[-1]))

            metadata = json.dumps({
                "source": str(data),
                "source_crs": source.crs.to_wkt(),
                "coordinate_units": "degrees",
                "longitude_range": "[-180, 180]",
                "removed_latitude_rows": int(source.height - latitude_rows.size),
                "tags": source.tags(),
                "description": "Approximate dust indices estimated from RGB colours",
            })

            return Map(
                values=values,
                granularity=(float(transform.a), float(-transform.e)),
                top_left=top_left,
                bottom_right=bottom_right,
                metadata=metadata,
                value_type=Datatype.DUST,
                title=f"Dust Map {top_left[0]} {top_left[1]} {bottom_right[0]} {bottom_right[1]}",
            )
