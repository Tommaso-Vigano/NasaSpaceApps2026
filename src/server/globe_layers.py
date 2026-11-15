"""Cached, georeferenced display textures; independent of routing datasets."""

from io import BytesIO
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import ColorInterp, Resampling
from scipy.ndimage import map_coordinates


LAYER_SPECS = {
    "albedo": {"label": "Albedo", "filename": "Mars_MGS_TES_Albedo_mosaic_global_7410m.tif",
               "units": "", "colors": ["#262a32", "#898b8c", "#f4eee0"]},
    "thermal-inertia": {"label": "Thermal inertia", "filename": "TES_Thermal_Inertia.tif",
                        "units": "J m^-2 K^-1 s^-1/2",
                        "colors": ["#161536", "#643c83", "#c45b68", "#eea35a", "#fcf1bb"]},
}


class GlobeLayers:
    def __init__(self, directory: str | Path, width: int = 2048) -> None:
        self.directory = Path(directory)
        self.width = width
        self._cache = {}
        self._lock = RLock()

    def catalog(self) -> list[dict]:
        return [{"id": id_, "label": spec["label"],
                 "available": (self.directory / spec["filename"]).is_file()}
                for id_, spec in LAYER_SPECS.items()]

    def render(self, layer_id: str) -> tuple[bytes, dict]:
        """Return a PNG and legend, refreshing the cache when the TIFF changes."""
        spec = LAYER_SPECS[layer_id]
        path = self.directory / spec["filename"]
        stat = path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(layer_id)
            if cached and cached[0] == version:
                return cached[1], cached[2]
            grid, units = self._geographic_grid(path)
            is_rgb = grid.ndim == 3
            valid = np.isfinite(grid).all(axis=-1) if is_rgb else np.isfinite(grid)
            if not valid.any():
                raise ValueError("The raster contains no valid geographic samples.")
            low = high = None
            if is_rgb:
                # These are display colours, not recoverable physical values.
                rgb = np.clip(np.nan_to_num(grid), 0, 255).astype(np.uint8)
            else:
                low, high = np.percentile(grid[valid], [2, 98])
                if low == high:
                    low, high = float(grid[valid].min()), float(grid[valid].max())
                normalized = np.zeros(grid.shape)
                if high > low:
                    normalized[valid] = np.clip((grid[valid] - low) / (high - low), 0, 1)
                palette = np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)]
                                    for color in spec["colors"]])
                stops = np.linspace(0, 1, len(palette))
                rgb = np.stack([np.interp(normalized, stops, palette[:, channel])
                                for channel in range(3)], axis=-1).astype(np.uint8)
            rgb[~valid] = (30, 33, 41)
            buffer = BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            metadata = {"id": layer_id, "label": spec["label"],
                        "minimum": None if is_rgb else float(low), "maximum": None if is_rgb else float(high),
                        "legend_type": "image" if is_rgb else "continuous",
                        "units": "" if is_rgb else (units or spec["units"]),
                        "colors": [] if is_rgb else spec["colors"],
                        "stretch": "Source image colours" if is_rgb else "2nd to 98th percentile (values outside are clipped)",
                        "missing_color": "#1e2129", "center_longitude": 180,
                        "version": str(stat.st_mtime_ns)}
            png = buffer.getvalue()
            self._cache[layer_id] = (version, png, metadata)
            return png, metadata

    def _geographic_grid(self, path: Path) -> tuple[np.ndarray, str | None]:
        """Sample a 0..360 East / north-up texture using the raster's own body CRS.

        Reading is downsampled to bound memory. Destination coordinates are
        inverse-transformed in row blocks; alternate longitude branches support
        both signed and 0..360 rasters without assuming an Earth datum.
        """
        height = self.width // 2
        with rasterio.open(path) as source:
            is_rgb = source.count in (3, 4) and source.colorinterp[:3] == (
                ColorInterp.red, ColorInterp.green, ColorInterp.blue,
            )
            if source.crs is None or (source.count != 1 and not is_rgb):
                raise ValueError("Globe layers require a scalar or RGB(A) raster with a CRS.")
            if is_rgb and any(dtype != "uint8" for dtype in source.dtypes[:3]):
                raise ValueError("RGB globe layers must use 8-bit display colours.")
            output = np.full((height, self.width, 3) if is_rgb else (height, self.width), np.nan)
            scale = min(1, 4096 / max(source.width, source.height))
            w, h = max(1, round(source.width * scale)), max(1, round(source.height * scale))
            # Read nodata masks and explicitly honour alpha for RGB imagery.
            values = source.read([1, 2, 3] if is_rgb else 1,
                                 out_shape=(3, h, w) if is_rgb else (h, w), masked=True,
                                 resampling=Resampling.average).astype(float)
            if not is_rgb:
                values = values * source.scales[0] + source.offsets[0]
            values = values.filled(np.nan)
            if is_rgb and ColorInterp.alpha in source.colorinterp:
                alpha = source.read(source.colorinterp.index(ColorInterp.alpha) + 1,
                                    out_shape=(h, w), resampling=Resampling.nearest)
                values[:, alpha == 0] = np.nan
            finite = np.isfinite(values).all(axis=0) if is_rgb else np.isfinite(values)
            numerator = np.where(np.isfinite(values), values, 0)
            support_values = finite.astype(float)
            if source.transform.determinant == 0:
                raise ValueError("The raster transform must be invertible.")
            inverse = ~source.transform
            crs = CRS.from_wkt(source.crs.to_wkt())
            if crs.geodetic_crs is None:
                raise ValueError("The layer CRS has no geodetic coordinate system.")
            transformer = Transformer.from_crs(crs.geodetic_crs, crs,
                                               always_xy=True, force_over=True)
            longitude = (np.arange(self.width) + 0.5) * 360 / self.width
            longitude = (longitude + 180) % 360 - 180
            for start in range(0, height, 64):
                end = min(height, start + 64)
                latitude = 90 - (np.arange(start, end) + 0.5) * 180 / height
                lons, lats = np.meshgrid(longitude, latitude)
                block = output[start:end]
                for offset in (0, 360, -360):
                    x, y = transformer.transform(lons + offset, lats)
                    columns = (inverse.a * x + inverse.b * y + inverse.c) * w / source.width - 0.5
                    rows = (inverse.d * x + inverse.e * y + inverse.f) * h / source.height - 0.5
                    inside = (np.isfinite(rows) & np.isfinite(columns)
                              & (rows >= -0.5) & (rows <= h - 0.5)
                              & (columns >= -0.5) & (columns <= w - 0.5)
                              & ~(np.isfinite(block).all(axis=-1) if is_rgb else np.isfinite(block)))
                    if not inside.any():
                        continue
                    points = [np.clip(rows[inside], 0, h - 1), np.clip(columns[inside], 0, w - 1)]
                    if is_rgb:
                        sampled = np.stack([map_coordinates(channel, points, order=1,
                                                            mode="nearest", prefilter=False)
                                            for channel in numerator], axis=-1)
                    else:
                        sampled = map_coordinates(numerator, points, order=1, mode="nearest", prefilter=False)
                    support = map_coordinates(support_values, points, order=1,
                                              mode="nearest", prefilter=False)
                    supported = support >= 1 - 1e-8
                    block[inside] = np.where(supported[:, None] if is_rgb else supported, sampled, np.nan)
            units = source.units[0]
            if (not is_rgb and not units and source.dtypes[0] == "uint8"
                    and source.scales[0] == 1 and source.offsets[0] == 0):
                units = "DN"
            return output, units
