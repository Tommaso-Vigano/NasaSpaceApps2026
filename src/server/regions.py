"""Small regional viewer: stable sector IDs and bounded TIFF window reads."""

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.data.types import Datatype


class Regions:
    def __init__(self, world_model):
        self.sectors = {}
        self._map_ids = {}
        self._source_maps = {}
        self._styles = {}
        for map_ in world_model.get_maps_by_datatype(Datatype.ELEVATION):
            try:
                path = Path(json.loads(map_.metadata)["source"]).resolve()
            except (ValueError, KeyError, TypeError):
                continue
            identity = f"{path}|{map_.title}|{map_.top_left}|{map_.bottom_right}"
            slug = re.sub(r"[^a-z0-9]+", "-", map_.title.lower()).strip("-") or "sector"
            sector_id = slug + "-" + sha256(identity.encode()).hexdigest()[:12]
            self.sectors[sector_id] = {"id": sector_id, "title": map_.title, "path": path,
                "west": map_.top_left[0], "north": map_.top_left[1],
                "east": map_.bottom_right[0], "south": map_.bottom_right[1]}
            self._map_ids[id(map_)] = sector_id
            self._source_maps[sector_id] = map_

    def source_map(self, sector_id):
        self.get(sector_id)
        return self._source_maps[sector_id]

    def id_for_map(self, map_):
        return self._map_ids.get(id(map_))

    def get(self, sector_id):
        if sector_id not in self.sectors:
            raise KeyError(sector_id)
        return self.sectors[sector_id]

    def select(self, sector_id, bounds):
        """Validate exactly one positive-area overlap and clip to its coverage."""
        sector = self.get(sector_id)
        if not bounds:
            bounds = {key: sector[key] for key in ("west", "south", "east", "north")}
        if (not all(math.isfinite(v) for v in bounds.values())
                or not -180 <= bounds["west"] < bounds["east"] <= 180
                or not -90 <= bounds["south"] < bounds["north"] <= 90):
            raise ValueError("Select a nonempty rectangle without crossing the antimeridian.")
        matches = [item for item in self.sectors.values()
                   if item["west"] < bounds["east"] and item["east"] > bounds["west"]
                   and item["south"] < bounds["north"] and item["north"] > bounds["south"]]
        if len(matches) != 1 or matches[0]["id"] != sector_id:
            raise ValueError("Select exactly one covered sector.")
        return {"west": max(bounds["west"], sector["west"]),
                "east": min(bounds["east"], sector["east"]),
                "south": max(bounds["south"], sector["south"]),
                "north": min(bounds["north"], sector["north"])}

    def info(self, sector_id, bounds):
        sector = self.get(sector_id)
        with rasterio.open(sector["path"]) as source:
            if source.count != 1 or source.crs is None:
                raise ValueError("The regional viewer requires a georeferenced single-band TIFF.")
            crs = CRS.from_wkt(source.crs.to_wkt())
            transformer = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
            # Sample the boundary, not just two corners, for projected rasters.
            t = np.linspace(0, 1, 17)
            xs = bounds["west"] + (bounds["east"] - bounds["west"]) * t
            ys = bounds["south"] + (bounds["north"] - bounds["south"]) * t
            lon = np.concatenate([xs, xs, np.full(17, bounds["west"]), np.full(17, bounds["east"])])
            lat = np.concatenate([np.full(17, bounds["north"]), np.full(17, bounds["south"]), ys, ys])
            x, y = transformer.transform(lon, lat, errcheck=True)
            inverse = ~source.transform
            cols = inverse.a * x + inverse.b * y + inverse.c
            rows = inverse.d * x + inverse.e * y + inverse.f
            left, right = np.clip([cols.min(), cols.max()], 0, source.width)
            top, bottom = np.clip([rows.min(), rows.max()], 0, source.height)
            if right <= left or bottom <= top:
                raise ValueError("Selected bounds contain no raster pixels.")
            to_geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
            grid_x, grid_y = np.meshgrid(np.linspace(0, source.width, 17),
                                        np.linspace(0, source.height, 17))
            world_x = source.transform.a*grid_x + source.transform.b*grid_y + source.transform.c
            world_y = source.transform.d*grid_x + source.transform.e*grid_y + source.transform.f
            grid_lon, grid_lat = to_geographic.transform(world_x, world_y)
            coordinate_grid = np.stack((grid_lon, grid_lat), axis=-1).tolist()
            return {"coordinate_grid": coordinate_grid, "id": sector_id, "title": sector["title"], "width": source.width,
                    "height": source.height, "bounds": bounds,
                    "initial_view": {"x": float(left), "y": float(top),
                                     "width": float(right-left), "height": float(bottom-top)}}

    def frame(self, sector_id, x, y, width, height, pixels=1024):
        if (not all(math.isfinite(v) for v in (x, y, width, height))
                or width <= 0 or height <= 0 or not 1 <= pixels <= 1024):
            raise ValueError("Invalid viewport or output size (maximum 1024 pixels per side).")
        sector = self.get(sector_id)
        with rasterio.open(sector["path"]) as source:
            left, top = max(0, x), max(0, y)
            right, bottom = min(source.width, x + width), min(source.height, y + height)
            if right <= left or bottom <= top:
                raise ValueError("Viewport is outside the sector.")
            window = Window(left, top, right-left, bottom-top)
            ratio = min(pixels / window.width, pixels / window.height)
            out_w, out_h = max(1, round(window.width*ratio)), max(1, round(window.height*ratio))
            # Every request reads a TIFF window directly, never the WorldModel array.
            values = source.read(1, window=window, out_shape=(out_h, out_w),
                                 masked=True, resampling=Resampling.average).astype(float)
            values = (values * source.scales[0] + source.offsets[0]).filled(np.nan)
            # A tiny overview gives every viewport the same display contrast.
            version = (sector["path"].stat().st_mtime_ns, source.width, source.height)
            cached = self._styles.get(sector_id)
            if cached is None or cached[0] != version:
                sample = source.read(1, out_shape=(min(128, source.height), min(128, source.width)),
                                     masked=True, resampling=Resampling.average).astype(float)
                sample = (sample * source.scales[0] + source.offsets[0]).filled(np.nan)
                valid = sample[np.isfinite(sample)]
                limits = np.percentile(valid, [2, 98]) if valid.size else [0, 1]
                self._styles[sector_id] = (version, limits)
            low, high = self._styles[sector_id][1]
            valid = np.isfinite(values)
            grey = np.zeros(values.shape, dtype=np.uint8)
            if high > low:
                grey[valid] = np.round(np.clip((values[valid]-low)/(high-low), 0, 1)*255).astype(np.uint8)
            else:
                grey[valid] = 160
            rgba = np.stack((grey, grey, grey, valid.astype(np.uint8)*255), axis=-1)
            buffer = BytesIO()
            Image.fromarray(rgba).save(buffer, format="PNG")
            return buffer.getvalue(), {"x": left, "y": top, "width": right-left, "height": bottom-top}
