"""Build routing grids only when requested and adapt executor results for the map."""

import math

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window

from src.app.routing_executor import RoutingExecutor
from src.basic import Coordinate
from src.data.data_model import WorldModel
from src.routing.profiles import HumanProfile


class NoRouteError(ValueError):
    pass


def route_region(regions, sector_id, payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object with start, end and subdivisions.")
    subdivisions = payload.get("subdivisions")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or not 10 <= subdivisions <= 1000:
        raise ValueError("Subdivisions must be an integer from 10 to 1000.")
    sector = regions.get(sector_id)
    reference = regions.source_map(sector_id)
    rows, cols = reference.values.shape
    if rows < 2 or cols < 2:
        raise ValueError("Routing requires at least two rows and columns.")
    points = []
    with rasterio.open(sector["path"]) as source:
        crs = CRS.from_wkt(source.crs.to_wkt())
        geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        projected = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        inverse = ~source.transform
        for name in ("start", "end"):
            point = payload.get(name)
            if not isinstance(point, dict):
                raise ValueError(f"{name} must contain source pixel x and y.")
            x, y = point.get("x"), point.get("y")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
                raise ValueError(f"{name} coordinates must be finite numbers.")
            if not (0 <= x < source.width and 0 <= y < source.height):
                raise ValueError(f"{name} is outside the sector.")
            sample = source.read(1, window=Window(int(x), int(y), 1, 1), masked=True)
            if np.ma.is_masked(sample) or not np.isfinite(sample).all():
                raise ValueError(f"{name} is on missing terrain data. Choose another point.")
            # Source pixels describe areas; route endpoints use cell centres.
            x, y = np.clip(x, 0.5, source.width-0.5), np.clip(y, 0.5, source.height-0.5)
            transform = source.transform
            wx = transform.a*x + transform.b*y + transform.c
            wy = transform.d*x + transform.e*y + transform.f
            lon, lat = geographic.transform(wx, wy, errcheck=True)
            points.append(Coordinate(
                lat=float(np.clip(lat, reference.bottom_right[1], reference.top_left[1])),
                lon=float(np.clip(lon, reference.top_left[0], reference.bottom_right[0])),
            ))
        dx, dy = reference.granularity
        # N divisions require N+1 nodes. Never invent detail finer than native data.
        precision = ((cols-1)*dx/min(subdivisions, cols-1),
                     (rows-1)*dy/min(subdivisions, rows-1))
        # Other layers currently have no registered default cost selectors.
        # Elevation-only routing applies the existing human slope policy.
        world_slice = WorldModel([reference]).make_slice(
            reference.top_left, reference.bottom_right, precision,
        )
        elevation = next(iter(world_slice))
        executor = RoutingExecutor(Coordinate(lat=reference.top_left[1], lon=reference.top_left[0]),
                                   Coordinate(lat=reference.bottom_right[1], lon=reference.bottom_right[0]),
                                   HumanProfile())

        executor.penalty_manager.configure_from_json(
            executor.profile,
            payload.get("penalty_config", {"penalties": {"SLOPE": True}}),
        )
        result = executor.execute(world_slice, *points, elevation_map=elevation, compact=True)
        if not result.found:
            raise NoRouteError("No traversable route was found at this fineness. Try different endpoints or subdivisions.")
        coordinates = np.asarray(result.coordinates)
        wx, wy = projected.transform(coordinates[:, 0], coordinates[:, 1], errcheck=True)
        pixel_x = inverse.a*wx + inverse.b*wy + inverse.c
        pixel_y = inverse.d*wx + inverse.e*wy + inverse.f
        return {"path": np.stack((pixel_x, pixel_y), axis=-1).tolist(),
                "coordinates": result.coordinates, "total_cost": result.total_cost,
                "grid_shape": list(elevation.values.shape), "granularity": list(precision),
                "subdivisions": subdivisions, "profile": "human", "layers": ["elevation"]}
