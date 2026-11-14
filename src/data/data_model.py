from collections.abc import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.basic import Coordinate
from src.aliases import Point2D
from src.data.map import Map
from src.data.types import Datatype

class WorldModel:
    """Collection of maps, preserving insertion order and allowing duplicate titles.
    Queries return new lists containing the original Map objects.
    """

    def __init__(self, maps: Iterable[Map] = ()) -> None:
        self._maps: list[Map] = list(maps)

    def add_map(self, map_: Map) -> None:
        """Append a map to the collection."""
        self._maps.append(map_)

    def get_all_maps(self) -> list[Map]:
        """Return all maps in insertion order."""
        return list(self._maps)

    def get_maps_by_datatype(self, datatype: Datatype) -> list[Map]:
        """Return maps measuring the requested datatype."""
        return [map_ for map_ in self._maps if map_.value_type == datatype]

    def get_maps_by_title_contains(self, text: str) -> list[Map]:
        """Return maps whose titles contain text (case-sensitive)."""
        return [map_ for map_ in self._maps if text in map_.title]

    def get_map_by_title(self, title: str) -> Map | None:
        """Return the first map with this exact title, or None if absent."""
        return next((map_ for map_ in self._maps if map_.title == title), None)

    def get_maps_at_coordinate(
        self, coordinate: Coordinate | Point2D,
    ) -> list[Map]:
        """Return maps covering a Coordinate or a (longitude, latitude) tuple.

        Longitude uses the same convention as the stored corners.
        """
        if isinstance(coordinate, Coordinate):
            longitude, latitude = coordinate.lon, coordinate.lat
        else:
            longitude, latitude = coordinate

        return [
            map_ for map_ in self._maps
            if min(map_.top_left[0], map_.bottom_right[0])
            <= longitude <= max(map_.top_left[0], map_.bottom_right[0])
            and min(map_.top_left[1], map_.bottom_right[1])
            <= latitude <= max(map_.top_left[1], map_.bottom_right[1])
        ]

    def make_slice(self, top_left: Coordinate, bot_right: Coordinate) -> None:
        """Extract a slice from the current collection of maps and data."""


class SamplingUtils:

    @staticmethod
    def resample_to_window(
        m: Map,
        top_left: tuple[float, float],
        bottom_right: tuple[float, float],
        new_stride: float,
    ) -> Map:
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        # Target grid in world coordinates
        xs = np.arange(x_min, x_max, new_stride)
        ys = np.arange(y_min, y_max, new_stride)

        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        # Original grid in world coordinates
        h, w = m.data.shape[:2]

        old_x = m.top_left[0] + np.arange(w) * m.stride
        old_y = m.top_left[1] + np.arange(h) * m.stride

        # Interpolate, filling outside regions with NaN
        interpolator = RegularGridInterpolator(
            (old_y, old_x),
            m.data.astype(float),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        new_data = interpolator((yy, xx))

        return Map(
            data=new_data,
            stride=new_stride,
            top_left=top_left,
        )