from collections.abc import Iterable, Iterator
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.basic import Coordinate
from src.aliases import Point2D
from src.data.map import Map
from src.data.types import Datatype

class WorldSlice:
    """ Iterable collection of equally shaped maps produced by a world slice. """

    def __init__(self, maps: Iterable[Map] = ()) -> None:
        self._maps = tuple(maps)
        if self._maps:
            shape = self._maps[0].values.shape
            if any(map_.values.shape != shape for map_ in self._maps):
                raise ValueError("All maps in a WorldSlice must have the same shape since they are all sampled to the "
                                 "same resolution.")

    def __iter__(self) -> Iterator[Map]:
        return iter(self._maps)

    @property
    def shape(self) -> tuple[int, int]:
        """ Return the shared (rows, columns), or (0, 0) for an empty slice. """
        return self._maps[0].values.shape if self._maps else (0, 0)


class WorldModel:
    """ Collection of maps, preserving insertion order and allowing duplicate titles.
    Queries return new lists containing the original Map objects.
    """

    def __init__(self, maps: Iterable[Map] = ()) -> None:
        self._maps: list[Map] = list(maps)

    def add_map(self, map_: Map) -> None:
        """ Append a map to the collection. """
        self._maps.append(map_)

    def get_all_maps(self) -> list[Map]:
        """ Return all maps in insertion order. """
        return list(self._maps)

    def get_maps_by_datatype(self, datatype: Datatype) -> list[Map]:
        """ Return maps measuring the requested datatype. """
        return [map_ for map_ in self._maps if map_.value_type == datatype]

    def get_maps_by_title_contains(self, text: str) -> list[Map]:
        """ Return maps whose titles contain text (case-sensitive). """
        return [map_ for map_ in self._maps if text in map_.title]

    def get_map_by_title(self, title: str) -> Map | None:
        """ Return the first map with this exact title, or None if absent. """
        return next((map_ for map_ in self._maps if map_.title == title), None)

    def get_maps_at_coordinate(
        self, coordinate: Coordinate | Point2D,
    ) -> list[Map]:
        """ Return maps covering a Coordinate or a (longitude, latitude) tuple
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

    def make_slice(
        self,
        top_left: Coordinate | Point2D,
        bottom_right: Coordinate | Point2D,
        resolution: float | Point2D,
    ) -> WorldSlice:
        """Return a new view with one map per intersecting datatype.

        Resolution is a positive angular spacing in degrees (scalar or dx, dy).
        The grid starts at top_left and includes bottom_right only when aligned
        with that spacing,  otherwise it stops at the last cell centre inside.
        Patches are interpolated independently, then finite overlapping samples
        are averaged with equal weight. Uncovered cells remain NaN.
        """
        xs, ys, spacing = SamplingUtils.window_grid(top_left, bottom_right, resolution)
        groups: dict[Datatype, list[Map]] = {}
        left, top = SamplingUtils.point(top_left)
        right, bottom = SamplingUtils.point(bottom_right)
        for map_ in self._maps:
            if (map_.top_left[0] <= right and map_.bottom_right[0] >= left
                    and map_.top_left[1] >= bottom and map_.bottom_right[1] <= top):
                groups.setdefault(map_.value_type, []).append(map_)
        return WorldSlice(
            SamplingUtils._merge_on_grid(patches, xs, ys, spacing)
            for patches in groups.values()
        )


class SamplingUtils:
    """Resampling of north-up geographic grids with cell-centre bounds.

    Coordinates must use a common longitude convention and geodetic CRS.
    Rectangles crossing the antimeridian must be split before sampling.
    """

    @staticmethod
    def point(coordinate: Coordinate | Point2D) -> Point2D:
        """Normalize a coordinate to finite (longitude, latitude)."""
        point = ((coordinate.lon, coordinate.lat)
                 if isinstance(coordinate, Coordinate) else coordinate)
        values = np.asarray(point, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("Coordinates must contain two finite values.")
        return float(values[0]), float(values[1])

    @staticmethod
    def window_grid(
        top_left: Coordinate | Point2D,
        bottom_right: Coordinate | Point2D,
        resolution: float | Point2D,
    ) -> tuple[np.ndarray, np.ndarray, Point2D]:
        """Build inclusive, regularly spaced axes inside the requested window.

        Zero-width or zero-height windows produce a single column or row.
        The requested spacing is preserved even when the extent is not divisible.
        """
        left, top = SamplingUtils.point(top_left)
        right, bottom = SamplingUtils.point(bottom_right)
        if right < left or bottom > top:
            raise ValueError("Expected a top-left and bottom-right ordered rectangle.")
        spacing = np.asarray(resolution, dtype=float)
        if spacing.ndim == 0:
            spacing = np.repeat(spacing, 2)
        if (spacing.shape != (2,) or not np.all(np.isfinite(spacing))
                or np.any(spacing <= 0)):
            raise ValueError("Resolution must be a finite positive scalar or (dx, dy).")
        dx, dy = float(spacing[0]), float(spacing[1])

        def axis(start: float, extent: float, step: float) -> np.ndarray:
            intervals = extent / step
            if not np.isfinite(intervals):
                raise ValueError("Window size divided by resolution must be finite.")
            # Tolerate rounding of an otherwise integral number of intervals.
            nearest = round(intervals)
            if np.isclose(intervals, nearest, rtol=0, atol=1e-10):
                intervals = nearest
            count = int(np.floor(intervals)) + 1
            return start + np.arange(count, dtype=float) * step

        xs = axis(left, right - left, dx)
        # Latitude decreases as the row index increases.
        ys = top - axis(0.0, top - bottom, dy)
        return xs, ys, (dx, dy)

    @staticmethod
    def sample_on_grid(m: Map, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """ Bilinearly interpolate a patch without extrapolation or filling holes.

        Only nonzero interpolation weights require finite source values, so a
        neighbouring NaN does not erase an exact valid source-cell sample.
        Singleton source axes are supported only at their stored coordinate.
        """
        values = np.asarray(m.values, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("Map values must be a nonempty two-dimensional array.")

        # Extract the point corresponding to the top left and bottom right corners
        left, top = SamplingUtils.point(m.top_left)
        right, bottom = SamplingUtils.point(m.bottom_right)
        spacing = np.asarray(m.granularity, dtype=float)

        if (spacing.shape != (2,) or not np.all(np.isfinite(spacing))
                or np.any(spacing <= 0)):
            raise ValueError("Map granularity must contain finite positive dx and dy for coordinates.")

        rows, columns = values.shape
        dx, dy = spacing
        expected = (left + (columns - 1) * dx, top - (rows - 1) * dy)
        if not np.allclose((right, bottom), expected, rtol=0, atol=1e-8):
            raise ValueError("Map corners, granularity and array shape are inconsistent so it was impossible"
                             "to sample the values on a grid.")

        old_x = left + np.arange(columns) * dx
        old_y = top - np.arange(rows) * dy
        result = np.full((ys.size, xs.size), np.nan)

        # Restrict queries to the patch, allowing only floating-point roundoff.
        tolerance = 32 * np.finfo(float).eps * max(
            1.0, abs(left), abs(right), abs(top), abs(bottom),
        )
        xi = np.flatnonzero((xs >= left - tolerance) & (xs <= right + tolerance))
        yi = np.flatnonzero((ys <= top + tolerance) & (ys >= bottom - tolerance))

        if xi.size == 0 or yi.size == 0:
            return result
        query_y, query_x = np.meshgrid(
            np.clip(ys[yi], bottom, top), np.clip(xs[xi], left, right), indexing="ij",
        )
        points = np.stack((query_y, query_x), axis=-1)
        valid = np.isfinite(values)

        # Zero-fill only for arithmetic; a second interpolator tracks whether
        # all contributing measurements were actually present.
        numerator = RegularGridInterpolator(
            (old_y, old_x), np.where(valid, values, 0.0),
            bounds_error=False, fill_value=None,
        )(points)
        support = RegularGridInterpolator(
            (old_y, old_x), valid.astype(float),
            bounds_error=False, fill_value=None,
        )(points)

        sampled = np.where(np.isclose(support, 1.0, rtol=0, atol=1e-12),
                           numerator, np.nan)
        result[np.ix_(yi, xi)] = sampled
        return result

    @staticmethod
    def resample_to_window(
        m: Map,
        top_left: Coordinate | Point2D,
        bottom_right: Coordinate | Point2D,
        new_stride: float | Point2D,
    ) -> Map:
        """ Resample one map; outside coverage and missing support become NaN."""
        return SamplingUtils.merge_to_window([m], top_left, bottom_right, new_stride)

    @staticmethod
    def merge_to_window(
        maps: Iterable[Map],
        top_left: Coordinate | Point2D,
        bottom_right: Coordinate | Point2D,
        resolution: float | Point2D,
    ) -> Map:
        """ Resample patches of one datatype and average finite overlaps.

        At least one patch is required. Source maps are never modified.
        """
        xs, ys, spacing = SamplingUtils.window_grid(top_left, bottom_right, resolution)
        return SamplingUtils._merge_on_grid(list(maps), xs, ys, spacing)

    @staticmethod
    def _merge_on_grid(
        maps: list[Map], xs: np.ndarray, ys: np.ndarray, spacing: Point2D,
    ) -> Map:
        if not maps:
            raise ValueError("At least one map is required for merging.")
        datatype = maps[0].value_type
        if any(m.value_type != datatype for m in maps):
            raise ValueError("Only maps of the same datatype can be merged.")
        mean = np.zeros((ys.size, xs.size), dtype=float)
        counts = np.zeros(mean.shape, dtype=np.int64)
        for m in maps:
            sampled = SamplingUtils.sample_on_grid(m, xs, ys)
            valid = np.isfinite(sampled)
            counts[valid] += 1
            # Incremental averaging avoids retaining a stack of resampled maps.
            weight = 1.0 / counts[valid]
            mean[valid] = mean[valid] * (1.0 - weight) + sampled[valid] * weight
        mean[counts == 0] = np.nan

        return Map(
            values=mean,
            granularity=spacing,
            top_left=(float(xs[0]), float(ys[0])),
            bottom_right=(float(xs[-1]), float(ys[-1])),
            metadata=json.dumps({
                "operation": "bilinear resampling and mean of finite overlapping patches",
                "sources": [{"title": m.title, "metadata": m.metadata} for m in maps],
            }),
            value_type=datatype,
            title=f"{datatype.value} world slice from",
        )
