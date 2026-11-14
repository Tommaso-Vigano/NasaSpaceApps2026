from src.data.types import SpatialUnit
from src.aliases import Point2D, Point3D

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class Map:

    values: NDArray[np.float64]
    granularity: tuple[float, float]
    top_left: Point2D
    crs: str
    value_type: SpatialUnit

    def __post_init__(self) -> None:
        """ Initialize the map object by ensuring arrays are numpy arrays and spacing is """
        self.values = np.asarray(self.values, dtype=np.float64)

        if self.values.ndim != 2 or self.values.size == 0:
            raise ValueError("The scalar field must be a two dimensional map.")

        spacing = np.asarray(self.granularity, dtype=float)
        if (
            spacing.shape != (2,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError(
                "granularity must contain two finite, positive values: (dx, dy)."
            )
        self.granularity = (float(spacing[0]), float(spacing[1]))

        corner = np.asarray(self.top_left, dtype=float)
        if corner.shape != (2,) or not np.all(np.isfinite(corner)):
            raise ValueError("top_left must contain two finite coordinates.")
        self.top_left = (float(corner[0]), float(corner[1]))

    @property
    def shape(self) -> tuple[int, int]:
        """Return (number of rows, number of columns)."""
        return self.values.shape

    @property
    def bottom_right(self) -> Point2D:
        """Return the centre of the bottom-right cell."""
        rows, columns = self.shape
        dx, dy = self.granularity
        x, y = self.top_left
        return x + (columns - 1) * dx, y - (rows - 1) * dy

    """
    def get_point(self, row: int, column: int) -> Point3D:
        Return (x, y, z). Missing elevation values remain NaN.
        rows, columns = self.shape

        if not (0 <= row < rows and 0 <= column < columns):
            raise IndexError("Cell is outside the map.")

        dx, dy = self.granularity
        x = self.top_left[0] + column * dx
        y = self.top_left[1] - row * dy
        z = float(self.values[row, column])

        return x, y, z

    def get_geographic_point(
            self, row: int, column: int
    ) -> Point3D:
        Return (longitude, latitude, elevation) for a cell.
        from pyproj import CRS, Transformer

        x, y, z = self.get_point(row, column)

        source_crs = CRS.from_wkt(self.crs)

        transformer = Transformer.from_crs(
            source_crs,
            source_crs.geodetic_crs,
            always_xy=True,
        )

        longitude, latitude = transformer.transform(x, y)

        return float(longitude), float(latitude), z
    """