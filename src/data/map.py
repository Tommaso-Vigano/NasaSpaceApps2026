from dataclasses import dataclass
from .types import SpatialUnit
import numpy as np
from numpy.typing import NDArray


Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


@dataclass
class Map:

    elevation: NDArray[np.float64]
    granularity: tuple[float, float]
    top_left: Point2D
    crs: str
    value_type: SpatialUnit

    def __post_init__(self) -> None:
        self.elevation = np.asarray(self.elevation, dtype=np.float64)

        if self.elevation.ndim != 2 or self.elevation.size == 0:
            raise ValueError("elevation must be a non-empty 2D array.")

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
        return self.elevation.shape
    @property
    def bottom_right(self) -> Point2D:
        """Return the centre of the bottom-right cell."""
        rows, columns = self.shape
        dx, dy = self.granularity
        x, y = self.top_left
        return x + (columns - 1) * dx, y - (rows - 1) * dy

    def get_point(self, row: int, column: int) -> Point3D:
        """Return (x, y, z). Missing elevation values remain NaN."""
        rows, columns = self.shape

        if not (0 <= row < rows and 0 <= column < columns):
            raise IndexError("Cell is outside the map.")

        dx, dy = self.granularity
        x = self.top_left[0] + column * dx
        y = self.top_left[1] - row * dy
        z = float(self.elevation[row, column])

        return x, y, z