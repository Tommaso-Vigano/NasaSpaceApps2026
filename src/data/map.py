from dataclasses import dataclass
from .types import SpatialUnit

import numpy as np
from numpy.typing import NDArray

"""
Dataclass for storing the parsed information about the map
"""
Point3D = tuple[float, float, float]


@dataclass
class Map:
    tensor: NDArray[np.float64]
    granularity: float
    top_left: Point3D
    bottom_right: Point3D
    value_type: SpatialUnit

    def __post_init__(self) -> None:
        self.tensor = np.asarray(self.tensor, dtype=np.float64)

        if self.tensor.ndim != 3:
            raise ValueError("tensor must have three dimensions: (x, y, z).")

        if not np.isfinite(self.granularity) or self.granularity <= 0:
            raise ValueError("granularity must be finite and positive.")

        for name in ("top_left", "bottom_right"):
            point = np.asarray(getattr(self, name), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{name} must contain three finite coordinates.")
            setattr(self, name, tuple(point.tolist()))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.tensor.shape
