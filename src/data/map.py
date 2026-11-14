from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.aliases import Point2D
from src.data.types import Datatype


@dataclass
class Map:
    """Scalar grid with geographic cell centres in (longitude, latitude).

    granularity is the approximate angular step (dlongitude, dlatitude)
    in degrees, positive rightwards and downwards respectively. Corners
    describe the centres of the first and last cells.
    """

    values: NDArray[np.float64]
    granularity: tuple[float, float]
    top_left: Point2D
    bottom_right: Point2D
    metadata: str
    value_type: Datatype

    title: str

