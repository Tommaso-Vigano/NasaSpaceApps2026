from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


PenaltyInput = float | NDArray[np.float64]


class Penalty(ABC):
    """Convert numerical values into penalty costs."""

    @abstractmethod
    def compute(self, values: PenaltyInput) -> PenaltyInput:
        pass

    def __call__(self, values: PenaltyInput) -> PenaltyInput:
        """Allow penalty(value) as shorthand for penalty.compute(value)."""
        return self.compute(values)