

from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class WeightPenalty(Penalty):
    """
    Compute weight * severity.

    severity converts measurements into nonnegative cost values.
    If omitted, the source values are used directly.
    """

    weight: float

    def __post_init__(self):
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and nonnegative.")

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        if np.any(values < 0):
            raise ValueError("Weighted penalty inputs must be nonnegative.")

        # Missing values remain blocked, including when weight is zero.
        costs = np.full(values.shape, np.inf, dtype=np.float64)
        valid = np.isfinite(values)
        costs[valid] = self.weight * values[valid]

        return costs.item() if costs.ndim == 0 else costs