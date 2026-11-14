from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class SoftThresholdPenalty(Penalty):
    """ values below a threshold cost nothing; above it, the cost increases gradually:
        C(v) = w*max(0;v-t) """
    threshold: float
    weight: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")

        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and nonnegative.")

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        costs = np.full(values.shape, np.inf, dtype=np.float64)
        valid = np.isfinite(values)

        costs[valid] = self.weight * np.maximum(
            0.0,
            values[valid] - self.threshold,
        )

        return costs.item() if costs.ndim == 0 else costs