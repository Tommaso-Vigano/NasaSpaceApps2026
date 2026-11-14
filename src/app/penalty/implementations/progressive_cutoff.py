from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class ProgressiveCutoff(Penalty):
    """
    Increase the penalty linearly with the input's magnitude, then
    block values at or above the threshold.

    Formula:
        cost = weight * abs(value) / threshold, if abs(value) < threshold
        cost = infinity, otherwise

    Positive and negative values receive the same penalty.
    Missing or nonfinite values receive infinite cost.

    Example with threshold=30 and weight=1:
        0  -> 0.0
        15 -> 0.5
       -27 -> 0.9
        30 -> infinity
    """

    threshold: float
    weight: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive.")

        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and nonnegative.")

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        """Compute penalties for a scalar or an entire NumPy array."""

        # Use magnitudes: uphill and downhill slopes are treated equally.
        magnitudes = np.abs(np.asarray(values, dtype=np.float64))

        # Initially mark every value as blocked.
        costs = np.full(magnitudes.shape, np.inf, dtype=np.float64)

        # Only finite values strictly below the threshold are allowed.
        allowed = (
            np.isfinite(magnitudes)
            & (magnitudes < self.threshold)
        )

        # Normalize by the threshold and scale by the weight.
        costs[allowed] = (
            self.weight * magnitudes[allowed] / self.threshold
        )

        # Return a Python float for scalar input, otherwise an array.
        return costs.item() if costs.ndim == 0 else costs