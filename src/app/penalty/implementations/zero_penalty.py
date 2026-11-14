from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class ZeroPenalty(Penalty):
    """0 everywhere, NaN included: for a layer that a profile does not use
    (e.g. the walk limits for a robotic rover, the thermal inertia for an astronaut).
    Every other template gives inf on NaN, which would turn missing data into obstacles."""

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        costs = np.zeros(values.shape, dtype=np.float64)

        return costs.item() if costs.ndim == 0 else costs
