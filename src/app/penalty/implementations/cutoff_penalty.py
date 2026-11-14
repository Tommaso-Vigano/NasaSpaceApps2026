from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class CutOffPenalty(Penalty):
    """ A simple penalty that penalizes when a threshold is excited otherwise let everything to 0"""
    threshold: float

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        costs = np.where(
            np.isfinite(values) & (values < self.threshold),
            0.0,
            np.inf,
        )

        return costs.item() if costs.ndim == 0 else costs