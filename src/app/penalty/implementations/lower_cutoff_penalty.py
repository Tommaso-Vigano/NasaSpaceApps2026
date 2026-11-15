from dataclasses import dataclass

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class LowerCutOffPenalty(Penalty):
    """Mirror of CutOffPenalty, for data where the LOW value is the dangerous one
    (thermal inertia: loose soil; solar flux: not enough light).

        cost = 0,   if value >= threshold
        cost = inf, if value < threshold or value is NaN

    The threshold itself is allowed, as in src/models: gates.daylight_window
    (flux >= 100 W/m^2 = GO) and rover.rover_hard_mask (TI >= 100 = ok).
    """
    threshold: float

    def __post_init__(self):
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        costs = np.where(
            np.isfinite(values) & (values >= self.threshold),
            0.0,
            np.inf,
        )

        return costs.item() if costs.ndim == 0 else costs
