from dataclasses import dataclass, field

import numpy as np

from .penalty import Penalty, PenaltyInput


@dataclass
class CombinedPenalty(Penalty):
    """Element-wise maximum of several penalties.

    Used for an expedition: on every cell the most limited participant decides
    (inf if any member forbids the cell)."""
    penalties: list[Penalty] = field(default_factory=list)

    def __post_init__(self):
        self.penalties = list(self.penalties)
        if not self.penalties:
            raise ValueError("CombinedPenalty needs at least one penalty.")

    def compute(self, values: PenaltyInput) -> PenaltyInput:
        values = np.asarray(values, dtype=np.float64)

        costs = np.full(values.shape, -np.inf, dtype=np.float64)
        for penalty in self.penalties:
            member = np.asarray(penalty.compute(values), dtype=np.float64)
            costs = np.maximum(costs, np.broadcast_to(member, values.shape))

        return costs.item() if costs.ndim == 0 else costs
