from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.data.map import Map
from src.routing.profiles import Profile

from .penalty_type import (
    PenaltyType,
    ProfileParameter,
    resolve_parameter,
)


@dataclass(frozen=True)
class WeightedPenalty(PenaltyType):
    """
    Compute weight * severity.

    severity converts measurements into nonnegative cost values.
    If omitted, the source values are used directly.
    """

    weight: ProfileParameter = 1.0
    severity: Callable[
        [Map, Profile], NDArray[np.float64]
    ] | None = None

    def compute(
        self,
        source: Map,
        profile: Profile,
    ) -> NDArray[np.float64]:
        weight = resolve_parameter(self.weight, profile)

        if weight < 0:
            raise ValueError("weight must be nonnegative.")

        raw = np.asarray(source.values, dtype=np.float64)

        severity = np.asarray(
            self.severity(source, profile)
            if self.severity is not None
            else raw,
            dtype=np.float64,
        )

        if np.any(severity < 0):
            raise ValueError("severity must be nonnegative.")

        valid = np.isfinite(raw) & np.isfinite(severity)
        costs = np.full(raw.shape, np.inf, dtype=np.float64)

        with np.errstate(over="raise", invalid="raise"):
            costs[valid] = weight * severity[valid]

        return costs