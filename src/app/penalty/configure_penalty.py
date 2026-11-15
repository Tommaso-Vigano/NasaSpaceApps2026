from src.data.types import Datatype

from .penalty_manager import PenaltyManager
from .implementations.cutoff_penalty import CutOffPenalty
from .implementations.weighing_penalty import WeightPenalty
from .implementations.soft_threshold_penalty import SoftThresholdPenalty


def configure_penalties(
    manager: PenaltyManager,
    temperature_limit: float,
    albedo_weight: float
) -> None:
    """Configure the penalty selector for each map category."""

    # CUT-OFF:
    # Temperature below the limit -> zero penalty.
    # Temperature at/above the limit -> blocked.
    manager.register_penalty(
        Datatype.TEMPERATURE,
        lambda profile: CutOffPenalty(
            threshold=temperature_limit,
        ),
    )

    # WEIGHTED:
    # Cost = albedo_weight * albedo.
    # This particular rule makes higher albedo more costly.
    manager.register_penalty(
        Datatype.ALBEDO,
        lambda profile: WeightPenalty(
            weight=albedo_weight,
        ),
    )
