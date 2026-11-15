
from dataclasses import dataclass
from abc import ABCMeta
from enum import Enum


class ConstraintType(Enum):
    THRESHOLD = "Threshold"
    WEIGHED = "Weighed"


@dataclass()
class Profile(metaclass=ABCMeta):
    """ An interface for the profile for the routing algorithm """
    pass

"""
If HARD CONSTRAINTS are not respected -> NO-GO
If SOFT CONSTRAINTS are not respected -> increase cost

PARAMETERS are used for computations.
PATHS are used for computations along the path.
"""

@dataclass
class HumanProfile(Profile):
    """
    HARD
    - Slope: 20° max
    - Step up: 0.30 m
    - Step down: 0.22 m
    - Physiological limit: 5.6 km/h (VO₂max reached at 3.5 mph in Mars offload)
    PATH
    - Distance from lander: 2km
    - EVA duration: 8 h (6 ± 2 h)
    - Walkback radius: 1.8 km for a 30-min return; 3.5 km for 1 h (at 2.2 mph)
    PARAMETER
    - Speed: 3.5 km/h nominal; 2.7 km/h beyond 1 h
    - Suit cost: +14 ml/min/kg of VO₂
    - Metabolic rate (M1 calibration): 352 W average planetary EVA; 498 W walkback; 674 W walkback 95th percentile
    """
    hard_constraints = {
        "slope": 30.0, # degree
        "step_up": 0.3, # meters
        "step_down": 0.22, # meters
        "max_speed": 15.6, # m/s
    }
    path = {
        "max_distance_from_lander": 2000.0, # m
        "max_time": 28800.0, # s (8h)
        "walkback_radius_30m": 1800.0, # m for 30 min return
        "walkback_radius_1h": 3500.0, # m for 1h
    }
    parameters = {
        "speed": 9.7, #m/s
        "cruising_speed": 7.5, #m/s
        "suit_cost": 0.00023, # (L/s) * (1/kg) of VO₂ = +14 ml/min/kg of VO₂
        "metabolic_rate": 352.0 # W
    }


@dataclass
class RoverProfile(Profile):
    """
    HARD
    - Tilt / slope: Designed for 45° in any direction; drivers avoid terrain tilting the rover more than 30° [1]
    - Step / obstacle: 0.40 m (conservative); maximum 0.525 m
    - Dunes and ripples: Opportunity embedded in a ~0.3 m ripple (Purgatory), more than five weeks to get out [3] (NO-GO)
    - Soft ground: TI < 100 J m⁻² K⁻¹ s⁻½ → NO-GO (needs numerical TI, Section 2) -> Landing-site load-bearing constraint TI > 100–140 [4],
        used here as a proxy (Team assumption)
    SOFT
    - Sharp rocks: Cost ∝ P(rock > 0.40 m) from M4
    PARAMETER
    - Speed: 4.2 cm/s ≈ 150 m/h
    """
    hard_constraints ={
        "tilt": 45.0, # in degrees
        "slope": 10.0, # in degrees
        "step": 0.4, # meters
        "dunes": 0.3, # meters
        "soft_ground": 100.0, # TI < 100 J m⁻² K⁻¹ s⁻½
    }
    soft_constraints ={
        "sharp_rocks": 0.4 # if a sharp rock is greater than 0.4 m
    }
    parameters = {
        "speed": 0.042 # m/s
    }


class ExpeditionProfile(Profile):

    def __init__(self):
        self.profiles = []

    def add_participant(self, participant: Profile):
        self.profiles.append(participant)

