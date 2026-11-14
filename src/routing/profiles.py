
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


@dataclass
class HumanProfile(Profile):

    tolerable_temp: float


@dataclass
class RoverProfile(Profile):

    max_height_drop: float


class ExpeditionProfile(Profile):

    def __init__(self):
        self.profiles = []

    def add_participant(self, participant: Profile):
        self.profiles.append(participant)

