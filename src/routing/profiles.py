from dataclasses import dataclass
from abc import ABCMeta


@dataclass()
class Profile(metaclass=ABCMeta):
    """ An interface for the profile for the routing algorithm """
    pass


@dataclass
class HumanProfile(Profile):
    pass


@dataclass
class RoverProfile(Profile):
    pass


class ExpeditionProfile(Profile):

    def __init__(self):
        self.profiles = []

