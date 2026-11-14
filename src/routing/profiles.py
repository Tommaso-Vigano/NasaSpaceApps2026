from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta


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


