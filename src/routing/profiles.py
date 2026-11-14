from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta


class Profile(metaclass=ABCMeta):
    """ An interface for the profile for the routing algorithm """
    pass


class HumanProfile(Profile):
    pass

class RoverProfile(Profile):
    pass


