from abc import ABC, abstractmethod
from src.app.penalty.penalty_map import PenaltyMap
from src.routing.profiles import Profile


class PenaltyManager(ABC):
    """ The interface for penalty managers which are entrusted with constructing the cost
    map the routing algorithm needs to use for his path.

    The main purpose of this object is using all the data adapters and, froom the reference
    user profiles, constructing a penalty mmap with a speciefi
    """

    @staticmethod
    def build_from(self, profile: Profile) -> PenaltyMap:
        pass

