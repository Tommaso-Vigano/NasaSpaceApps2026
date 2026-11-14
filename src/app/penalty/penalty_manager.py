from abc import ABC

from src.app.penalty.penalty_map import PenaltyMap
from src.routing.profiles import Profile, ExpeditionProfile
from src.aliases import percentage

class PenaltyManager(ABC):
    """ The interface for penalty managers which are entrusted with constructing the cost
    map the routing algorithm needs to use for his path.

    The main purpose of this object is using all the data adapters and, froom the reference
    user profiles, constructing a penalty mmap with a speciefi
    """

    @staticmethod
    def build_from(profile: Profile, granularity: percentage) -> PenaltyMap:
        """ Build a PenaltyMap from the given profile and granularity, constructing first
        the required penalty implementation additive components as individual pipelines. """

        if isinstance(profile, ExpeditionProfile):
            maps = []
            result = PenaltyMap.reduce(maps)
        else:
            result = None
        return result
