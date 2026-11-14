from abc import ABC
from dataclasses import fields

from src.app.penalty.penalty_map import PenaltyMap
from src.routing.profiles import Profile, ExpeditionProfile
from src.aliases import percentage

class PenaltySuccession:

    def __init__(self, addends: list):
        self.addends = addends

    def apply(self, raw_numpy_map) -> None:
        """ Apply all addends in the set of additive penalties """
        for addend in self.addends:
            addend(raw_numpy_map)

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

    @staticmethod
    def _construct_cost_pipeline(profile: Profile) -> PenaltySuccession:
        """ Construct a cost-based pipeline from the given profile. The profile
         CANNOT be an aggregate profile """
        if isinstance(profile, ExpeditionProfile):
            raise TypeError("The profile must not be an aggregate when constructing an individual cost "
                            "pipeline.")
        # Here construct a PenaltySuccession from the various fields in the profiles
        for field in fields(profile):
            pass

        return None