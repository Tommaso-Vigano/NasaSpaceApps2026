
from src.basic import Coordinate
from src.aliases import percentage
from src.routing.profiles import Profile

class RoutingExecutor:

    def __init__(self,
        top_left: Coordinate,
        bot_right: Coordinate,
        profile: Profile,
        fineness: percentage
    ):
        pass

    @staticmethod
    def load_profile_from_json(json: dict) -> Profile:
        pass


class RoutingResult:
    pass
