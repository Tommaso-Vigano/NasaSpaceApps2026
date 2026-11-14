from src.aliases import percentage


class PenaltyMap:

    def __init__(self,
        granularity: percentage
    ):
        """

        :param granularity:
        """
        self.granularity = granularity
        # Lazy initialize of the numpy array
        self.cost_map = None

    def get_obstructed_region(self):
        pass

    def get_constraints(self) -> list:
        pass

    @staticmethod
    def reduce(multiple_maps: list["PenaltyMap"]) -> "PenaltyMap":
        """ Reduce multiple penality maps into a single unified map """
        if len(set(mp.granularity for mp in multiple_maps)) > 1:
            raise ValueError("Multiple penality maps cannot be reduced if they have inhomogeneous "
                             "granularity. ")
        pass


    def to_pf_form(self) -> None:
        """ Converts the internal representation of the cost map to the one
        required for the path-finding library """
        self.cost_map += 1.0
        # +infinity should remain +infinity, or very, very large
        # pathfinding wants to use a cost map which is 0 to represent
        # unavailability of the region
        self.cost_map[self.cost_map > 1e9] = 0.0
