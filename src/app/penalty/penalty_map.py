from src.aliases import percentage


class PenaltyMap:

    def __init__(self,
        granularity: percentage
    ):
        """

        :param granularity:
        """
        self.granularity = granularity

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
