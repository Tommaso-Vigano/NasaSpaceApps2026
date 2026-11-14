from dataclasses import dataclass
from src.basic import Coordinate


@dataclass
class Landmark:
    """ The object representing a landmark, a salient point in space associated with
    a label or a certain action. """
    # Reference point at which landmark is positioned
    coo: Coordinate
