from abc import ABC, abstractmethod


class Adapter(ABC):
    """
    General interface of the adapter, which has the role of adapting the different
    data formats to the data structure built in Python
    """

    @abstractmethod
    def adapt(self, data):
        """
        Convert input data into the appropriate internal representation.

        Each concrete adapter defines its supported input format
        and the data structure it returns.
        """
        pass
