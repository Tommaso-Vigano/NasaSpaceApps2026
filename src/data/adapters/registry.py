
class AdapterRegistry:

    _REGISTERED_ADAPTERS = {}

    @staticmethod
    def get_available() -> list[str]:
        return []

