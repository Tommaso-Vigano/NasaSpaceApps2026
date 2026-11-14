
class AdapterRegistry:

    _REGISTERED_ADAPTERS = {}

    @staticmethod
    def register_adapter(name: str):
        """Register an adapter class under the given name."""
        def decorator(adapter_class):
            AdapterRegistry._REGISTERED_ADAPTERS[name] = adapter_class
            return adapter_class

        return decorator

    @staticmethod
    def get_available() -> list[str]:
        return list(AdapterRegistry._REGISTERED_ADAPTERS)

