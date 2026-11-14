
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

    @staticmethod
    def get_adapter(name: str):
        """Return the adapter class registered under name."""
        try:
            return AdapterRegistry._REGISTERED_ADAPTERS[name]
        except KeyError:
            raise ValueError(
                f"Unknown loader {name!r}. Available loaders: "
                f"{', '.join(AdapterRegistry.get_available())}"
            ) from None

