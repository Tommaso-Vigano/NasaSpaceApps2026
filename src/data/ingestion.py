
"""Load configured datasets using the registered adapters."""

import json
from pathlib import Path

from src.data.adapters.registry import AdapterRegistry
from src.data.map import Map


def load_datasets(config_path: str | Path) -> list[Map]:
    """Load each dataset once, resolving relative paths beside the config file.

    Entries require name, loader_name and path. Import built-in loaders explicitly
    so their decorators run even when no visualization/test module imported them.
    Errors abort startup rather than silently omitting a configured dataset.
    """

    # Ensure they are loaded property for registration!
    # DO NOT REMOVE EVEN IF GRAY!!!
    from src.data.adapters.implementations import (
        dust_adapter, map_adapter, temperature_adapter, thermal_inertia_adapter,
    )

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), list):
        raise ValueError(f"{config_path}: expected an object with a datasets list.")

    datasets = []
    for index, entry in enumerate(config["datasets"]):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in ("name", "loader_name", "path")
        ):
            raise ValueError(
                f"{config_path}: dataset {index} requires nonempty name, loader_name and path strings."
            )
        path = Path(entry["path"])
        if not path.is_absolute():
            path = config_path.parent / path
        try:
            loader = AdapterRegistry.get_adapter(entry["loader_name"])()
            map_ = loader.adapt(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Failed to load dataset {entry['name']!r} ({path}): {error}") from error
        map_.title = entry["name"]
        datasets.append(map_)
    return datasets
