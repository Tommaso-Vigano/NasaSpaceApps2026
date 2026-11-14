"""Inspect the MCD diurnal CSVs through TemperatureAdapter.

Numpy and the standard library only. Run from the repository root:
    python scripts/inspect_temperature.py [path/to/MCD/folder]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.adapters.implementations.temperature_adapter import (  # noqa: E402
    TemperatureAdapter,
)

DEFAULT_DIR = (
    Path.home() / "Documents" / "Michele documents" / "Family and Other"
    / "NASA space apps 2026 - Milan" / "jezero_data" / "06_polvere_atmosfera" / "MCD"
)
KELVIN_OFFSET = 273.15


def temperature(path: Path, hour: float, field: str) -> float:
    # A single cell is enough: the map is spatially uniform.
    m = TemperatureAdapter(
        local_time_h=hour, field=field,
        top_left=(77.45, 18.44), bottom_right=(77.45, 18.44),
    ).adapt(path)
    return float(m.values[0, 0])


def fmt(k: float) -> str:
    return f"{k:6.1f} K ({k - KELVIN_OFFSET:6.1f} °C)"


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(folder.glob("diurnal_*.csv"))
    if not files:
        sys.exit(f"No diurnal_*.csv in {folder}")

    for path in files:
        scenario = json.loads(TemperatureAdapter().adapt(path).metadata)["scenario"]
        print(f"== {scenario}")
        for field in ("air", "surface"):
            t13 = temperature(path, 13.0, field)
            t05 = temperature(path, 5.0, field)
            day = [temperature(path, h, field) for h in np.arange(0.0, 24.0, 0.25)]
            print(f"  {field:<7} 13h {fmt(t13)} | 5h {fmt(t05)} | "
                  f"min {fmt(min(day))} | max {fmt(max(day))}")


if __name__ == "__main__":
    main()
