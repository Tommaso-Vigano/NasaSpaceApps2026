"""Inspect the MCD diurnal CSVs through TemperatureAdapter.

Numpy and the standard library only. Run from the repository root:
    python scripts/inspect_temperature.py [path/to/MCD/folder] [--data-dir DIR]

Without a folder, it uses <data dir>/06_polvere_atmosfera/MCD, with the data dir chosen by
scripts/_data_dir.py (--data-dir, then JEZERO_DATA_DIR from the environment or .env, then
../jezero_data next to the repository).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import KELVIN_OFFSET, LANDER_SOL13_LONLAT, MCD_DIR_REL_PATH  # noqa: E402
from src.data.adapters.implementations.temperature_adapter import (  # noqa: E402
    TemperatureAdapter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_dir import resolve_data_dir  # noqa: E402  (scripts/_data_dir.py)


def temperature(path: Path, hour: float, field: str) -> float:
    # A single cell is enough: the map is spatially uniform.
    m = TemperatureAdapter(
        local_time_h=hour, field=field,
        top_left=LANDER_SOL13_LONLAT, bottom_right=LANDER_SOL13_LONLAT,
    ).adapt(path)
    return float(m.values[0, 0])


def fmt(k: float) -> str:
    return f"{k:6.1f} K ({k - KELVIN_OFFSET:6.1f} °C)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path, help="MCD folder (diurnal_*.csv)")
    parser.add_argument("--data-dir", type=Path, help="jezero_data folder")
    args = parser.parse_args()
    folder = args.folder if args.folder else resolve_data_dir(args.data_dir) / MCD_DIR_REL_PATH
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
