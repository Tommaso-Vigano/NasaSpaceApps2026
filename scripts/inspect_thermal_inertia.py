"""Inspect the THEMIS thermal inertia GeoTIFF of Jezero through ThermalInertiaAdapter.

Requires rasterio and pyproj. Run from the repository root:
    python scripts/inspect_thermal_inertia.py [path/to/Jezero_Thermal_Inertia_THEMIS_v1.tiff] [--data-dir DIR]

Without a path, it uses <data dir>/05_termico_suolo/numerici/..., with the data dir chosen by
scripts/_data_dir.py (--data-dir, then JEZERO_DATA_DIR from the environment or .env, then
../jezero_data next to the repository).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio  # noqa: E402

from src.data.adapters.implementations.thermal_inertia_adapter import (  # noqa: E402
    ThermalInertiaAdapter,
)
from src.constants import LANDER_SOL13_LONLAT, THEMIS_REL_PATH  # noqa: E402
from src.models.terrain import from_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_dir import resolve_data_dir  # noqa: E402  (scripts/_data_dir.py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raster", nargs="?", type=Path, help="THEMIS thermal inertia GeoTIFF")
    parser.add_argument("--data-dir", type=Path, help="jezero_data folder")
    args = parser.parse_args()
    path = args.raster if args.raster else resolve_data_dir(args.data_dir) / THEMIS_REL_PATH

    with rasterio.open(path) as src:
        print(f"File            : {path.name}")
        print(f"Source CRS      : {src.crs.to_string() if src.crs else None}")
        print(f"Size (rows×cols): {src.height} × {src.width}")
        print(f"dtype / nodata  : {src.dtypes[0]} / {src.nodata}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = ThermalInertiaAdapter().adapt(path)
    for w in caught:
        print(f"Warning         : {w.message}")

    (lon0, lat0), (lon1, lat1) = m.top_left, m.bottom_right
    dlon, dlat = m.granularity
    _, dx_m, dy_m = from_map(m)
    print(f"Lon bounds      : {lon0:.5f} .. {lon1:.5f} °E (cell centres)")
    print(f"Lat bounds      : {lat0:.5f} .. {lat1:.5f} °N (cell centres)")
    print(f"Step (deg)      : dlon = {dlon:.6g}, dlat = {dlat:.6g}")
    print(f"Step (m)        : dx = {dx_m:.1f}, dy = {dy_m:.1f}")

    v = m.values
    valid = v[np.isfinite(v)]
    total = v.size
    print(f"Valid pixels    : {valid.size} / {total} ({100 * valid.size / total:.1f}%)")
    if valid.size:
        print(f"min/median/max  : {valid.min():.1f} / {np.median(valid):.1f} / "
              f"{valid.max():.1f} J m^-2 K^-1 s^-1/2")
        print(f"% < 100         : {100 * np.mean(valid < 100):.2f}% of valid")
        in_band = (valid >= 80) & (valid <= 120)
        print(f"% in [80, 120]  : {100 * np.mean(in_band):.2f}% of valid")

    lon, lat = LANDER_SOL13_LONLAT  # Perseverance landing site (sol 13 waypoint)
    col = int(round((lon - lon0) / dlon))
    row = int(round((lat0 - lat) / dlat))
    if 0 <= row < v.shape[0] and 0 <= col < v.shape[1]:
        print(f"At {lat}N {lon}E: pixel (row {row}, col {col}) "
              f"centre ({lon0 + col * dlon:.5f}E, {lat0 - row * dlat:.5f}N) "
              f"-> {v[row, col]:.1f} J m^-2 K^-1 s^-1/2")
    else:
        print(f"At {lat}N {lon}E: outside the raster")


if __name__ == "__main__":
    main()
