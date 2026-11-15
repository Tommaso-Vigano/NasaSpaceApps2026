"""Inspect the NOAH-H terrain classes of Jezero through TerrainClassAdapter.

Reads a +-2 km box around the sol-13 lander position at 5 m and prints the
eta statistics. Requires rasterio and pyproj. Run from the repository root:
    python scripts/inspect_terrain.py [path/to/Jezero_NOAHH_mosaic_IG.tif] [--data-dir DIR]

Without an explicit raster path, the mosaic is looked up in the Jezero data folder:
--data-dir, else JEZERO_DATA_DIR (environment or .env), else ../jezero_data
(scripts/_data_dir.py).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio  # noqa: E402

from src.data.adapters.implementations.terrain_class_adapter import (  # noqa: E402
    TerrainClassAdapter,
)
from src.constants import (  # noqa: E402
    INSPECT_HALF_SIDE_M,
    INSPECT_RESOLUTION_M,
    LANDER_SOL13_LONLAT,
    MARS_RADIUS_M,
    NOAH_REL_PATH,
)

from _data_dir import resolve_data_dir  # noqa: E402  (scripts/_data_dir.py)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster", nargs="?", type=Path, help="NOAH-H interpretative groups GeoTIFF")
    ap.add_argument("--data-dir", type=Path, default=None, help="Jezero data folder")
    args = ap.parse_args()
    path = args.raster if args.raster else resolve_data_dir(args.data_dir) / NOAH_REL_PATH

    with rasterio.open(path) as src:
        print(f"File            : {path.name}")
        print(f"Source CRS      : {src.crs.to_string() if src.crs else None}")
        print(f"Size (rows×cols): {src.height} × {src.width}")
        print(f"dtype / nodata  : {src.dtypes[0]} / {src.nodata}")

    lon, lat = LANDER_SOL13_LONLAT
    # +-2 km on the ground in both directions.
    dlat = math.degrees(INSPECT_HALF_SIDE_M / MARS_RADIUS_M)
    dlon = dlat / math.cos(math.radians(lat))
    adapter = TerrainClassAdapter(
        (lon - dlon, lat + dlat), (lon + dlon, lat - dlat), resolution_m=INSPECT_RESOLUTION_M,
    )
    t0 = time.perf_counter()
    m = adapter.adapt(path)
    elapsed = time.perf_counter() - t0

    (lon0, lat0), (lon1, lat1) = m.top_left, m.bottom_right
    print(f"Box             : ±{INSPECT_HALF_SIDE_M:g} m around {lat}N {lon}E, "
          f"{INSPECT_RESOLUTION_M:g} m cells ({elapsed:.1f} s)")
    print(f"Shape (rows×cols): {m.values.shape[0]} × {m.values.shape[1]}")
    print(f"Lon bounds      : {lon0:.5f} .. {lon1:.5f} °E (cell centres)")
    print(f"Lat bounds      : {lat0:.5f} .. {lat1:.5f} °N (cell centres)")
    print(f"Step (deg)      : dlon = {m.granularity[0]:.6g}, dlat = {m.granularity[1]:.6g}")
    print(f"class -> eta    : {json.loads(m.metadata)['class_to_eta']}")

    v = m.values
    valid = v[np.isfinite(v)]
    print(f"NaN cells       : {100 * (1 - valid.size / v.size):.2f}%")
    if valid.size:
        print(f"eta min/median/mean/max: {valid.min():.3f} / {np.median(valid):.3f} / "
              f"{valid.mean():.3f} / {valid.max():.3f}")

    row = int(round((lat0 - lat) / m.granularity[1]))
    col = int(round((lon - lon0) / m.granularity[0]))
    print(f"At lander       : pixel (row {row}, col {col}) -> eta {v[row, col]:.3f}")


if __name__ == "__main__":
    main()
