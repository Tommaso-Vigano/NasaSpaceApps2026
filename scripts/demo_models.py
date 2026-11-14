"""Demo of src.models on a synthetic DEM (no real data, no plots).

Run from the repository root:  python scripts/demo_models.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import (  # noqa: E402
    ETA_TERRAIN,
    Astronaut,
    calibrate_dM_suit,
    edge_costs,
    edge_steps,
    eva_reach_mask,
    outbound_time_map,
    return_time_map,
    sep_isochrone_mask,
)

ROWS = COLS = 200
DX = DY = 10.0  # m/px
HILL_RC, HILL_H, HILL_SIGMA = (60, 100), 150.0, 300.0  # m
LANDER_RC = (170, 100)  # centre-south


def main():
    r, c = np.mgrid[0:ROWS, 0:COLS]
    d2 = ((r - HILL_RC[0]) * DY) ** 2 + ((c - HILL_RC[1]) * DX) ** 2
    z = HILL_H * np.exp(-d2 / (2 * HILL_SIGMA ** 2))

    astronaut = Astronaut()
    dM = calibrate_dM_suit()
    t0 = time.perf_counter()
    steps = edge_steps(z)
    t_edge, _ = edge_costs(z, DX, DY, ETA_TERRAIN["regolith"], astronaut, dM, steps=steps)
    t_ret = return_time_map(t_edge, LANDER_RC)
    t_out = outbound_time_map(t_edge, LANDER_RC)
    elapsed = time.perf_counter() - t0

    cell_km2 = DX * DY / 1e6
    sep = sep_isochrone_mask(t_ret, astronaut)
    reach = eva_reach_mask(t_out, t_ret, astronaut)
    top = HILL_RC

    print(f"DEM {ROWS}x{COLS} at {DX:.0f} m/px = {ROWS * COLS * cell_km2:.2f} km^2; "
          f"hill {HILL_H:.0f} m (sigma {HILL_SIGMA:.0f} m) at {HILL_RC}, lander at {LANDER_RC}")
    print(f"suit overhead dM_suit = {dM:.1f} W; costs + 2 Dijkstra in {elapsed:.2f} s")
    print(f"30-min SEP isochrone area : {sep.sum() * cell_km2:.3f} km^2")
    print(f"6-h action radius area    : {reach.sum() * cell_km2:.3f} km^2"
          f" (cells reachable: {reach.sum()} / {reach.size})")
    print(f"return time from hill top : {t_ret[top] / 60:.1f} min")
    print(f"hill top outbound / return: {t_out[top] / 60:.1f} min / {t_ret[top] / 60:.1f} min "
          f"(outbound is {100 * (t_out[top] / t_ret[top] - 1):+.1f}% vs return)")
    print(f"hill top inside SEP isochrone: {bool(sep[top])}")


if __name__ == "__main__":
    main()
