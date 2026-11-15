"""Demo of src.costs on a synthetic DEM (no real data, no plots).

Walker (Astronaut) and rover (RoboticRover) on the same terrain, with a
synthetic MCD profile that is GO and one that is NO-GO (dust storm).
The grid size below belongs to this demo only; src.costs takes it from z.

Run from the repository root:  python scripts/demo_costs.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.costs.combine import build_costs, path_edge_sum, plan_route, summary  # noqa: E402
from src.costs.context import CostContext, CostWeights, fill_eta  # noqa: E402
from src.models.constraints import ETA_TERRAIN, MARS_RADIUS_M, Astronaut, RoboticRover  # noqa: E402
from src.models.terrain import OFFSETS, horizontal_lengths  # noqa: E402

ROWS, COLS = 90, 120
DX = DY = 10.0                                         # m/px
STEP_DEG = DX / (MARS_RADIUS_M * math.pi / 180.0)      # same cell in degrees
TOP_LEFT = (77.40, 18.45)                              # (lon, lat) of cell (0, 0)
LANDER_RC = (75, 60)
START_RC, END_RC = (70, 15), (20, 100)
EVA_START_H = 9.0
ROVER_WINDOW_H = 6.0   # demo only: a rover has no eva_nominal_h, give its window explicitly


def rc_to_lonlat(rc):
    return TOP_LEFT[0] + rc[1] * STEP_DEG, TOP_LEFT[1] - rc[0] * STEP_DEG


def synthetic_terrain():
    r, c = np.mgrid[0:ROWS, 0:COLS]
    hill = 60.0 * np.exp(-(((r - 35) * DY) ** 2 + ((c - 55) * DX) ** 2) / (2 * 150.0 ** 2))
    z = hill + 0.002 * (c * DX)                        # gentle regional tilt
    eta = np.full(z.shape, ETA_TERRAIN["regolith"])
    eta[10:40, 75:105] = ETA_TERRAIN["sand"]           # ripple field
    eta[:, :8] = np.nan                                # outside NOAH-H coverage
    ripples = np.zeros(z.shape, dtype=bool)
    ripples[15:35, 80:100] = True                      # large ripples inside the field
    ti = np.full(z.shape, 250.0)
    ti[50:60, 20:45] = 60.0                            # loose soil patch
    return z, eta, ripples, ti


def synthetic_mcd(tau):
    h = np.arange(0.0, 25.0)
    swing = 0.5 * (1 - np.cos(2 * np.pi * (h - 3.0) / 24.0))
    return {
        "Local time (Martian hour)": h,
        "Incident solar flux on horizontal surface (W/m2)":
            600.0 * np.clip(np.sin(np.pi * (h - 6.0) / 12.0), 0.0, None),
        "Dust column visible optical depth above surface": np.full(h.shape, tau),
        "Temperature (K)": 190.0 + 60.0 * swing,
        "Surface temperature (K)": 180.0 + 100.0 * swing,
        "Horizontal wind speed (m/s)": 2.0 + 6.0 * swing,
    }


def path_stats(path_rc, report):
    """Horizontal length (m) and time (s, time + eta terms) along a path."""
    lengths = horizontal_lengths(DX, DY)
    k_of = {off: k for k, off in enumerate(OFFSETS)}
    length = sum(lengths[k_of[(r1 - r0, c1 - c0)]]
                 for (r0, c0), (r1, c1) in zip(path_rc, path_rc[1:]))
    t = path_edge_sum(report.term("time").edge_cost + report.term("eta").edge_cost, path_rc)
    return length, t


def print_summary(s):
    print(f"  {'term':<16}{'kind':<13}{'inf edges':>10}{'finite sum':>14}  reason")
    for name, v in s["terms"].items():
        print(f"  {name:<16}{v['kind']:<13}{v['inf_edges']:>10}{v['finite_sum']:>14.0f}  "
              f"{v['reason']}")


def run(label, agent, mcd, weights, z, eta, ripples, ti):
    ctx = CostContext(z=z, dx=DX, dy=DY, eta_map=fill_eta(eta, weights.eta_default),
                      lander_rc=LANDER_RC, agent=agent, mcd=mcd, eva_start_h=EVA_START_H,
                      eva_duration_h=None if hasattr(agent, "eva_nominal_h") else ROVER_WINDOW_H,
                      k_rocks=0.05, large_ripple_mask=ripples, thermal_inertia=ti)
    t0 = time.perf_counter()
    report = build_costs(ctx, weights)
    route = plan_route(report, TOP_LEFT, STEP_DEG, rc_to_lonlat(START_RC), rc_to_lonlat(END_RC))
    dt = time.perf_counter() - t0

    print(f"\n=== {label} ===  ({dt:.2f} s)")
    print_summary(summary(report))
    print(f"  GO: {report.go}" + ("" if report.go else f"  reasons: {report.reasons}"))
    if route.found:
        length, t = path_stats(route.path_rc, report)
        print(f"  route {START_RC} -> {END_RC}: {len(route.path_rc)} cells, "
              f"{length:.0f} m, time {t / 60:.1f} min, total cost {route.model_cost:.0f} s-eq, "
              f"{route.removed_edges} forbidden arcs removed")
    else:
        print(f"  no route: {route.reasons}")


def main():
    z, eta, ripples, ti = synthetic_terrain()
    print(f"DEM {ROWS}x{COLS} at {DX:.0f} m/px (demo only), lander {LANDER_RC}, "
          f"start {START_RC}, end {END_RC}")
    w = CostWeights(visibility_penalty_weight=0.25, wind_max_ms=20.0)
    go, nogo = synthetic_mcd(tau=0.5), synthetic_mcd(tau=5.0)
    run("walker, MCD GO", Astronaut(), go, w, z, eta, ripples, ti)
    run("rover, MCD GO", RoboticRover(), go, w, z, eta, ripples, ti)
    run("walker, MCD NO-GO (tau 5)", Astronaut(), nogo, w, z, eta, ripples, ti)


if __name__ == "__main__":
    main()
