import csv

import numpy as np
import pytest

from src.models.gates import (
    daylight_window,
    dust_storm_nogo,
    eva_go,
    load_mcd_diurnal,
    solar_power_ratio,
)
from src.models.viewshed import horizon_distance_m, viewshed

HEADER = [
    "Local time (Martian hour)",
    "Temperature (K)",
    "Incident solar flux on horizontal surface (W/m2)",
    "Daily mean dust column visible optical depth above surface",
    "Horizontal wind speed (m/s)",
    "W-E wind component (m/s)",
    "S-N wind component (m/s)",
]


def _flux(h, peak=500.0):
    if h <= 6 or h >= 18:
        return 0.0
    return peak * (1 - abs(h - 12) / 6)


def _write_mcd(path, tau=0.4, peak=500.0):
    with open(path, "w", newline="") as f:
        f.write("# synthetic Mars Climate Database profile\n")
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(HEADER)
        for h in range(24):
            w.writerow([float(h), 200.0 + 2 * h, _flux(h, peak), tau, 5.0, 1.0, -1.0])
    return path


def test_load_mcd_keeps_known_columns(tmp_path):
    mcd = load_mcd_diurnal(_write_mcd(tmp_path / "mcd.csv"))
    assert set(mcd) == set(HEADER[:5])
    assert mcd["Local time (Martian hour)"].shape == (24,)


def test_daylight_window(tmp_path):
    mcd = load_mcd_diurnal(_write_mcd(tmp_path / "mcd.csv"))
    start, end = daylight_window(mcd, 100.0)
    assert start == pytest.approx(7.2, abs=0.1)
    assert end == pytest.approx(16.8, abs=0.1)
    assert all(np.isnan(daylight_window(mcd, 1000.0)))


def test_dust_storm_and_go(tmp_path):
    storm = load_mcd_diurnal(_write_mcd(tmp_path / "storm.csv", tau=5.0, peak=150.0))
    clear = load_mcd_diurnal(_write_mcd(tmp_path / "clear.csv", tau=0.4))
    assert dust_storm_nogo(storm) is True
    assert dust_storm_nogo(clear) is False
    assert solar_power_ratio(storm, clear) == pytest.approx(0.3)

    go, reasons = eva_go(clear, 8.0, 6.0)
    assert go and reasons == []
    go, reasons = eva_go(clear, 6.0, 6.0)
    assert not go and "daylight" in reasons[0]
    go, reasons = eva_go(storm, 8.0, 6.0)
    assert not go and any("dust storm" in r for r in reasons)


def test_horizon_distance():
    assert horizon_distance_m(1.8) == pytest.approx(3497, abs=5)


def test_viewshed_flat_and_wall():
    dx = 10.0
    z = np.zeros((101, 101))
    obs = (50, 50)
    vis = viewshed(z, dx, dx, obs, max_range_m=400.0)
    r, c = np.mgrid[0:101, 0:101]
    dist = np.hypot(r - 50, c - 50) * dx
    assert np.all(vis[dist <= 390.0])
    assert not np.any(vis[dist > 420.0])

    z_wall = z.copy()
    z_wall[40:61, 60] = 20.0  # 20 m wall, 100 m east of the observer
    vis = viewshed(z_wall, dx, dx, obs, max_range_m=400.0)
    assert vis[50, 60]  # the wall itself is seen
    assert not np.any(vis[50, 61:66])
    assert vis[50, 40]  # the other side is unaffected


def test_viewshed_nan_blocks_ray():
    z = np.zeros((41, 41))
    z[:, 25] = np.nan
    vis = viewshed(z, 10.0, 10.0, (20, 20), max_range_m=300.0)
    assert vis[20, 24] and not vis[20, 25] and not vis[20, 30]
