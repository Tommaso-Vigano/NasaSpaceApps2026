"""Line-of-sight visibility (radio/visual contact with the lander) on a DEM."""

import numpy as np

from src.models.constraints import MARS_RADIUS_M


def horizon_distance_m(h):
    """Distance (m) to the geometric horizon for an eye/antenna at height h
    (m) above a smooth Mars sphere: sqrt(2 R h)."""
    return np.sqrt(2.0 * MARS_RADIUS_M * np.asarray(h, dtype=np.float64))


def viewshed(z, dx, dy, observer_rc, observer_h=2.0, target_h=1.8,
             max_range_m=5000.0, n_rays=720):
    """Visibility mask (rows, cols) from ``observer_rc`` with radial rays.

    Each ray is sampled every min(dx, dy) metres (nearest cell). Elevations
    are lowered by the curvature drop d^2 / (2 R). A cell is visible if the
    elevation angle of a target ``target_h`` above it is >= the maximum
    terrain angle met before it on the same ray. NaN or leaving the grid
    blocks the rest of the ray. A cell is visible if any ray sees it; the
    observer cell is visible.

    TEAM ASSUMPTIONS: observer_h = 2 m (lander antenna), target_h = 1.8 m
    (astronaut helmet). Far from the observer the rays are 2*pi*r/n_rays apart:
    increase n_rays if that exceeds the cell size.
    """
    z = np.asarray(z, dtype=np.float64)
    rows, cols = z.shape
    r0, c0 = (int(v) for v in observer_rc)
    if not (0 <= r0 < rows and 0 <= c0 < cols) or not np.isfinite(z[r0, c0]):
        raise ValueError("Observer must be inside the grid on a valid cell.")
    z_obs = z[r0, c0] + observer_h

    step = min(dx, dy)
    n_samples = max(int(max_range_m // step), 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)[:, None]
    d = step * np.arange(1, n_samples + 1)[None, :]
    rr = np.rint(r0 - d * np.sin(theta) / dy).astype(np.int64)  # row 0 = north
    cc = np.rint(c0 + d * np.cos(theta) / dx).astype(np.int64)

    inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    zz = z[np.clip(rr, 0, rows - 1), np.clip(cc, 0, cols - 1)]
    blocked = np.logical_or.accumulate(~inside | np.isnan(zz), axis=1)

    dist = np.hypot((cc - c0) * dx, (rr - r0) * dy)
    safe = np.where(dist > 0, dist, 1.0)
    z_eff = zz - dist ** 2 / (2.0 * MARS_RADIUS_M)
    a_terrain = (z_eff - z_obs) / safe
    a_target = (z_eff + target_h - z_obs) / safe

    a_terrain = np.where(blocked, np.inf, a_terrain)
    a_terrain = np.where(dist == 0, -np.inf, a_terrain)
    running = np.maximum.accumulate(a_terrain, axis=1)
    previous = np.concatenate([np.full((n_rays, 1), -np.inf), running[:, :-1]], axis=1)

    seen = (a_target >= previous) & ~blocked & (dist > 0) & (dist <= max_range_m)
    out = np.zeros((rows, cols), dtype=bool)
    out[rr[seen], cc[seen]] = True
    out[r0, c0] = True
    return out
