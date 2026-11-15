"""Terrain models on a DEM grid: slope, ruggedness and step heights.

Pure functions on ``np.ndarray`` with the grid spacing ``(dx, dy)`` in metres.
NaN means nodata and is propagated: any cell or edge touching a NaN is invalid.

Direction convention (shared by every module in ``src.models``)
----------------------------------------------------------------
``OFFSETS[k] = (drow, dcol)``, names N, NE, E, SE, S, SO, O, NO. Row 0 is north.
Per-edge arrays have shape ``(8, rows, cols)``: ``a[k, r, c]`` is the value of
the edge ``(r, c) -> (r + drow_k, c + dcol_k)``. Edges leaving the grid are
NaN (or inf for costs).
"""

import json

import numpy as np

# OFFSETS and DIRECTION_NAMES live in src/constants.py and are re-exported from here.
from src.constants import DIRECTION_NAMES, MARS_RADIUS_M, OFFSETS  # noqa: F401


def horizontal_lengths(dx: float, dy: float) -> np.ndarray:
    """Horizontal length (m) of each of the 8 edges: dy for N/S, dx for E/O,
    sqrt(dx^2 + dy^2) for diagonals."""
    lengths = []
    for drow, dcol in OFFSETS:
        if drow and dcol:
            lengths.append(float(np.hypot(dx, dy)))
        elif drow:
            lengths.append(float(dy))
        else:
            lengths.append(float(dx))
    return np.array(lengths)


def neighbor_stack(a: np.ndarray) -> np.ndarray:
    """Return ``out`` of shape (8, rows, cols) with
    ``out[k, r, c] = a[r + drow_k, c + dcol_k]``, NaN outside the grid."""
    a = np.asarray(a, dtype=np.float64)
    rows, cols = a.shape
    padded = np.pad(a, 1, mode="constant", constant_values=np.nan)
    out = np.empty((8, rows, cols))
    for k, (drow, dcol) in enumerate(OFFSETS):
        out[k] = padded[1 + drow:1 + drow + rows, 1 + dcol:1 + dcol + cols]
    return out


def horn_slope(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Slope in degrees with Horn's 3x3 finite-difference gradient [R29].

    Window ``a b c / d e f / g h i`` (row 0 north):
    dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 dx)
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 dy)
    Border cells are NaN; NaN propagates (also from the centre cell itself).
    """
    z = np.asarray(z, dtype=np.float64)
    nb = neighbor_stack(z)
    # k: 0=N(b) 1=NE(c) 2=E(f) 3=SE(i) 4=S(h) 5=SO(g) 6=O(d) 7=NO(a)
    b, c, f, i, h, g, d, a = nb
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[np.isnan(z)] = np.nan
    return slope


def slope_mask(slope_deg: np.ndarray, max_deg: float) -> np.ndarray:
    """True = forbidden (slope above ``max_deg`` or NaN)."""
    return ~(np.asarray(slope_deg) <= max_deg)


def tri(z: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index [R30]: sqrt(sum over the 8 neighbours of
    (z_neighbour - z_centre)^2), in metres. Border cells and NaN -> NaN."""
    z = np.asarray(z, dtype=np.float64)
    return np.sqrt(np.sum((neighbor_stack(z) - z) ** 2, axis=0))


def smooth_mean(z: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware k x k moving average (k odd).

    NaN values are ignored; near borders and nodata holes the window is
    partial (this biases the mean on ramps there). Cells whose window has no
    valid value are NaN.
    """
    total, count = _window_sum_count(z, k)
    out = np.full(count.shape, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _window_sum_count(z: np.ndarray, k: int, pad_mode: str = "nan") -> tuple:
    """Sum of the finite values and number of finite values in the k x k
    window centred on every cell (k odd).

    pad_mode "nan": outside the grid counts as NaN (missing).
    pad_mode "odd": the grid is extended by odd reflection,
    ``np.pad(z, k // 2, mode="reflect", reflect_type="odd")``, i.e.
    2 * z_border - z_mirror, which continues a linear ramp exactly; NaN cells
    reflect to NaN.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer.")
    if pad_mode not in ("nan", "odd"):
        raise ValueError("pad_mode must be 'nan' or 'odd'.")
    z = np.asarray(z, dtype=np.float64)
    half = k // 2
    if pad_mode == "nan":
        padded = np.pad(z, half, mode="constant", constant_values=np.nan)
    else:
        padded = np.pad(z, half, mode="reflect", reflect_type="odd")
    # Separable box sum with cumulative sums: O(cells) instead of the k*k-times larger
    # sliding_window_view temporary. Values are shifted by a reference elevation
    # before summing, so the cumulative sums stay small (rounding ~1e-12 m).
    valid = np.isfinite(padded)
    ref = float(np.median(padded[valid])) if valid.any() else 0.0
    total = _box_sum(np.where(valid, padded - ref, 0.0), k)
    count = _box_sum(valid.astype(np.int64), k)
    return total + count * ref, count


def _box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over every k x k window of ``a`` ('valid' windows: shape - (k - 1))."""
    for axis in (0, 1):
        c = np.cumsum(a, axis=axis)
        zero = np.zeros_like(np.take(c, [0], axis=axis))
        c = np.concatenate([zero, c], axis=axis)
        n = c.shape[axis]
        a = np.take(c, np.arange(k, n), axis=axis) - np.take(c, np.arange(0, n - k), axis=axis)
    return a


def step_reference_mean(z: np.ndarray, k: int) -> np.ndarray:
    """k x k reference mean used by ``edge_steps`` (``smooth_mean`` is unchanged).

    - Near the grid border the window is completed by odd reflection
      (see ``_window_sum_count``), so a linear ramp gives exactly the ramp
      value and creates no fake step, while a real step at the border stays
      visible.
    - Windows containing NaN (interior nodata holes, or NaN border cells whose
      reflection is NaN) fall back to the mean of their finite values only, as
      ``smooth_mean`` does. TEAM ASSUMPTION: this can create some fake steps
      around holes; it is the prudent choice (a fake step blocks, never opens).
    NaN where the window has no finite value.
    """
    total, count = _window_sum_count(z, k, pad_mode="odd")
    full = count == k * k
    out = smooth_mean(z, k)
    np.divide(total, count, out=out, where=full)
    return out


def edge_steps(z: np.ndarray, k: int = 5, compensate: bool = True) -> np.ndarray:
    """Signed step height (m) of every edge, shape (8, rows, cols).

    step(i -> j) = (z_j - z_i) - (zs_j - zs_i), with zs = step_reference_mean(z, k):
    the part of the elevation change not explained by the smooth slope.
    Positive = step up, negative = step down. NaN if an endpoint is NaN or the
    edge leaves the grid.

    Compensation (``compensate=True``, default): the k x k mean also follows
    part of a sharp step. For a step of height h aligned with a grid axis, zs
    jumps by h / k across it, so the raw value is h * (1 - 1/k) (0.8 h for
    k = 5). The raw value is divided by (1 - 1/k) so that such a step is
    measured at its true height. The factor is exact for axis-aligned steps,
    both on the straight edges and on the diagonal edges that cross them
    (zs changes only across the step, so the diagonal sees the same h / k).
    TEAM ASSUMPTION: for oblique steps (e.g. at 45 degrees) the mean takes a
    different fraction of the jump, so the compensated value is only
    approximate. ``compensate=False`` returns the raw value (previous
    behaviour). NaN stays NaN either way. With k = 1 the raw value is always
    0 and no compensation is applied.

    Reference mean (TEAM ASSUMPTION, see ``step_reference_mean``): at the
    border the k x k window is completed by odd reflection (exact on linear
    ramps, so no fake border steps and real border steps are still found);
    windows touching NaN use the mean of their finite values, which may create
    some fake steps around nodata holes.
    """
    z = np.asarray(z, dtype=np.float64)
    zs = step_reference_mean(z, k)
    zs = np.where(np.isnan(z), np.nan, zs)
    steps = (neighbor_stack(z) - z) - (neighbor_stack(zs) - zs)
    if compensate and k > 1:
        steps /= 1.0 - 1.0 / k
    return steps


def step_masks(steps: np.ndarray, max_up: float, max_down: float) -> np.ndarray:
    """True = forbidden edge: step > max_up, step < -max_down, or NaN."""
    steps = np.asarray(steps)
    return ~((steps <= max_up) & (steps >= -max_down))


def from_map(m) -> tuple:
    """Extract ``(z, dx, dy)`` in metres from a ``src.data.map.Map``.

    A Map grid is always geographic: ``granularity = (dlon, dlat)`` in degrees
    and corners are cell centres ``(lon, lat)``. The spacing is converted with
    a local spherical approximation at the central latitude (TEAM ASSUMPTION):
    dy = R * dlat, dx = R * dlon * cos(lat_centre), with R = MARS_RADIUS_M.

    Raises ValueError if the metadata JSON declares ``coordinate_units`` other
    than ``"degrees"``. Non-JSON metadata or a missing key means degrees.
    """
    try:
        meta = json.loads(m.metadata) if m.metadata else {}
    except (TypeError, ValueError):
        meta = {}
    units = meta.get("coordinate_units") if isinstance(meta, dict) else None
    if units is not None and units != "degrees":
        raise ValueError(
            f"from_map expects a Map with coordinate_units 'degrees', got {units!r}."
        )

    z = np.asarray(m.values, dtype=np.float64)
    dlon, dlat = m.granularity
    lat_centre = (m.top_left[1] + m.bottom_right[1]) / 2.0
    dy_m = MARS_RADIUS_M * np.radians(dlat)
    dx_m = MARS_RADIUS_M * np.radians(dlon) * np.cos(np.radians(lat_centre))
    return z, float(dx_m), float(dy_m)
