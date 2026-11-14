import numpy as np
import pytest

from src.models.terrain import (
    DIRECTION_NAMES,
    OFFSETS,
    edge_steps,
    horizontal_lengths,
    horn_slope,
    neighbor_stack,
    slope_mask,
    step_masks,
    tri,
)



def _k(name):
    """Index of a direction ("N", "NE", ..., "NO") in the per-edge arrays."""
    return DIRECTION_NAMES.index(name)


def _straight():
    return [_k(n) for n in ("N", "E", "S", "O")]


def _diagonal():
    return [_k(n) for n in ("NE", "SE", "SO", "NO")]


def _grid(rows, cols, dx=1.0, dy=1.0):
    r, c = np.mgrid[0:rows, 0:cols]
    return c * dx, r * dy


def test_neighbor_stack_convention():
    z = np.arange(12, dtype=float).reshape(3, 4)
    nb = neighbor_stack(z)
    assert nb.shape == (8, 3, 4)
    for k, (dr, dc) in enumerate(OFFSETS):
        assert nb[k, 1, 1] == z[1 + dr, 1 + dc]
    # edges leaving the grid are NaN
    assert np.isnan(nb[DIRECTION_NAMES.index("N"), 0, 2])
    assert np.isnan(nb[_k("E"), 1, 3])
    np.testing.assert_allclose(horizontal_lengths(2.0, 3.0),
                               [3, np.hypot(2, 3), 2, np.hypot(2, 3)] * 2)


def test_horn_slope_inclined_plane():
    x, _ = _grid(10, 10)
    slope = horn_slope(0.1 * x, 1.0, 1.0)
    expected = np.degrees(np.arctan(0.1))  # 5.71 deg
    np.testing.assert_allclose(slope[1:-1, 1:-1], expected, atol=0.01)
    assert np.all(np.isnan(slope[0, :])) and np.all(np.isnan(slope[-1, :]))
    assert np.all(np.isnan(slope[:, 0])) and np.all(np.isnan(slope[:, -1]))


def test_flat_plane_zero_everything():
    z = np.full((9, 9), 42.0)
    np.testing.assert_allclose(horn_slope(z, 1.0, 1.0)[1:-1, 1:-1], 0.0)
    np.testing.assert_allclose(tri(z)[1:-1, 1:-1], 0.0)
    steps = edge_steps(z)
    assert steps.shape == (8, 9, 9)
    np.testing.assert_allclose(steps[np.isfinite(steps)], 0.0, atol=1e-12)


def test_smooth_ramp_is_not_a_step():
    x, _ = _grid(15, 15, dx=2.0)
    z = np.tan(np.radians(10.0)) * x
    k = 5
    steps = edge_steps(z, k=k)
    h = k // 2
    # both endpoints must have a full smoothing window
    inner = steps[:, h + 1:-h - 1, h + 1:-h - 1]
    np.testing.assert_allclose(inner, 0.0, atol=1e-9)


def test_sharp_step_detected_and_masked():
    z = np.zeros((11, 11))
    z[:, 5:] = 0.5
    steps = edge_steps(z, k=5)
    assert abs(steps[_k("E"), 5, 4] - 0.5) <= 0.15  # step up going east
    assert abs(steps[_k("O"), 5, 5] + 0.5) <= 0.15  # step down going west
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    assert mask[_k("E"), 5, 4] and mask[_k("O"), 5, 5]
    # away from the step nothing is forbidden (except edges leaving the grid)
    assert not mask[_k("E"), 5, 1]
    assert mask[_k("E"), 5, 10]


def test_nan_propagates_to_slope_and_mask():
    x, _ = _grid(7, 7)
    z = 0.1 * x
    z[3, 3] = np.nan
    slope = horn_slope(z, 1.0, 1.0)
    assert np.all(np.isnan(slope[2:5, 2:5]))
    assert np.all(slope_mask(slope, 20.0)[2:5, 2:5])
    assert not slope_mask(slope, 20.0)[1, 1]
    assert np.isnan(tri(z)[2, 2])
    steps = edge_steps(z)
    assert np.isnan(steps[_k("E"), 3, 2]) and np.all(np.isnan(steps[:, 3, 3]))


# --- from_map ---------------------------------------------------------------

def _degree_map(metadata='{"coordinate_units": "degrees"}'):
    from src.data.map import Map
    from src.data.types import Datatype

    step = 1e-4
    rows, cols = 3, 4
    top_left = (77.45, 18.44 + step)
    bottom_right = (77.45 + (cols - 1) * step, 18.44 + step - (rows - 1) * step)
    return Map(
        values=np.arange(rows * cols, dtype=float).reshape(rows, cols),
        granularity=(step, step),
        top_left=top_left,
        bottom_right=bottom_right,
        metadata=metadata,
        value_type=Datatype.ELEVATION,
        title="Elevation Map test",
    )


def test_from_map_converts_degrees_to_metres():
    from src.models.terrain import from_map

    m = _degree_map()
    z, dx, dy = from_map(m)
    assert z.shape == (3, 4)
    assert abs(dy - 5.93) <= 0.01
    assert abs(dx - 5.62) <= 0.01


def test_from_map_rejects_non_degree_units():
    import pytest
    from src.models.terrain import from_map

    with pytest.raises(ValueError, match="degrees"):
        from_map(_degree_map('{"coordinate_units": "metres"}'))


# --- edge_steps: odd-reflection reference mean at the border ------------------



def _ramp(shape, deg_x=8.0, deg_y=5.0, dx=2.0):
    """Pure inclined plane with different slopes along x and y (no real step)."""
    x, y = _grid(*shape, dx=dx, dy=dx)
    return np.tan(np.radians(deg_x)) * x + np.tan(np.radians(deg_y)) * y


def _off_grid(shape):
    return np.isnan(neighbor_stack(np.zeros(shape)))


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
@pytest.mark.parametrize("deg_x, deg_y", [(8.0, 5.0), (-3.0, 12.0)])
def test_ramp_has_no_fake_steps_even_at_borders(shape, deg_x, deg_y):
    steps = edge_steps(_ramp(shape, deg_x, deg_y))
    off = _off_grid(shape)
    assert np.all(np.isnan(steps[off])) and np.all(np.isfinite(steps[~off]))
    np.testing.assert_allclose(steps[~off], 0.0, atol=1e-9)
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    np.testing.assert_array_equal(mask, off)   # only edges leaving the grid


def _step_east(shape):
    """Ramp + 0.5 m step between column cs - 1 and cs, on every row
    (so it reaches the north and south borders)."""
    cs = shape[1] // 2
    z = _ramp(shape)
    z[:, cs:] += 0.5
    return z, cs


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
def test_real_step_found_in_the_centre(shape):
    rows, cols = shape
    z, cs = _step_east(shape)
    k = 5
    steps = edge_steps(z, k=k)
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    r = rows // 2
    assert abs(steps[_k("E"), r, cs - 1] - 0.5) <= 0.15 and mask[_k("E"), r, cs - 1]
    assert abs(steps[_k("O"), r, cs] + 0.5) <= 0.15 and mask[_k("O"), r, cs]
    # nothing forbidden far from the step column (apart from leaving the grid)
    _, c = np.mgrid[0:rows, 0:cols]
    far = np.broadcast_to(np.abs(c - cs) > k, mask.shape)
    np.testing.assert_array_equal(mask[far], _off_grid(shape)[far])


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
def test_real_step_found_at_the_border(shape):
    rows, _ = shape
    z, cs = _step_east(shape)
    steps = edge_steps(z)
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    for r in (0, 1, rows - 2, rows - 1):        # first and last two rows
        assert abs(steps[_k("E"), r, cs - 1] - 0.5) <= 0.15 and mask[_k("E"), r, cs - 1], r
        assert abs(steps[_k("O"), r, cs] + 0.5) <= 0.15 and mask[_k("O"), r, cs], r


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
def test_nan_hole_stays_nan_and_does_not_spread(shape):
    rows, cols = shape
    z = _ramp(shape)
    hole = (slice(rows // 2 - 1, rows // 2 + 1), slice(cols // 2 - 1, cols // 2 + 1))
    z[hole] = np.nan
    steps = edge_steps(z)
    nan_cell = np.isnan(z)
    assert np.all(np.isnan(steps[:, nan_cell]))           # edges from the hole
    touches = nan_cell[None] | np.isnan(neighbor_stack(z))  # edges touching the hole
    off = _off_grid(shape)
    np.testing.assert_array_equal(np.isnan(steps), touches | off)


# --- edge_steps: compensation of the k x k mean (divide by 1 - 1/k) -----------

from src.models.terrain import step_reference_mean  # noqa: E402



def _interior(shape, margin):
    """(rows, cols) mask of cells at least `margin` cells from every border."""
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    return ((r >= margin) & (r < shape[0] - margin)
            & (c >= margin) & (c < shape[1] - margin))


def _crossing_edges(upper, shape, k):
    """Expected step (+-1 per crossing, 0 elsewhere) and mask of the interior
    edges whose endpoints lie on different sides of the step."""
    u = upper.astype(float)
    jump = neighbor_stack(u) - u
    cross = np.nan_to_num(jump) != 0
    cross &= _interior(shape, k // 2 + 2)[None]
    return jump, cross


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_step_measured_at_true_height(shape, axis):
    k, h = 5, 0.5
    idx = np.mgrid[0:shape[0], 0:shape[1]][axis]
    upper = idx >= shape[axis] // 2       # step across rows (axis 0) or columns (1)
    z = _ramp(shape) + h * upper
    steps = edge_steps(z, k=k)
    jump, cross = _crossing_edges(upper, shape, k)
    # both straight and diagonal edges cross the step
    assert cross[_straight()].any() and cross[_diagonal()].any()
    np.testing.assert_allclose(steps[cross], h * jump[cross], rtol=0, atol=1e-9)
    # the uncompensated value is h * (1 - 1/k) on the same edges
    raw = edge_steps(z, k=k, compensate=False)
    np.testing.assert_allclose(raw[cross], h * (1 - 1 / k) * jump[cross], atol=1e-9)


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
@pytest.mark.parametrize("deg_x, deg_y", [(8.0, 5.0), (-3.0, 12.0)])
@pytest.mark.parametrize("compensate", [True, False])
def test_ramp_has_no_forbidden_edge_with_or_without_compensation(
        shape, deg_x, deg_y, compensate):
    steps = edge_steps(_ramp(shape, deg_x, deg_y), compensate=compensate)
    off = _off_grid(shape)
    np.testing.assert_allclose(steps[~off], 0.0, atol=1e-9)
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    np.testing.assert_array_equal(mask, off)


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
def test_oblique_step_report(shape):
    """45-degree step: compensation is only approximate here (report only)."""
    k, h = 5, 0.5
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    upper = (r + c) >= (shape[0] + shape[1]) // 2
    z = _ramp(shape) + h * upper
    jump, cross = _crossing_edges(upper, shape, k)
    for compensate in (False, True):
        steps = edge_steps(z, k=k, compensate=compensate)
        up = steps / np.where(cross, jump, np.nan)      # signed -> height
        for name, dirs in (("straight", _straight()), ("diagonal", _diagonal())):
            vals = up[dirs][cross[dirs]]
            assert vals.size and np.all(np.isfinite(vals))
            print(f"\noblique 45deg h={h} shape={shape} compensate={compensate} "
                  f"{name}: min {vals.min():.4f} max {vals.max():.4f} "
                  f"mean {vals.mean():.4f} (n={vals.size})")


@pytest.mark.parametrize("shape", [(15, 21), (22, 17)])
def test_compensate_false_is_the_previous_formula(shape):
    k = 5
    rows, cols = shape
    z, _ = _step_east(shape)
    z[rows // 3, cols // 3] = np.nan                    # a NaN hole too
    zs = step_reference_mean(z, k)
    zs = np.where(np.isnan(z), np.nan, zs)
    previous = (neighbor_stack(z) - z) - (neighbor_stack(zs) - zs)
    raw = edge_steps(z, k=k, compensate=False)
    np.testing.assert_array_equal(raw, previous)        # bitwise, NaN included
    comp = edge_steps(z, k=k)
    np.testing.assert_array_equal(np.isnan(comp), np.isnan(previous))
    np.testing.assert_allclose(comp, previous / (1 - 1 / k), rtol=1e-12)
