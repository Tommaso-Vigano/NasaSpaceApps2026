import numpy as np

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

E = DIRECTION_NAMES.index("E")
O = DIRECTION_NAMES.index("O")


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
    assert np.isnan(nb[E, 1, 3])
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
    assert abs(steps[E, 5, 4] - 0.5) <= 0.15  # step up going east
    assert abs(steps[O, 5, 5] + 0.5) <= 0.15  # step down going west
    mask = step_masks(steps, max_up=0.30, max_down=0.225)
    assert mask[E, 5, 4] and mask[O, 5, 5]
    # away from the step nothing is forbidden (except edges leaving the grid)
    assert not mask[E, 5, 1]
    assert mask[E, 5, 10]


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
    assert np.isnan(steps[E, 3, 2]) and np.all(np.isnan(steps[:, 3, 3]))


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
