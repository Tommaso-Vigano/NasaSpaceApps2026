import importlib.util
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from rasterio.transform import from_origin  # noqa: E402

from src.data.adapters.implementations.thermal_inertia_adapter import (  # noqa: E402
    ThermalInertiaAdapter,
)
from src.data.adapters.registry import AdapterRegistry  # noqa: E402
from src.data.map import Map  # noqa: E402
from src.data.types import Datatype  # noqa: E402

ROWS, COLS = 6, 8
STEP = 0.01
WEST, NORTH = 77.40, 18.50  # outer edge of the top-left pixel
MARS_CRS = "+proj=longlat +R=3396190 +no_defs"


@pytest.fixture
def ti_tiff(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.uniform(150.0, 600.0, size=(ROWS, COLS)).astype(np.float32)
    values[0, 0] = 0.0
    values[2, 3] = 0.0
    values[5, 7] = 0.0
    values[4, 1] = 8000.0
    path = tmp_path / "ti.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=ROWS, width=COLS, count=1,
        dtype="float32", crs=MARS_CRS, transform=from_origin(WEST, NORTH, STEP, STEP),
    ) as dst:
        dst.write(values, 1)
    return path, values


@pytest.fixture
def ti_map(ti_tiff):
    path, _ = ti_tiff
    with pytest.warns(UserWarning, match="1 thermal inertia value"):
        return ThermalInertiaAdapter().adapt(path)


def _assert_grid_consistent(m):
    # Same check as SamplingUtils.sample_on_grid.
    rows, cols = m.values.shape
    dx, dy = m.granularity
    assert dx > 0 and dy > 0
    expected = (m.top_left[0] + (cols - 1) * dx, m.top_left[1] - (rows - 1) * dy)
    assert np.allclose(m.bottom_right, expected, rtol=0, atol=1e-8)


def test_registered():
    assert "Jezero-thermal-inertia" in AdapterRegistry.get_available()


def test_values(ti_tiff, ti_map):
    _, raw = ti_tiff
    m = ti_map
    assert m.values.shape == (ROWS, COLS)
    for r, c in [(0, 0), (2, 3), (5, 7)]:
        assert np.isnan(m.values[r, c])          # 0 = nodata
    assert np.isnan(m.values[4, 1])              # 8000 out of range
    valid = np.isfinite(m.values)
    assert valid.sum() == ROWS * COLS - 4
    assert np.allclose(m.values[valid], raw.astype(np.float64)[valid])


def test_type_title_metadata(ti_map):
    m = ti_map
    assert m.value_type == Datatype.THERMAL_INERTIA
    assert m.title.startswith("Thermal Inertia Map ")
    meta = json.loads(m.metadata)
    assert meta["coordinate_units"] == "degrees"
    assert meta["value_unit"] == "J m^-2 K^-1 s^-1/2"
    assert "10.5281/zenodo.4931348" in meta["citation"]
    assert meta["license"] == "CC BY 4.0"
    assert "20%" in meta["accuracy_note"]
    assert meta["description"]


def test_grid(ti_map):
    m = ti_map
    assert m.top_left == pytest.approx((WEST + STEP / 2, NORTH - STEP / 2))
    assert m.granularity == pytest.approx((STEP, STEP))
    assert m.bottom_right == pytest.approx(
        (WEST + STEP / 2 + (COLS - 1) * STEP, NORTH - STEP / 2 - (ROWS - 1) * STEP)
    )
    _assert_grid_consistent(m)


def test_no_warning_when_all_in_range(tmp_path):
    import warnings

    path = tmp_path / "ok.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32",
        crs=MARS_CRS, transform=from_origin(WEST, NORTH, STEP, STEP),
    ) as dst:
        dst.write(np.array([[0, 200], [300, 400]], dtype=np.float32), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = ThermalInertiaAdapter().adapt(path)
    assert np.isnan(m.values[0, 0]) and np.isfinite(m.values).sum() == 3


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy not installed")
def test_world_model_slice_with_elevation(ti_map):
    from src.data.data_model import WorldModel

    ti = ti_map
    rows, cols = 12, 16
    step = STEP / 2
    elevation = Map(
        values=np.linspace(-2600.0, -2500.0, rows * cols).reshape(rows, cols),
        granularity=(step, step),
        top_left=(WEST + step / 2, NORTH - step / 2),
        bottom_right=(WEST + step / 2 + (cols - 1) * step,
                      NORTH - step / 2 - (rows - 1) * step),
        metadata=json.dumps({"coordinate_units": "degrees", "value_unit": "m"}),
        value_type=Datatype.ELEVATION,
        title="Elevation Map synthetic",
    )
    world = WorldModel([ti, elevation])
    sliced = world.make_slice(ti.top_left, ti.bottom_right, 0.005)
    maps = list(sliced)
    assert len(maps) == 2
    assert {m.value_type for m in maps} == {Datatype.THERMAL_INERTIA, Datatype.ELEVATION}
    assert maps[0].values.shape == maps[1].values.shape == sliced.shape
    for m in maps:
        assert np.isfinite(m.values).any()
