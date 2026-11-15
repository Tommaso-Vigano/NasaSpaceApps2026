import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from rasterio.transform import from_origin  # noqa: E402

from src.data.adapters.implementations.terrain_class_adapter import (  # noqa: E402
    TerrainClassAdapter,
)
from src.data.adapters.registry import AdapterRegistry  # noqa: E402
from src.data.types import Datatype  # noqa: E402
from src.models.constraints import ETA_TERRAIN  # noqa: E402

def _geo():
    """Geometry of the synthetic NOAH-H raster (eqc on the Mars sphere, near Jezero).

    Top row: one 5 m cell per case; bottom row: all Non-bedrock.
      0..4 classes 1..5, 5 class 0, 6 nodata 127,
      7 mixed: 2 m Bedrock + 3 m Small ripples,
      8 40% valid (2 m Bedrock + 3 m nodata), 9 60% valid (3 m Boulder + 2 m nodata)
    """
    r = 3396190.0
    cell_px = 20                      # one 5 m output cell
    n_cells = 10
    return SimpleNamespace(
        r=r,
        crs=f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R={r:.0f} +units=m +no_defs",
        pixel=0.25,                   # m
        cell_px=cell_px,
        stripe_px=4,                  # one 1 m intermediate column
        x0=4_590_000.0,               # outer top-left edge
        y0=1_094_000.0,
        n_cells=n_cells,
        rows_px=2 * cell_px,
        cols_px=n_cells * cell_px,
        stripes={7: [2, 2, 4, 4, 4], 8: [2, 2, 127, 127, 127], 9: [3, 3, 3, 0, 127]},
    )


def lon(x):
    return math.degrees(x / _geo().r)


def lat(y):
    return math.degrees(y / _geo().r)


def box(x_left, y_top, width_m, height_m):
    return (lon(x_left), lat(y_top)), (lon(x_left + width_m), lat(y_top - height_m))


@pytest.fixture
def noah_tiff(tmp_path):
    g = _geo()
    values = np.ones((g.rows_px, g.cols_px), dtype=np.uint8)
    top = values[:g.cell_px]
    for cell, cls in enumerate([1, 2, 3, 4, 5, 0, 127]):
        top[:, cell * g.cell_px:(cell + 1) * g.cell_px] = cls
    for cell, stripes in g.stripes.items():
        for i, cls in enumerate(stripes):
            c = cell * g.cell_px + i * g.stripe_px
            top[:, c:c + g.stripe_px] = cls
    path = tmp_path / "noah.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=g.rows_px, width=g.cols_px, count=1,
        dtype="uint8", nodata=127, crs=g.crs,
        transform=from_origin(g.x0, g.y0, g.pixel, g.pixel),
    ) as dst:
        dst.write(values, 1)
    return path


def full_box():
    g = _geo()
    return box(g.x0, g.y0, g.cols_px * g.pixel, g.rows_px * g.pixel)


@pytest.fixture
def eta_map(noah_tiff):
    return TerrainClassAdapter(*full_box()).adapt(noah_tiff)


def _assert_grid_consistent(m):
    rows, cols = m.values.shape
    dx, dy = m.granularity
    assert dx > 0 and dy > 0
    expected = (m.top_left[0] + (cols - 1) * dx, m.top_left[1] - (rows - 1) * dy)
    assert np.allclose(m.bottom_right, expected, rtol=0, atol=1e-8)


def test_registered():
    assert "Jezero-NOAH-H-terrain" in AdapterRegistry.get_available()


def test_class_eta(eta_map):
    g = _geo()
    v = eta_map.values
    assert v.shape == (2, g.n_cells)
    expected = [ETA_TERRAIN[k] for k in ("regolith", "rock", "rocky", "sand", "sand")]
    assert v[0, :5] == pytest.approx(expected)
    assert np.allclose(v[1], ETA_TERRAIN["regolith"])


def test_nodata_is_nan(eta_map):
    assert np.isnan(eta_map.values[0, 5])   # class 0
    assert np.isnan(eta_map.values[0, 6])   # nodata 127


def test_block_mean(eta_map):
    expected = (2 * ETA_TERRAIN["rock"] + 3 * ETA_TERRAIN["sand"]) / 5
    assert eta_map.values[0, 7] == pytest.approx(expected)


def test_min_valid_fraction(noah_tiff, eta_map):
    assert np.isnan(eta_map.values[0, 8])                              # 40% < 50%
    assert eta_map.values[0, 9] == pytest.approx(ETA_TERRAIN["rocky"])  # 60%, valid only
    loose = TerrainClassAdapter(*full_box(), min_valid_fraction=0.3).adapt(noah_tiff)
    assert loose.values[0, 8] == pytest.approx(ETA_TERRAIN["rock"])
    strict = TerrainClassAdapter(*full_box(), min_valid_fraction=0.7).adapt(noah_tiff)
    assert np.isnan(strict.values[0, 9])


def test_grid(eta_map):
    g = _geo()
    m = eta_map
    step = math.degrees(5.0 / g.r)
    assert m.granularity == pytest.approx((step, step), rel=1e-12)
    assert m.top_left == pytest.approx((lon(g.x0 + 2.5), lat(g.y0 - 2.5)), abs=1e-10)
    assert m.bottom_right == pytest.approx(
        (lon(g.x0 + (g.n_cells - 0.5) * 5), lat(g.y0 - 1.5 * 5)), abs=1e-10,
    )
    _assert_grid_consistent(m)


def test_sample_on_grid_accepts_map(eta_map):
    pytest.importorskip("scipy")
    from src.data.data_model import SamplingUtils

    m = eta_map
    rows, cols = m.values.shape
    xs = m.top_left[0] + np.arange(cols) * m.granularity[0]
    ys = m.top_left[1] - np.arange(rows) * m.granularity[1]
    sampled = SamplingUtils.sample_on_grid(m, xs, ys)
    assert np.allclose(sampled, m.values, equal_nan=True)


def test_resolution_change(noah_tiff):
    g = _geo()
    m = TerrainClassAdapter(*full_box(), resolution_m=1.0).adapt(noah_tiff)
    assert m.values.shape == (g.rows_px // g.stripe_px, g.cols_px // g.stripe_px)
    assert m.values[0, 35] == pytest.approx(ETA_TERRAIN["rock"])  # cell 7, first metre
    assert m.values[0, 37] == pytest.approx(ETA_TERRAIN["sand"])
    _assert_grid_consistent(m)


def test_partially_outside(noah_tiff):
    g = _geo()
    tl, br = box(g.x0, g.y0, g.cols_px * g.pixel + 10.0, g.rows_px * g.pixel)
    m = TerrainClassAdapter(tl, br).adapt(noah_tiff)
    assert m.values.shape == (2, g.n_cells + 2)
    assert np.isnan(m.values[:, -2:]).all()
    assert m.values[0, 0] == pytest.approx(ETA_TERRAIN["regolith"])
    _assert_grid_consistent(m)


def test_entirely_outside(noah_tiff):
    g = _geo()
    tl, br = box(g.x0 + 1000.0, g.y0, 20.0, 20.0)
    with pytest.raises(ValueError, match="entirely outside"):
        TerrainClassAdapter(tl, br).adapt(noah_tiff)


def test_custom_mapping(noah_tiff):
    mapping = {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    m = TerrainClassAdapter(*full_box(), class_to_eta=mapping).adapt(noah_tiff)
    assert m.values[0, :4] == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert np.isnan(m.values[0, 4])           # class 5 not mapped
    meta = json.loads(m.metadata)
    assert meta["class_to_eta"] == {"1": 2.0, "2": 3.0, "3": 4.0, "4": 5.0}


def test_type_title_metadata(eta_map):
    m = eta_map
    assert m.value_type == Datatype.TERRAIN_ETA
    assert m.title.startswith("Terrain Eta Map ")
    meta = json.loads(m.metadata)
    assert meta["coordinate_units"] == "degrees"
    assert meta["value_unit"] == "dimensionless (Pandolf terrain coefficient eta)"
    assert meta["class_names"]["3"] == "Boulder fields"
    assert meta["resolution_m"] == 5.0
    assert "10.21954/ou.rd.17121482.v2" in meta["citation"]
    assert meta["license"] == "CC BY 4.0"
    assert "TEAM ASSUMPTION" in meta["note"]


@pytest.mark.parametrize("kwargs", [
    {"resolution_m": 2.5},
    {"resolution_m": 0.5},
    {"min_valid_fraction": 1.5},
    {"class_to_eta": {127: 1.0}},
    {"class_to_eta": {1: -1.0}},
])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        TerrainClassAdapter(*full_box(), **kwargs)


def test_swapped_corners():
    tl, br = full_box()
    with pytest.raises(ValueError):
        TerrainClassAdapter(br, tl)
