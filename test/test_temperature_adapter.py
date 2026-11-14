import json

import numpy as np
import pytest

from src.data.adapters.implementations.temperature_adapter import TemperatureAdapter
from src.data.adapters.registry import AdapterRegistry
from src.data.types import Datatype

HEADER = [
    "Local time (Martian hour)",
    "Temperature (K)",
    "Surface temperature (K)",
    "Horizontal wind speed (m/s)",
    "Surface pressure (Pa)",
    "W-E wind component (m/s)",
]


def _write_csv(path, header=HEADER, hours=range(25)):
    lines = [",".join(f'"{h}"' for h in header)]
    for h in hours:
        # air = 200 + h, surface = 180 + h, but hour 24 repeats hour 0 (cyclic).
        hh = h % 24
        row = [str(h), str(200.0 + hh), str(180.0 + hh), "3.1", "730.0", "-1.0"]
        lines.append(",".join(row[: len(header)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def diurnal(tmp_path):
    return _write_csv(tmp_path / "diurnal_Ls270_DUSTSTORM.csv")


def _assert_grid_consistent(m):
    rows, cols = m.values.shape
    dx, dy = m.granularity
    expected = (m.top_left[0] + (cols - 1) * dx, m.top_left[1] - (rows - 1) * dy)
    assert np.allclose(m.bottom_right, expected, rtol=0, atol=1e-8)


def test_registered():
    assert "MCD-temperature" in AdapterRegistry.get_available()


def test_air_temperature_at_13(diurnal):
    m = TemperatureAdapter(local_time_h=13.0).adapt(diurnal)
    assert m.value_type == Datatype.TEMPERATURE
    assert np.all(m.values == pytest.approx(213.0))
    meta = json.loads(m.metadata)
    assert meta["value_unit"] == "K"
    assert meta["coordinate_units"] == "degrees"
    assert meta["field"] == "air"
    assert meta["scenario"] == "Ls270_DUSTSTORM"
    assert meta["local_time_h"] == 13.0
    assert "Mars Climate Database" in meta["citation"]
    assert m.title.startswith("Temperature Map ")


def test_linear_interpolation(diurnal):
    m = TemperatureAdapter(local_time_h=13.5).adapt(diurnal)
    assert np.allclose(m.values, 213.5)


def test_surface_field(diurnal):
    m = TemperatureAdapter(local_time_h=13.0, field="surface").adapt(diurnal)
    assert np.allclose(m.values, 193.0)
    assert json.loads(m.metadata)["field"] == "surface"


def test_cyclic(diurnal):
    at_0 = TemperatureAdapter(local_time_h=0.0).adapt(diurnal).values
    at_24 = TemperatureAdapter(local_time_h=24.0).adapt(diurnal).values
    assert np.allclose(at_0, at_24)
    assert np.allclose(at_0, 200.0)
    # Between 23 h and 24 h (= 0 h) the value goes from 223 back to 200.
    late = TemperatureAdapter(local_time_h=23.5).adapt(diurnal).values
    assert np.allclose(late, 211.5)


def test_cyclic_without_hour_24(tmp_path):
    path = _write_csv(tmp_path / "diurnal_x.csv", hours=range(24))
    late = TemperatureAdapter(local_time_h=23.5).adapt(path).values
    assert np.allclose(late, 211.5)


def test_default_grid_is_consistent(diurnal):
    m = TemperatureAdapter().adapt(diurnal)
    assert m.granularity == (0.01, 0.01)
    assert m.top_left == pytest.approx((77.160, 18.721))
    # Last cell centres inside the CTX bounds.
    assert m.bottom_right == pytest.approx((77.69, 18.211))
    assert m.values.shape == (52, 54)
    _assert_grid_consistent(m)
    assert m.bottom_right[0] <= 77.699 and m.bottom_right[1] >= 18.211 - 1e-12


def test_custom_grid_is_consistent(diurnal):
    m = TemperatureAdapter(
        top_left=(77.0, 18.5), bottom_right=(77.05, 18.47), resolution_deg=0.007,
    ).adapt(diurnal)
    _assert_grid_consistent(m)
    assert m.bottom_right[0] <= 77.05 and m.bottom_right[1] >= 18.47


def test_missing_local_time_raises_value_error(tmp_path):
    path = tmp_path / "not_diurnal.csv"
    path.write_text('"Ls","Temperature (K)"\n0,200\n15,210\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Local time"):
        TemperatureAdapter().adapt(path)


def test_missing_field_column_raises_key_error(tmp_path):
    path = _write_csv(tmp_path / "diurnal_y.csv", header=HEADER[:2])
    with pytest.raises(KeyError, match="Surface temperature"):
        TemperatureAdapter(field="surface").adapt(path)


def test_invalid_field():
    with pytest.raises(ValueError):
        TemperatureAdapter(field="soil")
