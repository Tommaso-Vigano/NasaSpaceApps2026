"""GO/NO-GO gates for an EVA from a Mars Climate Database diurnal profile [R24].

The MCD web interface exports a CSV with one row per local hour and quoted
headers such as "Local time (Martian hour)" or
"Incident solar flux on horizontal surface (W/m2)". Columns are matched by
substring, so small changes in the headers and extra columns are tolerated.
"""

import csv
import math

import numpy as np

# Column substrings (any other column is ignored), the exclusion table and the gate
# thresholds live in src/constants.py; the COL_* names are re-exported from here.
from src.constants import (  # noqa: F401
    COL_FLUX,
    COL_TAU,
    COL_TEMP,
    COL_TIME,
    COL_TSURF,
    COL_WIND,
    DUST_TAU_MAX,
    MCD_COLUMN_EXCLUDE,
    MCD_COLUMN_KEYS,
    SOLAR_FLUX_MIN_W_M2,
)


def _matches(key, name):
    return key in name and not any(w in name.lower() for w in MCD_COLUMN_EXCLUDE.get(key, ()))


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load_mcd_diurnal(csv_path):
    """Read an MCD diurnal CSV -> dict {original column name: np.array}.

    Blank lines and lines starting with '#' are skipped; the first remaining
    row is the header. Columns are matched by substring and only the first
    column matching each of "Local time", "solar flux", "optical depth",
    "Temperature (K)" (air), "Surface temperature", "wind speed" is kept.
    The air key "Temperature (K)" skips any column containing "surface"
    (case-insensitive), so air and surface temperature are both kept and never
    swapped, in any column order. Non-numeric cells become NaN.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.reader(f)
            if row and any(cell.strip() for cell in row)
            and not row[0].lstrip().startswith("#")
        ]
    if not rows:
        raise ValueError(f"{csv_path}: no header found.")
    header = [h.strip() for h in rows[0]]
    data = rows[1:]

    out = {}
    for key in MCD_COLUMN_KEYS:
        for j, name in enumerate(header):
            if _matches(key, name):
                out[name] = np.array(
                    [_to_float(r[j]) if j < len(r) else math.nan for r in data]
                )
                break
    return out


def _column(mcd, substring):
    for name, values in mcd.items():
        if _matches(substring, name):
            return np.asarray(values, dtype=np.float64)
    raise KeyError(f"No MCD column containing {substring!r}; columns: {list(mcd)}")


def daylight_window(mcd, min_flux=SOLAR_FLUX_MIN_W_M2):
    """(start_h, end_h) when the horizontal solar flux is >= ``min_flux``
    (W/m^2), interpolating linearly between hourly samples. (nan, nan) if the
    flux never reaches the threshold. Assumes a single contiguous daytime window.

    TEAM ASSUMPTION: min_flux = 100 W/m^2.
    """
    hours = _column(mcd, COL_TIME)
    flux = _column(mcd, COL_FLUX)
    order = np.argsort(hours)
    hours, flux = hours[order], flux[order]
    above = np.flatnonzero(flux >= min_flux)
    if above.size == 0:
        return math.nan, math.nan

    def cross(i0, i1):
        h0, h1, f0, f1 = hours[i0], hours[i1], flux[i0], flux[i1]
        return float(h0 + (min_flux - f0) * (h1 - h0) / (f1 - f0))

    i, j = above[0], above[-1]
    start = float(hours[i]) if i == 0 else cross(i - 1, i)
    end = float(hours[j]) if j == hours.size - 1 else cross(j, j + 1)
    return start, end


def dust_storm_nogo(mcd, tau_max=DUST_TAU_MAX):
    """True if the dust optical depth reaches ``tau_max``.

    TEAM ASSUMPTION: tau >= 2 means an active dust storm -> NO-GO.
    """
    return bool(np.nanmax(_column(mcd, COL_TAU)) >= tau_max)


def solar_power_ratio(mcd, mcd_reference):
    """Peak solar flux / peak solar flux of a reference profile (e.g. clear sky)."""
    return float(np.nanmax(_column(mcd, COL_FLUX)) / np.nanmax(_column(mcd_reference, COL_FLUX)))


def eva_go(mcd, start_h, duration_h, min_flux=SOLAR_FLUX_MIN_W_M2, tau_max=DUST_TAU_MAX):
    """(go, reasons): NO-GO if a dust storm is active or the EVA
    [start_h, start_h + duration_h] leaves the daylight window."""
    reasons = []
    if dust_storm_nogo(mcd, tau_max):
        tau = float(np.nanmax(_column(mcd, COL_TAU)))
        reasons.append(f"dust storm: optical depth {tau:.2f} >= {tau_max}")
    start, end = daylight_window(mcd, min_flux)
    stop_h = start_h + duration_h
    if math.isnan(start):
        reasons.append(f"no daylight: solar flux never reaches {min_flux} W/m2")
    elif start_h < start or stop_h > end:
        reasons.append(
            f"EVA {start_h:.2f}-{stop_h:.2f} h outside daylight window "
            f"{start:.2f}-{end:.2f} h (flux >= {min_flux} W/m2)"
        )
    return not reasons, reasons
