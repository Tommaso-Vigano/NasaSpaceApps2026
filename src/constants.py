"""Project-wide constants (team rule: modules hold no module-level constants).

Every value carries the ID of its source in square brackets (full references in
``SOURCES``) or the label "TEAM ASSUMPTION" when it is not taken from a reference.
All quantities are SI (m, s, kg, W) and degrees for angles, unless stated.

This module imports only ``math`` and ``pathlib``: nothing from the project, so it can
be imported anywhere without circular imports. Project types (e.g. Datatype) are
referred to by name and resolved by the modules that use them.
"""

import math
from pathlib import Path

# ==========================================================================
# Physics
# ==========================================================================
MARS_RADIUS_M = 3_396_190.0  # IAU Mars sphere radius (Mars Trek / USGS) [R31]
G_MARS = 3.721  # m/s^2, commonly used surface gravity (fact sheet: 3.71) [R32]
G_EARTH = 9.80665  # m/s^2, standard gravity (CGPM 1901)
G_REL = G_MARS / G_EARTH  # ~0.379, ratio Mars/Earth gravity

# Terrain coefficient eta for the Pandolf equation.
# TEAM ASSUMPTION, derived from the terrain factors reported in [R18].
ETA_TERRAIN = {
    "rock": 1.0,
    "delta": 1.1,
    "regolith": 1.1,
    "rocky": 1.3,
    "sand": 1.5,
}

SOURCES = {
    "R1": "NASA, Mars 2020 Perseverance Rover Components (mars.nasa.gov)",
    "R2": "Rankin et al. 2023, IEEE Aerospace Conference (Perseverance driving performance)",
    "R3": "NASA/JPL PIA07999, Opportunity at Purgatory Dune",
    "R4": "Morgan et al. 2018, Space Science Reviews 214:104 (InSight landing site thermal inertia)",
    "R5": "Golombek & Rapp 1997, JGR 102(E2):4117 (Mars rock size-frequency distributions)",
    "R7": "NASA SP-4214 / NASA TM X-64623, Lunar Roving Vehicle operations handbook",
    "R9": "MIT 16.89 Space Systems Engineering, 2007",
    "R10": "NASA Space Exploration Vehicle (SEV) fact sheet",
    "R11": "NASA EVA-EXP-0075, Exploration EVA System Concept of Operations",
    "R12": "NASA ICES-2025-294, step height for suited locomotion",
    "R13": "NASA ICES 2026, Mars EVA suit mass and metabolic rates",
    "R15": "Pandolf, Givoni & Goldman 1977, J Appl Physiol 43:577",
    "R18": "Richmond et al. 2015, J Sport Hum Perf 3:1 (terrain factors)",
    "R20": "NASA NESC, space weather / SEP warning and shelter times",
    "R24": "Mars Climate Database v6.2",
    "R27": "Minetti et al. 2002, J Appl Physiol 93:1039 (energy cost on gradients)",
    "R28": "Tobler 1993, Three presentations on geographical analysis (hiking function)",
    "R29": "Horn 1981, Proc IEEE 69:14 (hill shading and slope)",
    "R30": "Riley, DeGloria & Elliot 1999, Terrain Ruggedness Index",
    "R31": (
        "Archinal et al. 2018, Report of the IAU Working Group on Cartographic Coordinates "
        "and Rotational Elements: 2015, Celest. Mech. Dyn. Astr. 130:22 "
        "(Mars sphere R = 3396.19 km, used by Mars Trek / USGS products)"
    ),
    "R32": (
        "NASA NSSDCA Mars Fact Sheet (surface gravity 3.71 m/s^2; "
        "3.721 m/s^2 is the commonly used value)"
    ),
}

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
KELVIN_OFFSET = 273.15  # 0 degC in kelvin (K -> degC conversion)

# ==========================================================================
# Grid
# ==========================================================================
# 8-connected neighbours as (drow, dcol), names N, NE, E, SE, S, SO, O, NO; row 0 is
# north. Per-edge arrays (8, rows, cols) follow this order (src.models.terrain).
# It is the same order as src.routing.algorithm.DIRECTIONS, which is (dcol, drow).
OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIRECTION_NAMES = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")
GRID_ATOL = 1e-9  # degrees: tolerance when comparing the corners / steps of two grids
SQRT2 = math.sqrt(2.0)  # length factor of a diagonal step

# ==========================================================================
# Mars Climate Database (MCD) diurnal CSV [R24]
# ==========================================================================
# Substrings of the columns we use (matched by substring in src.models.gates).
COL_TIME = "Local time"
COL_FLUX = "solar flux"
COL_TAU = "optical depth"
COL_TEMP = "Temperature (K)"  # air temperature (2 m above the surface)
COL_TSURF = "Surface temperature"
COL_WIND = "wind speed"
MCD_COLUMN_KEYS = (COL_TIME, COL_FLUX, COL_TAU, COL_TEMP, COL_TSURF, COL_WIND)
# A column matching a key is skipped if it also contains one of these words
# (case-insensitive), so COL_TEMP never takes the surface temperature.
MCD_COLUMN_EXCLUDE = {COL_TEMP: ("surface",)}

# ==========================================================================
# GO / NO-GO gates (src.models.gates, penalty_config)
# ==========================================================================
DUST_TAU_MAX = 2.0  # TEAM ASSUMPTION: optical depth >= 2 = active dust storm -> NO-GO
SOLAR_FLUX_MIN_W_M2 = 100.0  # TEAM ASSUMPTION: horizontal solar flux >= 100 W/m^2 = daylight

# ==========================================================================
# Penalties (src.app.penalty: penalty_manager_fixed, penalty_config)
# Unit: "cell equivalents", 1 = crossing one flat cell on rock.
# ==========================================================================
BASE_COST = 1.0  # TEAM ASSUMPTION: cost of one flat cell on rock
# TEAM ASSUMPTION. Walker: with Tobler the time grows by 85% at 10 deg and by 258% at
# 20 deg; the line 2*|s|/20 sits in between. Rover: speed does not depend on the slope,
# the low weight only makes it prefer flatter ground.
DEFAULT_SLOPE_WEIGHTS = {"walker": 2.0, "rover": 0.5}
# Datatype names with no penalty by default (they are not costs); resolved to
# src.data.types.Datatype in penalty_manager_fixed.py.
DEFAULT_IGNORED_DATATYPE_NAMES = ("ALBEDO", "DUST", "RADIATION")
# TEAM ASSUMPTION: leaving the lander's line of sight costs like half a cell more per
# cell, so the path prefers to stay in view without long detours for it.
VISIBILITY_WEIGHT = 0.5
# TERRAIN_ETA for the walker: SoftThreshold(1, 1) = eta - 1, the extra time of
# "time x eta" (option B, TEAM ASSUMPTION).
TERRAIN_ETA_SOFT_THRESHOLD = 1.0
TERRAIN_ETA_SOFT_WEIGHT = 1.0

# ==========================================================================
# Models and adapters
# ==========================================================================
# Upper bound on n_rays * n_samples of the viewshed in ModelMapsAdapter (memory: a few
# float64 arrays of this size, 3e7 ~ 240 MB each). TEAM ASSUMPTION.
MAX_VIEWSHED_SAMPLES = 30_000_000
LANDER_ANTENNA_H_M = 2.0  # TEAM ASSUMPTION: lander antenna height above the ground

# Horizontal field of view (deg) used to draw the photo cones (src.landmarks.photo_stops).
CAMERA_FOV_DEG = {"NAVCAM_LEFT": 45.0, "MCZ_LEFT": 15.0}
DEFAULT_FOV_DEG = 45.0

# ==========================================================================
# Jezero data for the scripts (scripts/demo_penalties_jezero.py, inspect_terrain.py)
# ==========================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
# Data folder, chosen by the scripts (scripts/_data_dir.py) in this order:
# --data-dir, the environment variable JEZERO_DATA_DIR_ENV (also from REPO_ROOT/.env),
# DEFAULT_JEZERO_DATA_DIR. Provisional choice, to be decided with the team.
DEFAULT_JEZERO_DATA_DIR = REPO_ROOT.parent / "jezero_data"  # "jezero_data" next to the repo
JEZERO_DATA_DIR_ENV = "JEZERO_DATA_DIR"
# Paths relative to the data folder.
# Provisional DEM: +-3 km around the lander at 1 m; to be replaced by a colleague's crop.
DEM_REL_PATH = "01_topografia/jezero_HiRISE_DEM_lander_6km_1m_PROVVISORIO.tif"
# "Intrepretative" is spelled this way in the dataset.
NOAH_REL_PATH = ("11_terreno_NOAH-H/estratto/NOAH-H_terrain_classifications_of_Jezero_Crater_Mars/"
                 "NOAH-H_mosaics/Intrepretative_groups/Jezero_NOAHH_mosaic_IG.tif")
THEMIS_REL_PATH = "05_termico_suolo/numerici/Jezero_Thermal_Inertia_THEMIS_v1.tiff"
MCD_DIR_REL_PATH = "06_polvere_atmosfera/MCD"
MCD_REL_PATH = MCD_DIR_REL_PATH + "/diurnal_Ls021p7_today.csv"
WAYPOINTS_REL_PATH = "08_perseverance/M20_waypoints.json"
PHOTOS_REL_PATH = "08_perseverance/raw_images/foto_georeferenziate.csv"
DEMO_MAP_SCRIPT = REPO_ROOT / "scripts" / "demo_map.py"

LANDER_SOL13_LONLAT = (77.45088572, 18.44462715)  # (lon E, lat N), M20_waypoints.json, sol 13

# demo_penalties_jezero.py defaults
DEMO_RES_M = 20.0  # grid resolution: GridGraph (networkx) is slow, 300x300 ~ 2 s
DEMO_EVA_START_H = 9.0  # TEAM ASSUMPTION: local hour, to be decided with the team
DEMO_EVA_DURATION_H = 6.0  # [R11] nominal EVA; start hour to be decided with the team
DEMO_OUT_DIR = "out_demo"
# Destination radius: 30-minute walkback radius (SEP warning) [R11, R20],
# HumanProfile.path["walkback_radius_30m"].
DEST_MAX_DIST_M = 1800.0
# Uniform (MCD) layers checked for NO-GO before build_graph; Datatype names.
UNIFORM_DATATYPE_NAMES = ("DUST_TAU", "SOLAR_FLUX", "WIND_SPEED", "TEMPERATURE")

# inspect_terrain.py defaults
INSPECT_HALF_SIDE_M = 2000.0  # half side of the box around the lander
INSPECT_RESOLUTION_M = 5.0
