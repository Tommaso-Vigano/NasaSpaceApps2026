# Profile-based penalty configuration

Call the manager directly instead of the external `configure_penalties` helper.
The existing `RoutingExecutor`, profiles, cost formulas, and pathfinder API stay the
same. This method also works on `FixedPenaltyManager`.

```python
import json
from src.app.routing_executor import RoutingExecutor

# executor = RoutingExecutor(top_left, bottom_right, profile)
with open("penalties.json", encoding="utf-8") as file:
    config = json.load(file)
executor.penalty_manager.configure_from_json(executor.profile, config)
# result = executor.execute(world_slice, start, end, elevation_map=elevation)
```

Example `penalties.json`:

```json
{
  "penalties": {
    "SLOPE": true,
    "STEP_HEIGHT": true,
    "THERMAL_INERTIA": true,
    "TERRAIN_ETA": true,
    "LANDER_DISTANCE": false,
    "TEMPERATURE": false,
    "ALBEDO": false
  }
}
```

The method accepts a decoded JSON object or a JSON string, not a file path.
**Omitted switches are disabled**, including SLOPE. `{}` disables all penalties;
it retains the movement baseline and missing-elevation handling. For elevation-only
routing use `{"penalties": {"SLOPE": true}}`. Missing values in disabled cell layers
do not block cells. Configuration replaces previous registrations and validates
all fields before changing the manager. Without calling the new method, existing
manual registration and default slope behavior are unchanged.

Keys use exact `Datatype` member names. Use `SLOPE`, not `ELEVATION`, for elevation
edge penalties; ELEVATION and PENALTY are not cell-penalty switches. JSON determines
which penalties are enabled; Python profile type selects the existing rule:

| Layer | HumanProfile (astronaut) | RoverProfile |
|---|---|---|
| TERRAIN_ETA | Soft threshold | Dune cutoff |
| STEP_HEIGHT | Astronaut step cutoff | Rover step cutoff |
| THERMAL_INERTIA | No penalty | Minimum thermal inertia cutoff |
| LANDER_DISTANCE, RETURN_TIME, EVA_TIME | Astronaut cutoffs | No penalty |
| VISIBILITY | Weighted penalty | Weighted penalty |
| DUST_TAU, SOLAR_FLUX | Existing environmental cutoffs | Same cutoffs |

ExpeditionProfile combines member cell penalties by element-wise maximum, as in
the existing default rules. Cell thresholds come from `src/models/constraints.py`
and `src/constants.py`; profile dictionaries are not modified. Each manager keeps
its existing slope thresholds and weights (the original and fixed managers differ).

TEMPERATURE, WIND_SPEED, and ALBEDO need explicit parameters to be enabled:

```json
{
  "penalties": {
    "SLOPE": true,
    "TEMPERATURE": true,
    "WIND_SPEED": true,
    "ALBEDO": true
  },
  "weights": {
    "temperature_min_k": 200,
    "temperature_max_k": 300,
    "wind_speed_max_ms": 20,
    "albedo_weight": 2,
    "visibility": 0.5
  }
}
```

These numbers illustrate configuration, not validated operational limits. Units
must match layer values (temperature Kelvin, wind m/s). Temperature permits either
or both bounds. Bounds and weights must be finite nonnegative numbers; optional
bounds and albedo weight may be null. Temperature minimum must be below maximum.
DUST and RADIATION have no default rules and cannot be enabled through this method;
no scientific policy is invented for them. Use DUST_TAU for the existing optical-depth
rule. Unsupported names and non-boolean switches produce clear errors.

The existing region-routing JSON request can include a `penalty_config` field with
this object. Omitting that field keeps its previous elevation-only slope behavior.
The service currently supplies only elevation data and uses HumanProfile: switches
do not load new datasets or change the profile. Other application callers select
HumanProfile, RoverProfile, or ExpeditionProfile when constructing their executor.

The older `configure_penalties` and `register_default_penalties` entry points remain
available for compatibility. New callers need neither external setup function.
