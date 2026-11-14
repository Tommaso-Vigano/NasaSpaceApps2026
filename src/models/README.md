# `src/models` — mobility constraint models

These are pure functions on `numpy` arrays (numpy and the standard library only). The grid spacing `(dx, dy)` is always in **metres**. All units are SI: m, s, kg, W, J. Angles are in degrees. The grade `G` is a percentage (`100·rise/run`). **NaN means nodata.** Any cell or edge that touches a NaN is invalid: its slope is NaN, its mask is `True` and its cost is `inf`.

The models do not depend on `Map`. `terrain.from_map(m) -> (z, dx, dy)` is the bridge. A `Map` grid is always in degrees `(dlon, dlat)`; `from_map` converts the spacing to metres at the central latitude (`dy = R·dlat`, `dx = R·dlon·cos(lat)`) and raises `ValueError` if the metadata declares `coordinate_units` other than `"degrees"`.

## Direction convention

`OFFSETS = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]` as `(drow, dcol)`, named N, NE, E, SE, S, SO, O, NO. Row 0 is north.

The horizontal length of an edge is `dy` for N/S, `dx` for E/O and `sqrt(dx²+dy²)` for the diagonals.

Per-edge arrays have shape `(8, rows, cols)`. The value `a[k, r, c]` belongs to the edge `(r,c) → (r+drow_k, c+dcol_k)`. An edge that leaves the grid is NaN, or `inf` for costs.

## Files

| File | Content |
|---|---|
| `constraints.py` | Frozen dataclasses `RoboticRover`, `CrewedRover`, `Astronaut`; `MARS_RADIUS_M`, `G_MARS`, `G_EARTH`, `G_REL`, `ETA_TERRAIN`, `SOURCES` |
| `terrain.py` | slope, TRI, smoothing, step heights, masks |
| `metabolic.py` | Pandolf, Minetti, Tobler, per-edge time/energy, per-cell bridge |
| `rocks.py` | Golombek & Rapp rock abundance, probability of a rock taller than a limit |
| `rover.py` | rover no-go mask and per-edge travel time |
| `budget.py` | Dijkstra time maps, EVA action radius, SEP isochrone, path check with point of no return |
| `gates.py` | GO/NO-GO from a Mars Climate Database diurnal CSV (daylight, dust storm) |
| `viewshed.py` | line of sight from the lander, horizon distance |

The demo script `scripts/demo_models.py` runs on a synthetic DEM and only prints results. Run it from the repo root with `python scripts/demo_models.py`.

## Formulas

**Terrain**
- `horn_slope` [R29]: `dz/dx = ((c+2f+i)-(a+2d+g))/(8dx)` and `dz/dy = ((g+2h+i)-(a+2b+c))/(8dy)`, then `slope = atan(hypot)` in degrees. Border cells are NaN.
- `tri` [R30]: `sqrt(Σ_8 (z_n − z_c)²)` in m. Border cells are NaN.
- `smooth_mean(z, k)`: a NaN-aware k×k moving average built with `sliding_window_view`.
- `edge_steps(z, k=5)`: `step(i→j) = (z_j − z_i) − (zs_j − zs_i)`, where `zs = smooth_mean(z, k)`. This is the part of the elevation change that the smooth slope does not explain. A positive value is a step up.
- `step_masks`: an edge is forbidden if `step > max_up` or `step < −max_down`.

**Metabolic**
- Pandolf [R15], in W: `M = 1.5W + 2(W+L)(L/W)² + η(W+L)(1.5V² + 0.35VG)`. W and L are in kg, V in m/s, G in %.
- Minetti [R27], in J/(kg·m): `Cw(i) = 280.5i⁵ − 58.7i⁴ − 76.8i³ + 51.9i² + 19.6i + 2.5`. The fit is valid for |i| ≤ 0.45. Values outside this range are clipped and a warning is emitted.
- Walking power on Mars, in W:
  - For `G ≥ 0`: `G_REL·Pandolf + ΔM_suit`.
  - For `G < 0`: `(G_REL·Pandolf(G=0) + ΔM_suit)·Cw(G/100)/Cw(0)`.
  - The result is floored at the standing cost `G_REL·1.5W + ΔM_suit`.
- `calibrate_dM_suit`: chooses `ΔM_suit` so that the flat-ground power at 3.5 km/h equals 498 W. That value is the NASA average walkback rate of 1700 BTU/h [R13]. With the defaults, `ΔM_suit ≈ 117.5 W`.
- Tobler [R28]: `v = 6·exp(−3.5·|tanθ + 0.05|)` km/h. It is rescaled so that `v(0) = v_flat` and then clipped at `v_max`. The fastest speed is at a grade of about −5 %.
- `edge_costs` → `(time_s, energy_J)`, each of shape `(8, rows, cols)`:
  - `d3D = hypot(dh, dz)` and `G = 100·dz/dh`.
  - V comes from Tobler. P is the walking power with η taken as the mean of the two cells.
  - `t = d3D/V` and `E = P·t`.
  - An edge is `inf` if its slope exceeds `max_slope_deg`, if its step is beyond the limits (when `steps` is given), if it touches a NaN, or if it leaves the grid.
  - The cost is **asymmetric**: going uphill and going downhill along the same edge cost different amounts.
- `cell_cost_from_edges(edge_cost, dx, dy, reduce="mean"|"max")`: takes the mean or max of the finite outgoing edges per metre and multiplies it by `dx`. The result is `inf` if no outgoing edge is finite. This is a bridge for cell-based pathfinders. **It loses the uphill/downhill asymmetry.** Note that it also takes `dx, dy`, because it needs the edge lengths.

**Rocks** [R5]

- `q(k) = 1.79 + 0.152/k` in 1/m.
- `cfa(D, k) = k·exp(−q·D)`: the fraction of area covered by rocks with diameter ≥ D (m).
- `number_density_above(D0, k)`: the number of rocks per m² with diameter ≥ D0. It is `∫_{D0}^{10 m} k q e^{−qx} / (πx²/4) dx`, computed with the trapezoidal rule on a logarithmic grid of 2000 points. It is vectorised over maps of k by integrating once per unique k value.
- `p_rock_taller_than(h, k, A) = 1 − exp(−N(D ≥ h/0.5)·A)`: the Poisson probability of at least one rock taller than h in an area A.

**Rover**

- `rover_hard_mask`: a cell is forbidden if its slope is above `max_slope_deg`, its thermal inertia is below `min_thermal_inertia` (when `ti` is given), or it is a dune (when `dunes_nogo` is set and a dune mask is given). NaN is forbidden.
- `rover_edge_time`: `t = d3D / speed_ms · (1 + p_rock)`.
  - k is the mean of the two cells, and the area is the horizontal edge length × `width_m`.
  - The edge is `inf` if either end cell is forbidden, if `|step| > max_step_m`, if it touches a NaN, or if it leaves the grid.
  - By default the slope comes from `horn_slope`, which is NaN on the border, so **border cells are forbidden for rovers**.

**Budget** (times in s)

- `dijkstra_grid(edge_cost, sources, reverse=False, solver=None)` uses a heapq Dijkstra on the 8-connected grid.
  - With `reverse=True` it gives the cost from every cell back to the sources, using the cost of the edge `v → u` (direction `(k+4) % 8`). Costs are asymmetric, so `return_time_map ≠ outbound_time_map`.
  - It is pure Python and runs in O(N log N): about 0.1 s for 40 000 cells. Crop to a few km around the lander at 5–20 m/px.
  - `solver` is a hook for a faster implementation with the same contract.
- `eva_reach_mask`: `t_out + dwell + t_ret ≤ eva_nominal_h`. The 2 km walkback limit (`max_distance_from_lander_m`) is **not** applied here.
- `sep_isochrone_mask`: `t_ret ≤ sep_return_min` [R20].
- `check_eva(path, ...) → EvaReport`, with one value per path point k:
  - `t_elapsed[k]`, the cumulative time including egress.
  - `nominal_ok[k]`: `t_elapsed + t_ret ≤ 6 h`.
  - `contingency_ok[k]`: `t_elapsed + 1.5·t_ret ≤ 6 h + 2 h` [R11].
  - `point_of_no_return`: the last index before the first contingency failure. It is `None` if every point passes and `−1` if the first point already fails.
  - `energy_cum_J`, `mean_power_W` and `sep_ok`.
  - `feasible`: every point is `nominal_ok` and the total time plus ingress is ≤ 6 h.
  - Raises `ValueError` if two consecutive points are not adjacent.

**Gates** [R24]

- `load_mcd_diurnal(csv)` reads the CSV with the `csv` module and skips `#` comment lines. Columns are matched by substring: "Local time", "solar flux", "optical depth", "Temperature (K)", "wind speed". Only the first match of each is kept, and extra columns such as the wind components are ignored.
- `daylight_window(mcd, 100)`: the first and last time with horizontal flux ≥ 100 W/m², interpolated linearly between hours. It assumes a single daytime window and an EVA that does not cross midnight.
- `dust_storm_nogo(mcd, 2.0)`: τ ≥ 2.
- `solar_power_ratio(mcd, ref)`: peak flux divided by the reference peak flux.
- `eva_go(...) → (go, reasons)`: lists the reasons for a NO-GO.

**Viewshed**

- `viewshed(z, dx, dy, observer, observer_h=2, target_h=1.8, max_range_m=5000, n_rays=720)`:
  - Radial rays are sampled every `min(dx, dy)` at the nearest cell, with the curvature drop `d²/(2R)` applied.
  - A cell is visible if the angle to a target `target_h` above it is ≥ the maximum terrain angle earlier on the ray.
  - A NaN or the grid edge blocks the rest of the ray.
  - Far away the rays are `2πr/n_rays` apart. Increase `n_rays` if that exceeds the cell size.
- `horizon_distance_m(h) = sqrt(2·R·h)`, which gives 3497 m for h = 1.8 m.

## Sources

The full references are in `constraints.SOURCES`:

- R1: NASA Perseverance components
- R2: Rankin 2023
- R3: PIA07999
- R4: Morgan 2018
- R5: Golombek & Rapp 1997
- R7: LRV (SP-4214 / TM X-64623)
- R9: MIT 16.89
- R10: NASA SEV
- R11: EVA-EXP-0075
- R12: ICES-2025-294
- R13: ICES 2026 Mars suit
- R15: Pandolf 1977
- R18: Richmond 2015
- R20: NASA NESC space weather
- R24: Mars Climate Database v6.2
- R27: Minetti 2002
- R28: Tobler 1993
- R29: Horn 1981
- R30: Riley 1999
- R31: Archinal et al. 2018, IAU WGCCRE 2015 report (Mars sphere R = 3396.19 km, used for `MARS_RADIUS_M`)
- R32: NASA NSSDCA Mars Fact Sheet (surface gravity 3.71 m/s²; `G_MARS` uses the common value 3.721 m/s²)

## TEAM ASSUMPTIONS

None of these are taken from a reference.

1. **Gravity scaling.** Pandolf's metabolic cost is scaled linearly by `G_REL = g_Mars/g_Earth ≈ 0.379`.
2. **Downhill cost.** For `G < 0`, the shape of Minetti's curve replaces Pandolf's grade term, because Pandolf underestimates the cost of walking downhill.
3. **Tobler on Mars.** Tobler's terrestrial hiking function is used on Mars. Only its scale is changed, so that it gives the nominal suited speed on flat ground.
4. **Suit overhead.** `ΔM_suit` is a constant overhead, calibrated on the 498 W walkback figure from [R13].
5. **`ETA_TERRAIN`.** The terrain coefficients {rock 1.0, delta 1.1, regolith 1.1, rocky 1.3, sand 1.5} are our values, derived from [R18].
6. **Step down.** `max_step_down_m = 0.225` m, which is 75 % of the step-up limit.
7. **Agent dimensions.** Astronaut body mass 80 kg and eye height 1.8 m; rover widths of 2.7 m (robotic) and 3.0 m (crewed).
8. **Thermal inertia.** `min_thermal_inertia = 100` (SI), our threshold for loose soil, based on [R4].
9. **Smoothing window.** `smooth_mean` uses partial windows near borders and nodata holes. This slightly biases `edge_steps` on ramps in those places.
10. **Degree conversion.** Grids in degrees are converted to metres with a local sphere of radius `MARS_RADIUS_M`, evaluated at the central latitude.
11. **Rock height.** Rock height is 0.5 × diameter (`h_over_d`), and the integration stops at `D_max = 10 m`.
12. **Rover rock penalty.** Rover travel time is multiplied by `(1 + p_rock)`, a penalty for detours around rocks.
13. **Contingency return.** `degraded_return_factor = 1.5`: in a contingency the walk back is 50 % slower.
14. **Daylight threshold.** An EVA needs a horizontal solar flux of at least 100 W/m².
15. **Dust storm threshold.** A dust optical depth τ ≥ 2 counts as an active dust storm, which is a NO-GO.
16. **Viewshed heights.** `observer_h = 2 m` for the lander antenna and `target_h = 1.8 m` for the astronaut's helmet. Line of sight is geometric only, with no radio diffraction.
