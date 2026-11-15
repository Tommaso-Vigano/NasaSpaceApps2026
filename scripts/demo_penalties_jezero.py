"""Catena completa delle penalità su dati veri di Jezero, dal DEM al percorso e alla mappa HTML.

    DEM HiRISE + NOAH-H + THEMIS --make_slice--> griglia comune a --res-m
    + ModelMapsAdapter (gradino, distanza, tempi, visibilità) + McdMapsAdapter (polvere, luce, vento)
    -> FixedPenaltyManager + register_default_penalties -> build_graph -> shortest_path
    -> CSV lat,lon -> scripts/demo_map.py (mappa HTML)

Uso (dalla radice del repo, con il venv attivo):
    python scripts/demo_penalties_jezero.py [--profile human|rover] [--res-m 20]
        [--dest LON LAT] [--mcd diurnal_Ls270_DUSTSTORM.csv] [--eva-start-h 9] [--eva-duration-h 6]
        [--data-dir ...] [--dem ...] [--out ./out_demo] [--no-map]

Cartella dati: --data-dir, altrimenti la variabile JEZERO_DATA_DIR (ambiente o file .env alla
radice del repo, ignorato da git), altrimenti ../jezero_data accanto al repo (scripts/_data_dir.py).

I percorsi relativi (--dem, --mcd) si cercano così come sono, poi dentro --data-dir, poi (solo
per --mcd) in --data-dir/06_polvere_atmosfera/MCD. Lo script non dipende dalle dimensioni del
DEM: il riquadro è letto dai bordi del DEM e la griglia dalla risoluzione scelta.

Exit code: 0 percorso trovato, 2 NO-GO (meteo) o nessun percorso, 1 errore nei dati.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.penalty.penalty_config import register_default_penalties  # noqa: E402
from src.app.penalty.penalty_manager_fixed import FixedPenaltyManager  # noqa: E402
from src.basic import Coordinate  # noqa: E402
from src.constants import (  # noqa: E402
    DEM_REL_PATH,
    DEMO_EVA_DURATION_H,
    DEMO_EVA_START_H,
    DEMO_MAP_SCRIPT,
    DEMO_OUT_DIR,
    DEMO_RES_M,
    DEST_MAX_DIST_M,
    DUST_TAU_MAX,
    MARS_RADIUS_M,
    MCD_DIR_REL_PATH,
    MCD_REL_PATH,
    NOAH_REL_PATH,
    OFFSETS,
    PHOTOS_REL_PATH,
    SOLAR_FLUX_MIN_W_M2,
    THEMIS_REL_PATH,
    UNIFORM_DATATYPE_NAMES,
    WAYPOINTS_REL_PATH,
)
from src.data.adapters.implementations.map_adapter import MapAdapter  # noqa: E402
from src.data.adapters.implementations.mcd_maps_adapter import McdMapsAdapter  # noqa: E402
from src.data.adapters.implementations.model_maps_adapter import ModelMapsAdapter  # noqa: E402
from src.data.adapters.implementations.terrain_class_adapter import TerrainClassAdapter  # noqa: E402
from src.data.adapters.implementations.thermal_inertia_adapter import (  # noqa: E402
    ThermalInertiaAdapter)
from src.data.data_model import WorldModel  # noqa: E402
from src.data.types import Datatype  # noqa: E402
from src.landmarks.photo_stops import distance_m, group_into_stops, load_photos  # noqa: E402
from src.models.constraints import Astronaut, RoboticRover  # noqa: E402
from src.models.rover import rover_edge_time  # noqa: E402
from src.models.terrain import from_map, neighbor_stack  # noqa: E402
from src.routing.profiles import HumanProfile, RoverProfile  # noqa: E402

from _data_dir import resolve_data_dir  # noqa: E402  (scripts/_data_dir.py)

class Phases:
    """Cronometro delle fasi."""

    def __init__(self):
        self.times = []

    def run(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.times.append((name, time.perf_counter() - t0))
        print(f"  [{self.times[-1][1]:6.2f} s] {name}")
        return out

    def report(self):
        print("\nTempi delle fasi:")
        for name, t in self.times:
            print(f"  {name:45s} {t:7.2f} s")
        print(f"  {'totale':45s} {sum(t for _, t in self.times):7.2f} s")


def resolve(path_str, data_dir, *fallback_dirs):
    p = Path(path_str).expanduser()
    for cand in (p, data_dir / p, *(data_dir / d / p.name for d in fallback_dirs)):
        if cand.exists():
            return cand
    raise SystemExit(f"File non trovato: {path_str} (cercato anche in {data_dir})")


def lander_from_waypoints(path):
    """(lon, lat, sol) del waypoint con il sol più basso (primo in ordine di file a parità)."""
    with open(path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    best = min(features, key=lambda ft: ft["properties"]["sol"])
    p = best["properties"]
    return float(p["lon"]), float(p["lat"]), int(p["sol"])


def outer_box(m):
    """Bordi esterni (lon, lat) di una Map: i suoi angoli sono centri di cella."""
    gx, gy = m.granularity
    return ((m.top_left[0] - gx / 2, m.top_left[1] + gy / 2),
            (m.bottom_right[0] + gx / 2, m.bottom_right[1] - gy / 2))


def nogo_reasons(manager, profile, world, eva):
    """Layer uniformi con costo inf ovunque -> motivi di NO-GO (prima di build_graph)."""
    uniform = {getattr(Datatype, n) for n in UNIFORM_DATATYPE_NAMES if hasattr(Datatype, n)}
    reasons = []
    for m in world:
        dt = m.value_type
        if dt not in uniform or dt in manager.ignored_datatypes:
            continue
        values = np.asarray(m.values, dtype=np.float64)
        cost = np.asarray(manager.select_cost_function(profile, dt).compute(values))
        if not np.all(np.isinf(cost)):
            continue
        v = float(np.nanmax(values))
        if dt == Datatype.DUST_TAU:
            reasons.append(f"tempesta di polvere: tau {v:.1f} >= {DUST_TAU_MAX}")
        elif dt == Datatype.SOLAR_FLUX:
            reasons.append(f"luce insufficiente: flusso minimo {v:.0f} W/m2 < "
                           f"{SOLAR_FLUX_MIN_W_M2:.0f} W/m2 nella finestra EVA "
                           f"{eva[0]:.2f}-{eva[1]:.2f} h")
        elif dt == Datatype.WIND_SPEED:
            reasons.append(f"vento: {v:.1f} m/s oltre la soglia")
        else:
            reasons.append(f"{dt.value}: {v:.1f} fuori dalle soglie")
    return reasons


def print_penalty_stats(manager, profile, world, elevation):
    """Per ogni Datatype: frazione di celle inf e media dei costi finiti."""
    print("\nPenalità per Datatype (celle equivalenti):")
    print(f"  {'Datatype':22s} {'celle inf':>10s} {'media finiti':>13s}")
    for m in manager._construct_cost_pipeline(profile, world).maps:
        v = np.asarray(m.values)
        fin = v[np.isfinite(v)]
        print(f"  {m.value_type.name:22s} {np.isinf(v).mean():10.1%} "
              f"{(fin.mean() if fin.size else float('nan')):13.3f}")
    ignored = sorted({m.value_type.name for m in world if m.value_type in manager.ignored_datatypes})
    nan_elev = np.isnan(np.asarray(elevation.values)).mean()
    slopes = manager.slope_costs(profile, elevation)
    valid = ~np.isnan(slopes)
    fin = slopes[np.isfinite(slopes)]
    print(f"  {'ELEVATION (quota NaN)':22s} {nan_elev:10.1%} {'':>13s}")
    print(f"  {'SLOPE (archi)':22s} {np.isinf(slopes[valid]).mean():10.1%} "
          f"{(fin.mean() if fin.size else float('nan')):13.3f}")
    if ignored:
        print(f"  ignorati: {', '.join(ignored)}")


def choose_destination(graph, start_node, lander_ll, photos_csv):
    """Stop fotografico più lontano dal lander entro DEST_MAX_DIST_M e raggiungibile."""
    import networkx as nx
    stops = group_into_stops(load_photos(photos_csv))
    lander = Coordinate(lander_ll[1], lander_ll[0])
    reach = nx.descendants(graph.graph, start_node) | {start_node}
    cands, out_grid, unreachable = [], 0, 0
    for s in stops:
        d = distance_m(lander, s.coo)
        if d > DEST_MAX_DIST_M:
            continue
        x = int(round((s.coo.lon - graph.top_left_lon) / graph.step_length))
        y = int(round((graph.top_left_lat - s.coo.lat) / graph.step_length))
        if not (0 <= x < graph.cols and 0 <= y < graph.rows):
            out_grid += 1
            continue
        if (x, y) not in reach:
            unreachable += 1
            continue
        cands.append((d, s))
    print(f"\nDestinazione automatica: {len(stops)} stop fotografici, "
          f"{len(cands) + out_grid + unreachable} entro {DEST_MAX_DIST_M:.0f} m dal lander "
          f"(raggio di rientro in 30 min); {out_grid} fuori griglia, {unreachable} non raggiungibili "
          f"(ostacolo o zona isolata), {len(cands)} candidati.")
    if not cands:
        return None
    d, s = max(cands, key=lambda c: c[0])
    print(f"  scelto: {s.title} ({s.description}) a {d:.0f} m dal lander, "
          f"lon {s.coo.lon:.6f} lat {s.coo.lat:.6f}: il più lontano tra i raggiungibili.")
    return s.coo.lon, s.coo.lat


def path_walk_time(path, edge_time):
    """Somma dei tempi d'arco lungo il percorso; conta gli archi che il modello vieta (inf)."""
    k_of = {(dc, dr): k for k, (dr, dc) in enumerate(OFFSETS)}
    total, forbidden = 0.0, 0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        t = edge_time[k_of[(x1 - x0, y1 - y0)], y0, x0]
        if np.isfinite(t):
            total += float(t)
        else:
            forbidden += 1
    return total, forbidden


def path_length_m(path, dx, dy):
    a = np.asarray(path, dtype=np.float64)
    d = np.diff(a, axis=0)
    return float(np.hypot(d[:, 0] * dx, d[:, 1] * dy).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="cartella jezero_data (default: JEZERO_DATA_DIR da ambiente/.env, poi ../jezero_data)")
    ap.add_argument("--dem", default=DEM_REL_PATH, help="DEM (provvisorio: ±3 km dal lander, 1 m)")
    ap.add_argument("--mcd", default=MCD_REL_PATH, help="profilo diurno MCD (CSV)")
    ap.add_argument("--eva-start-h", type=float, default=DEMO_EVA_START_H, help="ora locale di inizio EVA")
    ap.add_argument("--eva-duration-h", type=float, default=DEMO_EVA_DURATION_H, help="durata EVA in ore")
    ap.add_argument("--res-m", type=float, default=DEMO_RES_M,
                    help="risoluzione della griglia in m, intera (default 20: GridGraph è lento)")
    ap.add_argument("--profile", choices=("human", "rover"), default="human")
    ap.add_argument("--dest", type=float, nargs=2, metavar=("LON", "LAT"))
    ap.add_argument("--out", type=Path, default=Path(DEMO_OUT_DIR))
    ap.add_argument("--no-map", action="store_true", help="non lanciare demo_map.py")
    args = ap.parse_args()
    if args.res_m < 1 or abs(args.res_m - round(args.res_m)) > 1e-9:
        ap.error("--res-m deve essere un numero intero di metri >= 1 (NOAH-H è mediato a blocchi da 1 m)")

    data_dir = resolve_data_dir(args.data_dir)
    dem_path = resolve(args.dem, data_dir)
    mcd_path = resolve(args.mcd, data_dir, MCD_DIR_REL_PATH)
    eva = (args.eva_start_h, args.eva_start_h + args.eva_duration_h)
    human = args.profile == "human"
    profile = HumanProfile() if human else RoverProfile()
    tag = f"{args.profile}_{int(round(args.res_m))}m_{mcd_path.stem}"

    print("=" * 78)
    print(f"  FINESTRA EVA: {eva[0]:.2f} - {eva[1]:.2f} h ora locale (durata {args.eva_duration_h:g} h)")
    print("  >>> valori di default DA DECIDERE CON IL TEAM (--eva-start-h, --eva-duration-h) <<<")
    print("=" * 78)
    print(f"Profilo: {args.profile} ({type(profile).__name__}; soglie da src/models/constraints.py)")
    print(f"DEM: {dem_path.name}\nMCD: {mcd_path.name}")

    ph = Phases()
    print("\nFasi:")
    dem = ph.run("lettura DEM (MapAdapter)", MapAdapter(Datatype.ELEVATION).adapt, dem_path)
    box_tl, box_br = outer_box(dem)
    eta_src = ph.run("NOAH-H sul riquadro del DEM (TerrainClassAdapter)",
                     TerrainClassAdapter(box_tl, box_br, resolution_m=round(args.res_m)).adapt,
                     resolve(NOAH_REL_PATH, data_dir))
    ti_src = ph.run("THEMIS (ThermalInertiaAdapter)", ThermalInertiaAdapter().adapt,
                    resolve(THEMIS_REL_PATH, data_dir))

    res_deg = math.degrees(args.res_m / MARS_RADIUS_M)
    world_model = WorldModel([dem, eta_src, ti_src])
    sl = ph.run("make_slice", world_model.make_slice, dem.top_left, dem.bottom_right, res_deg)
    by_type = {m.value_type: m for m in sl}
    elevation = by_type[Datatype.ELEVATION]
    eta = by_type.get(Datatype.TERRAIN_ETA)
    _, dx, dy = from_map(elevation)
    rows, cols = elevation.values.shape
    print(f"\nDEM: {dem.values.shape[0]} x {dem.values.shape[1]} celle, "
          f"granularity {dem.granularity[0]:.3e} deg; riquadro lon {box_tl[0]:.5f}..{box_br[0]:.5f}, "
          f"lat {box_br[1]:.5f}..{box_tl[1]:.5f}")
    print(f"Griglia: {rows} x {cols} celle a {args.res_m:g} m ({res_deg:.3e} deg; "
          f"dx {dx:.2f} m, dy {dy:.2f} m), {rows * cols} celle")
    del world_model

    lon, lat, sol = lander_from_waypoints(resolve(WAYPOINTS_REL_PATH, data_dir))
    print(f"Lander: waypoint del sol {sol} (il più basso), lon {lon:.6f} lat {lat:.6f}")

    try:
        mcd_maps = ph.run("McdMapsAdapter",
                          McdMapsAdapter(elevation, args.eva_start_h, args.eva_duration_h).adapt,
                          mcd_path)
    except ValueError as e:
        print(f"\nERRORE nel profilo MCD: {e}")
        return 1
    for m in mcd_maps:
        meta = json.loads(m.metadata)
        print(f"  MCD {m.value_type.name}: {meta['value']:.3f} {meta['value_unit']} ({meta['statistic']})")

    manager = register_default_penalties(FixedPenaltyManager())
    # NO-GO meteo prima dei modelli e di build_graph: i layer uniformi bastano a deciderlo
    reasons = nogo_reasons(manager, profile, list(sl) + mcd_maps, eva)
    if reasons:
        print_penalty_stats(manager, profile, list(sl) + mcd_maps, elevation)
        print("\n" + "!" * 78)
        print("  NO-GO: nessun percorso, il grafo non viene costruito.")
        for r in reasons:
            print(f"   - {r}")
        print("!" * 78)
        ph.report()
        return 2

    agent = Astronaut() if human else RoboticRover()
    try:
        # step_dem: i gradini si misurano sul DEM a 1 m (a 20 m edge_steps misura le ondulazioni)
        model_adapter = ModelMapsAdapter((lon, lat), agent=agent, eta_map=eta, step_dem=dem)
        model_maps = ph.run("ModelMapsAdapter (gradino, distanza, tempi, visibilità)",
                            model_adapter.adapt, elevation)
        del dem  # il DEM a 1 m non serve più
    except ValueError as e:
        print(f"\nERRORE nei dati del lander: {e}")
        return 1

    world = list(sl) + model_maps + mcd_maps
    print_penalty_stats(manager, profile, world, elevation)

    graph = ph.run("build_graph", manager.build_graph, profile, world)
    print(f"\nGrafo: {graph.graph.number_of_nodes()} nodi su {rows * cols} celle, "
          f"{graph.graph.number_of_edges()} archi ({graph.forbidden_arcs} vietati dalla pendenza, "
          f"{graph.removed_arcs} tolti)")

    try:
        start_node = graph._to_node(lon, lat)
    except ValueError as e:
        print(f"\nNessun percorso: la cella del lander è un ostacolo per {args.profile} ({e}).")
        ph.report()
        return 2

    if args.dest:
        dest = tuple(args.dest)
        print(f"\nDestinazione da --dest: lon {dest[0]:.6f} lat {dest[1]:.6f}")
    else:
        dest = ph.run("scelta destinazione (stop fotografici)", choose_destination, graph,
                      start_node, (lon, lat), resolve(PHOTOS_REL_PATH, data_dir))
        if dest is None:
            print("Nessuno stop fotografico raggiungibile entro il raggio: usa --dest LON LAT.")
            ph.report()
            return 2

    try:
        path, cost = ph.run("shortest_path", graph.shortest_path, (lon, lat), dest)
    except ValueError as e:
        print(f"\nNessun percorso: partenza o arrivo non utilizzabili ({e}).")
        ph.report()
        return 2
    if not path:
        print("\nNessun percorso: arrivo non raggiungibile dalla partenza (ostacoli in mezzo).")
        ph.report()
        return 2

    length = path_length_m(path, dx, dy)
    z = np.asarray(elevation.values, dtype=np.float64)
    if human:
        edge_time = model_adapter.last_edge_times
    else:
        ti = by_type.get(Datatype.THERMAL_INERTIA)
        edge_time = rover_edge_time(z, dx, dy, agent, ti=None if ti is None else ti.values)
        step_cell = {m.value_type: m.values for m in model_maps}[Datatype.STEP_HEIGHT]
        bad = ~(step_cell < agent.max_step_m)  # gradini dal DEM a 1 m, come la penalità
        edge_time = np.where(bad[None] | (neighbor_stack(bad.astype(float)) != 0), np.inf, edge_time)
    walk_s, model_forbidden = path_walk_time(path, edge_time)
    print(f"\nPERCORSO TROVATO ({args.profile}):")
    print(f"  celle: {len(path)}  lunghezza: {length:.0f} m  costo: {cost:.1f} celle equivalenti")
    print(f"  tempo di {'cammino' if human else 'guida'} stimato: {walk_s / 60:.1f} min "
          f"({'edge_costs' if human else 'rover_edge_time'})"
          + (f"; ATTENZIONE: {model_forbidden} archi del percorso vietati dal modello dei tempi "
             "(non sommati)" if model_forbidden else ""))
    if human:
        mm = {m.value_type: m.values for m in model_maps}
        x, y = path[-1]
        print(f"  all'arrivo: distanza dal lander {mm[Datatype.LANDER_DISTANCE][y, x]:.0f} m, "
              f"rientro minimo {mm[Datatype.RETURN_TIME][y, x] / 60:.1f} min, "
              f"EVA andata+ritorno {mm[Datatype.EVA_TIME][y, x] / 60:.1f} min")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"percorso_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon"])
        for plon, plat in graph.convert_idx_to_polar(path):
            w.writerow([f"{plat:.8f}", f"{plon:.8f}"])
    print(f"  CSV: {csv_path.resolve()}")

    if not args.no_map:
        html = args.out / f"mappa_{tag}.html"
        cmd = [sys.executable, str(DEMO_MAP_SCRIPT), str(resolve(PHOTOS_REL_PATH, data_dir)),
               "--data-dir", str(data_dir), "--out", str(html), "--path-csv", str(csv_path)]
        t0 = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, text=True)
        ph.times.append(("demo_map.py (mappa HTML)", time.perf_counter() - t0))
        if done.returncode == 0:
            print(f"  mappa HTML: {html.resolve()}")
        else:
            print(f"  demo_map.py fallito (exit {done.returncode}):\n{done.stderr[-2000:]}")
    ph.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
