"""Demo "street view": carica le foto di Perseverance, le raggruppa in stop e mostra
quali stop incontra un percorso finto in linea retta fra due stop reali.

Uso:
    python scripts/demo_photo_stops.py path/to/foto_georeferenziate.csv [--radius 40] [--step-m 5]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# Permette di lanciare lo script direttamente (python scripts/demo_photo_stops.py ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.basic import Coordinate  # noqa: E402
from src.landmarks.photo_stops import (  # noqa: E402
    distance_m,
    find_photo_stops,
    group_into_stops,
    load_photos,
    turn_instruction,
)


def pick_endpoints(stops, min_sep_m=200.0):
    """Primo stop e il primo stop successivo ad almeno min_sep_m (fallback: il più lontano)."""
    a = stops[0]
    for s in stops[1:]:
        if distance_m(a.coo, s.coo) >= min_sep_m:
            return a, s
    return a, max(stops[1:], key=lambda s: distance_m(a.coo, s.coo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="CSV delle foto georeferenziate")
    ap.add_argument("--radius", type=float, default=40.0, help="raggio di ricerca in metri (default 40)")
    ap.add_argument("--step-m", type=float, default=5.0,
                    help="spaziatura dei punti del percorso finto in metri (default 5)")
    args = ap.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    photos = load_photos(args.csv_path)
    stops = group_into_stops(photos)
    print(f"Righe nel CSV:     {n_rows}")
    print(f"Foto tenute:       {len(photos)} (scartate {n_rows - len(photos)})")
    print(f"Stop fotografici:  {len(stops)}")
    if len(stops) < 2:
        print("Servono almeno 2 stop per costruire il percorso finto.")
        return

    if args.step_m <= 0:
        ap.error("--step-m deve essere > 0")
    a, b = pick_endpoints(stops)
    # un punto ogni step_m metri (al più): nessuno stop vicino alla linea cade fra due punti
    n_points = max(2, int(math.ceil(distance_m(a.coo, b.coo) / args.step_m)) + 1)
    t = np.linspace(0.0, 1.0, n_points)
    path = [Coordinate(float(a.coo.lat + k * (b.coo.lat - a.coo.lat)),
                       float(a.coo.lon + k * (b.coo.lon - a.coo.lon))) for k in t]
    print(f"\nPercorso finto: ({a.coo.lat:.6f}, {a.coo.lon:.6f}) -> ({b.coo.lat:.6f}, {b.coo.lon:.6f}), "
          f"{distance_m(a.coo, b.coo):.0f} m, {len(path)} punti (passo <= {args.step_m:g} m), "
          f"raggio {args.radius:.0f} m")

    events = find_photo_stops(path, stops, radius_m=args.radius)
    print(f"Eventi trovati: {len(events)}\n")
    for ev in events:
        ph = ev.best_photo
        print(f"[punto {ev.path_index:3d}] stop a {ev.distance_m:5.1f} m, "
              f"marcia {ev.travel_heading_deg:5.1f}°, {len(ev.stop.photos)} foto")
        print(f"    guarda {turn_instruction(ev.rel_bearing_deg)}  "
              f"(foto {ph.heading_deg:.0f}°, relativo {ev.rel_bearing_deg:+.0f}°)")
        print(f"    {ph.file}  sol {ph.sol}  {ph.camera}")
        print(f"    {ph.url}")


if __name__ == "__main__":
    main()
