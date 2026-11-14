"""Mappa HTML "street view" su Marte (Leaflet via folium) per la demo.

Mostra l'immagine orbitale CTX di Jezero, un percorso, gli stop fotografici di Perseverance,
i coni di vista delle foto e uno slider che fa camminare un "astronauta" lungo il percorso:
quando entra nel raggio di uno stop, il pannello laterale mostra la foto e dove guardare.

Dipendenze extra (solo per questo script): pip install -r scripts/requirements-demo.txt

Uso:
    python scripts/demo_map.py <foto_georeferenziate.csv> --data-dir <jezero_data> --out <cartella>/mappa.html
        [--path-csv percorso.csv] [--demo-stops 5] [--step-m 5] [--radius 40] [--max-px 4000]

Accanto all'HTML vengono scritti <nome>_ctx.png (sfondo) e img/ (le foto mostrate): la cartella
si può zippare e aprire altrove (Leaflet viene comunque caricato da CDN, serve internet).

Coordinate: gradi planetocentrici, longitudine Est positiva. La mappa usa L.CRS.Simple con lat/lon
in gradi come coordinate piane (plate carrée), cioè la stessa geometria dell'immagine Mars Trek
equirettangolare: nessun CRS terrestre.
"""
import argparse
import csv
import html
import json
import math
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

# Permette di lanciare lo script direttamente (python scripts/demo_map.py ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.basic import Coordinate  # noqa: E402
from src.landmarks.photo_stops import (  # noqa: E402
    CAMERA_FOV_DEG,
    DEFAULT_FOV_DEG,
    MARS_RADIUS_M,
    bearing_deg,
    destination,
    distance_m,
    find_photo_stops,
    group_into_stops,
    load_photos,
    turn_instruction,
    view_cone,
)

CTX_REL_PATH = Path("02_immagini") / "jezero_CTX_ortho_6m.tif"
CONE_LENGTH_M = 150.0
LATERAL_OFFSETS_M = (15.0, 20.0, 25.0)   # spostamento laterale dei waypoint del percorso finto
MIN_STOP_SEP_M = 300.0
DEMO_LENGTH_RANGE_M = (2000.0, 4000.0)
CAMERA_COLOR = {"NAVCAM_LEFT": "#4fc3f7", "MCZ_LEFT": "#ffd54f"}

# Tag GeoTIFF
TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEO_DOUBLE, TAG_GEO_ASCII = 33550, 33922, 34736, 34737


# ---------------------------------------------------------------------------
# Sfondo orbitale
# ---------------------------------------------------------------------------

def load_ctx_background(tif_path: Path, out_png: Path, max_px: int):
    """Legge il GeoTIFF Mars Trek (equirettangolare in metri, sfera di Marte), lo riduce con un
    fattore intero (lato massimo <= max_px), rende trasparente il NoData (0) e lo salva in PNG.
    Restituisce (bounds [[lat_s, lon_w], [lat_n, lon_e]], info) con lon = deg(x/R), lat = deg(y/R)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(tif_path)
    tags = img.tag_v2
    ascii_params = str(tags.get(TAG_GEO_ASCII, ""))
    geo_double = tuple(tags.get(TAG_GEO_DOUBLE, ()))
    if "Equirectangular" not in ascii_params or "Mars" not in ascii_params:
        raise SystemExit(f"{tif_path.name}: atteso 'Equirectangular Mars ...', trovato {ascii_params!r}")
    if not any(abs(v - MARS_RADIUS_M) < 1.0 for v in geo_double):
        raise SystemExit(f"{tif_path.name}: raggio della sfera diverso da {MARS_RADIUS_M} m: {geo_double}")

    sx, sy = tags[TAG_PIXEL_SCALE][:2]
    i0, j0, _, x0, y0, _ = tags[TAG_TIEPOINT][:6]
    x_left, y_top = x0 - i0 * sx, y0 + j0 * sy        # angolo NW (PixelIsArea)

    w, h = img.size
    f = max(1, math.ceil(max(w, h) / max_px))
    w_c, h_c = (w // f) * f, (h // f) * f              # ritaglio a multipli di f: bounds esatti
    gray = img.convert("L").crop((0, 0, w_c, h_c))
    valid = Image.fromarray(((np.asarray(gray) > 0) * 255).astype(np.uint8))
    if f > 1:
        gray, valid = gray.reduce(f), valid.reduce(f)
    alpha = valid.point(lambda v: 255 if v >= 128 else 0)
    out = Image.merge("LA", (gray, alpha))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_png, optimize=True)

    x_right, y_bottom = x_left + w_c * sx, y_top - h_c * sy
    lon_w, lon_e = math.degrees(x_left / MARS_RADIUS_M), math.degrees(x_right / MARS_RADIUS_M)
    lat_n, lat_s = math.degrees(y_top / MARS_RADIUS_M), math.degrees(y_bottom / MARS_RADIUS_M)
    info = (f"{tif_path.name}: {w}x{h} px a {sx:g} m/px -> {out.size[0]}x{out.size[1]} px "
            f"a {sx * f:g} m/px; lon {lon_w:.5f}..{lon_e:.5f}, lat {lat_s:.5f}..{lat_n:.5f}")
    return [[lat_s, lon_w], [lat_n, lon_e]], info


# ---------------------------------------------------------------------------
# Percorso
# ---------------------------------------------------------------------------

def read_path_csv(path_csv) -> list[Coordinate]:
    with open(path_csv, newline="", encoding="utf-8") as f:
        return [Coordinate(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)]


def densify(waypoints: list[Coordinate], step_m: float) -> list[Coordinate]:
    """Un punto ogni step_m metri (al più) su ogni tratto, vertici inclusi una volta sola.
    Interpolazione lineare in lat/lon: va bene per tratti di qualche km."""
    out = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        n = max(1, math.ceil(distance_m(a, b) / step_m))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append(Coordinate(float(a.lat + t * (b.lat - a.lat)), float(a.lon + t * (b.lon - a.lon))))
    return out


def pick_endpoints(stops, min_sep_m=200.0):
    """Primo stop e il primo successivo ad almeno min_sep_m (fallback: il più lontano)."""
    a = stops[0]
    for s in stops[1:]:
        if distance_m(a.coo, s.coo) >= min_sep_m:
            return a, s
    return a, max(stops[1:], key=lambda s: distance_m(a.coo, s.coo))


def choose_chain(stops, n, min_sep_m=MIN_STOP_SEP_M, length_range=DEMO_LENGTH_RANGE_M):
    """Catena greedy di n stop distanti fra loro almeno min_sep_m (tutte le coppie).
    Prova ogni stop di partenza (in ordine di sol) e prende il vicino valido più prossimo;
    restituisce la prima catena con lunghezza in length_range, altrimenti la più vicina al range."""
    m = len(stops)
    d = np.array([[distance_m(a.coo, b.coo) for b in stops] for a in stops])
    lo, hi = length_range
    best, best_err, best_len = None, math.inf, 0.0
    for start in range(m):
        chain = [start]
        while len(chain) < n:
            cands = [k for k in range(m) if k not in chain and all(d[k, c] >= min_sep_m for c in chain)]
            if not cands:
                break
            chain.append(min(cands, key=lambda k: d[chain[-1], k]))
        if len(chain) < n:
            continue
        length = float(sum(d[a, b] for a, b in zip(chain, chain[1:])))
        err = 0.0 if lo <= length <= hi else min(abs(length - lo), abs(length - hi))
        if err < best_err:
            best, best_err, best_len = chain, err, length
        if err == 0.0:
            break
    if best is None:
        raise SystemExit(f"Impossibile scegliere {n} stop distanti almeno {min_sep_m:.0f} m: prova --demo-stops minore")
    return [stops[k] for k in best], best_len, best_err


def offset_waypoints(points: list[Coordinate]) -> list[Coordinate]:
    """Sposta ogni waypoint di 15/20/25 m (a rotazione), alternando destra/sinistra,
    perpendicolarmente alla direzione locale (bisettrice dei tratti adiacenti)."""
    out = []
    last = len(points) - 1
    for i, p in enumerate(points):
        if i == 0:
            h = bearing_deg(points[0], points[1])
        elif i == last:
            h = bearing_deg(points[-2], points[-1])
        else:
            h_in = math.radians(bearing_deg(points[i - 1], p))
            h_out = math.radians(bearing_deg(p, points[i + 1]))
            h = math.degrees(math.atan2(math.sin(h_in) + math.sin(h_out), math.cos(h_in) + math.cos(h_out)))
        side = 1.0 if i % 2 == 0 else -1.0
        lat, lon = destination(p.lat, p.lon, h + 90.0 * side, LATERAL_OFFSETS_M[i % len(LATERAL_OFFSETS_M)])
        out.append(Coordinate(float(lat), float(lon)))
    return out


def build_demo_path(stops, n_stops, step_m):
    """Percorso finto: n_stops=2 -> retta fra due stop; altrimenti spezzata spostata di lato."""
    if n_stops == 2:
        a, b = pick_endpoints(stops)
        chosen, waypoints = [a, b], [a.coo, b.coo]
        desc = f"retta fra 2 stop, {distance_m(a.coo, b.coo):.0f} m"
    else:
        chosen, length, err = choose_chain(stops, n_stops)
        waypoints = offset_waypoints([s.coo for s in chosen])
        desc = f"spezzata fra {n_stops} stop (spostata 15-25 m di lato), {length:.0f} m fra gli stop"
        if err > 0:
            desc += f"  [ATTENZIONE: fuori dal range {DEMO_LENGTH_RANGE_M[0]:.0f}-{DEMO_LENGTH_RANGE_M[1]:.0f} m]"
    return densify(waypoints, step_m), chosen, desc


# ---------------------------------------------------------------------------
# Foto e contenuti HTML
# ---------------------------------------------------------------------------

def prepare_images(events, raw_dir: Path, img_dir: Path) -> dict:
    """Copia in img_dir i JPG di tutte le foto degli stop trovati; restituisce file -> src
    (relativo all'HTML, o l'URL NASA se il JPG locale non c'è). img_dir viene rigenerata."""
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    src = {}
    for ev in events:
        for ph in ev.stop.photos:
            local = raw_dir / ph.file
            if local.is_file():
                shutil.copy2(local, img_dir / ph.file)
                src[ph.file] = f"{img_dir.name}/{quote(ph.file)}"
            else:
                src[ph.file] = ph.url
    return src


def event_info(ev, src) -> dict:
    ph = ev.best_photo
    return {
        "img": src[ph.file],
        "file": ph.file,
        "sol": ph.sol,
        "camera": ph.camera,
        "dist": round(ev.distance_m, 1),
        "travel": round(ev.travel_heading_deg),
        "photo_heading": round(ph.heading_deg),
        "rel": round(ev.rel_bearing_deg),
        "instr": turn_instruction(ev.rel_bearing_deg),
        "n_photos": len(ev.stop.photos),
        "lat": ev.stop.coo.lat,
        "lon": ev.stop.coo.lon,
    }


def popup_html(info: dict, others: list[tuple[str, str]]) -> str:
    e = html.escape
    thumbs = "".join(
        f'<a href="{e(s)}" target="_blank"><img src="{e(s)}" title="{e(f)}" '
        f'style="width:60px;height:45px;object-fit:cover;margin:2px"></a>' for f, s in others[:6])
    return (
        f'<div style="font:13px sans-serif;width:300px">'
        f'<a href="{e(info["img"])}" target="_blank"><img src="{e(info["img"])}" style="width:100%"></a>'
        f'<div style="font-size:15px;font-weight:bold;margin:6px 0">Guarda {e(info["instr"])}</div>'
        f'sol {info["sol"]} · {e(info["camera"])} · {info["n_photos"]} foto<br>'
        f'distanza dal percorso: {info["dist"]:.1f} m<br>'
        f'marcia {info["travel"]}° · foto {info["photo_heading"]}° (relativo {info["rel"]:+d}°)<br>'
        f'<span style="color:#777;font-size:11px">{e(info["file"])}</span>'
        f'{"<div>" + thumbs + "</div>" if thumbs else ""}</div>'
    )


def active_event_per_point(path, events, radius_m) -> list[int]:
    """Per ogni punto del percorso, l'indice dell'evento più vicino entro radius_m (o -1)."""
    if not events:
        return [-1] * len(path)
    d = np.array([[distance_m(p, ev.stop.coo) for ev in events] for p in path])  # (N, E)
    k = d.argmin(axis=1)
    inside = d[np.arange(len(path)), k] <= radius_m
    return [int(j) if ok else -1 for j, ok in zip(k, inside)]


SLIDER_CSS = """
<style>
#sv-title{position:fixed;top:10px;left:56px;z-index:1000;background:rgba(20,20,20,.85);color:#eee;
  padding:6px 12px;border-radius:6px;font:14px sans-serif}
#sv-bar{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:1000;width:min(640px,90vw);
  background:rgba(20,20,20,.85);color:#eee;padding:8px 12px;border-radius:8px;font:13px sans-serif;
  display:flex;gap:10px;align-items:center}
#sv-bar input[type=range]{flex:1}
#sv-bar button{background:#e4572e;color:#fff;border:0;border-radius:4px;padding:4px 10px;cursor:pointer}
#sv-panel{position:fixed;top:10px;right:10px;z-index:1000;width:min(340px,80vw);max-height:calc(100vh - 100px);
  overflow:auto;background:rgba(20,20,20,.92);color:#eee;padding:10px;border-radius:8px;font:13px sans-serif;
  display:none}
#sv-panel img{width:100%;border-radius:4px}
#sv-panel .instr{font-size:20px;font-weight:bold;color:#ffcc80;margin:8px 0}
#sv-panel .meta{color:#bbb;line-height:1.5}
</style>
<div id="sv-title">Jezero · street view (demo)</div>
<div id="sv-panel"></div>
<div id="sv-bar"><button id="sv-play">▶</button><input id="sv-slider" type="range" min="0" value="0">
  <span id="sv-label"></span></div>
"""

SLIDER_JS = """
window.addEventListener("load", function () {
  var map = %(map)s;
  var D = %(data)s;
  var slider = document.getElementById("sv-slider");
  var label = document.getElementById("sv-label");
  var panel = document.getElementById("sv-panel");
  var play = document.getElementById("sv-play");
  slider.max = D.path.length - 1;
  var astro = L.circleMarker(D.path[0], {radius: 9, color: "#fff", weight: 3, fillColor: "#e4572e",
                                         fillOpacity: 1}).addTo(map).bindTooltip("Astronauta");
  var shown = -2, timer = null;
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function update(i) {
    var ll = D.path[i];
    astro.setLatLng(ll);
    if (!map.getBounds().contains(ll)) { map.panTo(ll); }
    label.textContent = (D.cum[i] / 1000).toFixed(2) + " / " + (D.cum[D.cum.length - 1] / 1000).toFixed(2) + " km";
    var e = D.active[i];
    if (e === shown) { return; }
    shown = e;
    if (e < 0) { panel.style.display = "none"; return; }
    var ev = D.events[e];
    panel.innerHTML = '<a href="' + esc(ev.img) + '" target="_blank"><img src="' + esc(ev.img) + '"></a>' +
      '<div class="instr">Guarda ' + esc(ev.instr) + '</div>' +
      '<div class="meta">sol ' + ev.sol + ' · ' + esc(ev.camera) + ' · ' + ev.n_photos + ' foto<br>' +
      'stop a ' + ev.dist.toFixed(1) + ' m dal percorso<br>' +
      'marcia ' + ev.travel + '° · foto ' + ev.photo_heading + '° (relativo ' + (ev.rel > 0 ? '+' : '') + ev.rel + '°)<br>' +
      '<span style="font-size:11px">' + esc(ev.file) + '</span></div>';
    panel.style.display = "block";
  }
  slider.addEventListener("input", function () { update(+slider.value); });
  play.addEventListener("click", function () {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶"; return; }
    if (+slider.value >= +slider.max) { slider.value = 0; }
    play.textContent = "❚❚";
    timer = setInterval(function () {
      if (+slider.value >= +slider.max) { clearInterval(timer); timer = null; play.textContent = "▶"; return; }
      slider.value = +slider.value + 1; update(+slider.value);
    }, 80);
  });
  update(0);
});
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="CSV delle foto georeferenziate (in jezero_data/08_perseverance/raw_images)")
    ap.add_argument("--data-dir", required=True, help="cartella jezero_data")
    ap.add_argument("--out", required=True, help="file HTML di output (PNG e img/ vengono scritti accanto)")
    ap.add_argument("--path-csv", help="percorso vero (colonne lat,lon); se assente si usa il percorso finto")
    ap.add_argument("--demo-stops", type=int, default=5,
                    help="stop da collegare nel percorso finto (2 = retta fra due stop; default 5)")
    ap.add_argument("--step-m", type=float, default=5.0, help="spaziatura del percorso finto in metri (default 5)")
    ap.add_argument("--radius", type=float, default=40.0, help="raggio street view in metri (default 40)")
    ap.add_argument("--max-px", type=int, default=4000, help="lato massimo dell'immagine di sfondo (default 4000)")
    args = ap.parse_args()
    if args.step_m <= 0:
        ap.error("--step-m deve essere > 0")
    if args.demo_stops < 2:
        ap.error("--demo-stops deve essere >= 2")

    try:
        import folium
    except ImportError:
        raise SystemExit("Manca folium: pip install -r scripts/requirements-demo.txt")

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_path)
    out_html = Path(args.out).resolve()
    out_dir = out_html.parent
    out_png = out_dir / f"{out_html.stem}_ctx.png"
    img_dir = out_dir / "img"

    # Dati
    photos = load_photos(csv_path)
    stops = group_into_stops(photos)
    print(f"Foto tenute: {len(photos)}  -> stop: {len(stops)}")

    bounds, bg_info = load_ctx_background(data_dir / CTX_REL_PATH, out_png, args.max_px)
    print(f"Sfondo: {bg_info}")

    if args.path_csv:
        path, chosen = read_path_csv(args.path_csv), []
        desc = f"da {args.path_csv}"
    else:
        path, chosen, desc = build_demo_path(stops, args.demo_stops, args.step_m)
    cum = np.concatenate(([0.0], np.cumsum([distance_m(a, b) for a, b in zip(path, path[1:])])))
    print(f"Percorso: {desc}; {len(path)} punti, {cum[-1]:.0f} m")
    for s in chosen:
        print(f"  waypoint: stop ({s.coo.lat:.6f}, {s.coo.lon:.6f}), {len(s.photos)} foto, sol {s.photos[0].sol}")

    events = find_photo_stops(path, stops, radius_m=args.radius)
    print(f"Eventi street view: {len(events)}")
    for ev in events:
        print(f"  [punto {ev.path_index:4d}, {cum[ev.path_index]:6.0f} m] stop a {ev.distance_m:5.1f} m, "
              f"guarda {turn_instruction(ev.rel_bearing_deg)} ({ev.best_photo.camera}, sol {ev.best_photo.sol})")

    src = prepare_images(events, csv_path.parent, img_dir)
    infos = [event_info(ev, src) for ev in events]

    # Mappa: CRS Simple = lat/lon in gradi come coordinate piane (plate carrée), nessuna mappa terrestre
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=center, crs="Simple", tiles=None, zoom_start=10, control_scale=False,
                   **{"minZoom": 8, "maxZoom": 22, "zoomSnap": 0.25})
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container{background:#1b1b1b}</style>"))

    # folium incorporerebbe il PNG in base64: costruisco l'overlay con un segnaposto e punto al file
    overlay = folium.raster_layers.ImageOverlay(np.zeros((1, 1, 4), dtype=np.uint8), bounds=bounds,
                                                name="CTX 6 m (Mars Trek)", opacity=1.0)
    overlay.url = quote(out_png.name)
    overlay.add_to(m)

    fg_path = folium.FeatureGroup(name="Percorso").add_to(m)
    folium.PolyLine([[p.lat, p.lon] for p in path], color="#ff7043", weight=4, opacity=0.9).add_to(fg_path)

    fg_all = folium.FeatureGroup(name="Tutti gli stop").add_to(m)
    for s in stops:
        folium.CircleMarker([s.coo.lat, s.coo.lon], radius=3, color="#bdbdbd", weight=1, fill=True,
                            fill_color="#9e9e9e", fill_opacity=0.9,
                            tooltip=f"{len(s.photos)} foto · sol {s.photos[0].sol}").add_to(fg_all)

    fg_cones = folium.FeatureGroup(name="Coni di vista").add_to(m)
    fg_found = folium.FeatureGroup(name="Stop street view").add_to(m)
    for ev, info in zip(events, infos):
        for ph in ev.stop.photos:
            fov = CAMERA_FOV_DEG.get(ph.camera, DEFAULT_FOV_DEG)
            cone = view_cone(ev.stop.coo, ph.heading_deg, fov, CONE_LENGTH_M)
            best = ph is ev.best_photo
            color = CAMERA_COLOR.get(ph.camera, "#ffffff")
            folium.Polygon([[c.lat, c.lon] for c in cone], color=color, weight=2 if best else 1,
                           fill=True, fill_color=color, fill_opacity=0.35 if best else 0.12,
                           tooltip=f"{ph.camera} · {ph.heading_deg:.0f}° · sol {ph.sol}").add_to(fg_cones)
        others = [(ph.file, src[ph.file]) for ph in ev.stop.photos if ph is not ev.best_photo]
        folium.CircleMarker([ev.stop.coo.lat, ev.stop.coo.lon], radius=8, color="#ffffff", weight=2,
                            fill=True, fill_color="#e53935", fill_opacity=0.95,
                            tooltip=f"Street view: guarda {info['instr']}",
                            popup=folium.Popup(popup_html(info, others), max_width=320)).add_to(fg_found)

    folium.LayerControl(collapsed=True, position="bottomright").add_to(m)
    lats, lons = [p.lat for p in path], [p.lon for p in path]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(80, 80))

    # Slider "astronauta" + pannello laterale (dati precalcolati in Python, niente distanze in JS)
    data = {
        "path": [[round(p.lat, 7), round(p.lon, 7)] for p in path],
        "cum": [round(float(c), 1) for c in cum],
        "active": active_event_per_point(path, events, args.radius),
        "events": infos,
    }
    m.get_root().html.add_child(folium.Element(SLIDER_CSS))
    m.get_root().script.add_child(folium.Element(
        SLIDER_JS % {"map": m.get_name(), "data": json.dumps(data, ensure_ascii=False)}))

    out_dir.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    n_local = sum(1 for v in src.values() if not v.startswith("http"))
    print(f"\nScritto: {out_html}")
    print(f"         {out_png} ({out_png.stat().st_size / 1e6:.1f} MB)")
    print(f"         {img_dir}/ ({n_local} JPG locali, {len(src) - n_local} da URL NASA)")


if __name__ == "__main__":
    main()
