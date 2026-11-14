"""Photo stops: posizioni da cui Perseverance ha scattato foto, e ricerca di quelle
vicine a un percorso per una funzione "street view" (mostra la foto e dice dove guardare).

Convenzioni: lat/lon in gradi planetocentrici, longitudine Est positiva; heading/bearing
in gradi da nord vero, in senso orario, nell'intervallo [0, 360).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from src.basic import Coordinate
from src.landmarks.landmark import Landmark

MARS_RADIUS_M = 3_396_190.0


@dataclass
class Photo:
    """Una singola foto georeferenziata del rover."""
    file: str
    sol: int
    camera: str
    lat: float
    lon: float
    elev_m: float
    heading_deg: float  # direzione della foto, da nord vero, orario
    el_deg: float       # elevazione del mast (negativa = verso il suolo)
    url: str


@dataclass
class PhotoStop(Landmark):
    """Tutte le foto scattate dalla stessa posizione. `coo` è il centro dello stop."""
    photos: list[Photo] = field(default_factory=list)


@dataclass
class StopEvent:
    """Uno stop fotografico incontrato lungo il percorso."""
    path_index: int            # punto del percorso più vicino allo stop
    stop: PhotoStop
    distance_m: float          # distanza fra path[path_index] e lo stop
    travel_heading_deg: float  # direzione di marcia in path_index (smussata sulla finestra)
    best_photo: Photo          # foto più allineata con la direzione di marcia
    rel_bearing_deg: float     # direzione della foto rispetto alla marcia, (-180, 180]


# ---------------------------------------------------------------------------
# Geometria (versioni numpy con broadcasting + wrapper scalari)
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    """Distanza ortodromica in metri sulla sfera di Marte. Accetta array (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * MARS_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _bearing(lat1, lon1, lat2, lon2):
    """Azimut iniziale da (lat1, lon1) a (lat2, lon2), in gradi [0, 360). Accetta array."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2) - np.radians(lon1)
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return np.degrees(np.arctan2(x, y)) % 360.0


def distance_m(a: Coordinate, b: Coordinate) -> float:
    """Distanza haversine in metri fra due coordinate su Marte."""
    return float(_haversine(a.lat, a.lon, b.lat, b.lon))


def bearing_deg(a: Coordinate, b: Coordinate) -> float:
    """Azimut iniziale da a verso b, in gradi da nord vero, orario, in [0, 360)."""
    return float(_bearing(a.lat, a.lon, b.lat, b.lon))


def relative_bearing(photo_heading: float, travel_heading: float) -> float:
    """Direzione della foto rispetto alla direzione di marcia, in (-180, 180].

    0 = davanti, +90 = destra, -90 = sinistra, 180 = dietro.
    """
    r = (photo_heading - travel_heading + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else float(r)


def turn_instruction(rel: float) -> str:
    """Istruzione in italiano per guardare nella direzione `rel` (vedi relative_bearing)."""
    a = abs(rel)
    if a <= 20:
        return "davanti"
    if a >= 160:
        return "dietro"
    side = "a destra" if rel > 0 else "a sinistra"
    if a >= 120:
        side = "dietro " + side
    return f"{side} (~{int(round(a / 10.0) * 10)}°)"


# ---------------------------------------------------------------------------
# Coni di vista
# ---------------------------------------------------------------------------

# Campo di vista orizzontale (gradi) usato per disegnare i coni delle foto
CAMERA_FOV_DEG = {"NAVCAM_LEFT": 45.0, "MCZ_LEFT": 15.0}
DEFAULT_FOV_DEG = 45.0


def destination(lat, lon, bearing, dist_m):
    """Punto raggiunto partendo da (lat, lon) lungo `bearing` (gradi da N, orario) per
    `dist_m` metri sulla sfera di Marte. Accetta array (broadcasting). Restituisce (lat, lon)."""
    p1, l1 = np.radians(lat), np.radians(lon)
    th = np.radians(bearing)
    d = np.asarray(dist_m, dtype=float) / MARS_RADIUS_M
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def view_cone(coo: Coordinate, heading_deg: float, fov_deg: float,
              length_m: float = 150.0, n_arc: int = 8) -> list[Coordinate]:
    """Poligono del cono di vista: il vertice in `coo`, poi n_arc+1 punti sull'arco a
    distanza length_m, da heading-fov/2 (sinistra) a heading+fov/2 (destra)."""
    bearings = heading_deg + np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_arc + 1)
    lat, lon = destination(coo.lat, coo.lon, bearings, length_m)
    return [Coordinate(coo.lat, coo.lon)] + [Coordinate(float(a), float(b)) for a, b in zip(lat, lon)]


# ---------------------------------------------------------------------------
# Caricamento e raggruppamento
# ---------------------------------------------------------------------------

def load_photos(csv_path, min_elevation_deg: float = -20.0) -> list[Photo]:
    """Legge il CSV delle foto georeferenziate.

    Scarta le foto con mastEl_deg <= min_elevation_deg (guardano il suolo) e le righe
    con lat/lon/direzione/elevazione mancanti o non numeriche.
    """
    photos: list[Photo] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                heading = float(row["view_dir_deg_from_N"])
                el = float(row["mastEl_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite([lat, lon, heading, el])):
                continue
            if el <= min_elevation_deg:
                continue
            try:
                elev_m = float(row.get("elev_m") or "nan")
            except ValueError:
                elev_m = float("nan")
            try:
                sol = int(float(row.get("sol") or -1))
            except ValueError:
                sol = -1
            photos.append(Photo(
                file=row.get("file", ""),
                sol=sol,
                camera=row.get("camera", ""),
                lat=lat,
                lon=lon,
                elev_m=elev_m,
                heading_deg=heading % 360.0,
                el_deg=el,
                url=row.get("full_res_url", ""),
            ))
    return photos


def group_into_stops(photos: list[Photo], merge_radius_m: float = 5.0) -> list[PhotoStop]:
    """Raggruppa le foto scattate dalla stessa posizione (entro merge_radius_m).

    Clustering greedy deterministico: le foto sono visitate in ordine (sol, file); ognuna
    va nello stop più vicino se entro il raggio (il centro diventa la media delle foto),
    altrimenti apre un nuovo stop.
    """
    stops: list[PhotoStop] = []
    centers_lat: list[float] = []
    centers_lon: list[float] = []
    for p in sorted(photos, key=lambda p: (p.sol, p.file)):
        if stops:
            d = _haversine(p.lat, p.lon, np.asarray(centers_lat), np.asarray(centers_lon))
            j = int(np.argmin(d))
            if d[j] <= merge_radius_m:
                stop = stops[j]
                stop.photos.append(p)
                centers_lat[j] = float(np.mean([q.lat for q in stop.photos]))
                centers_lon[j] = float(np.mean([q.lon for q in stop.photos]))
                stop.coo = Coordinate(centers_lat[j], centers_lon[j])
                continue
        stops.append(PhotoStop(coo=Coordinate(p.lat, p.lon), photos=[p]))
        centers_lat.append(p.lat)
        centers_lon.append(p.lon)
    return stops


# ---------------------------------------------------------------------------
# Ricerca lungo il percorso
# ---------------------------------------------------------------------------

def _travel_headings(p_lat: np.ndarray, p_lon: np.ndarray, window_m: float) -> np.ndarray:
    """Direzione di marcia (gradi) per ogni punto del percorso, smussata su una finestra.

    Al punto i la direzione è il bearing dal punto che si trova ~window_m/2 prima al punto
    ~window_m/2 dopo, misurando la distanza cumulativa lungo il percorso; ai bordi la finestra
    è troncata. Serve perché il pathfinding su griglia senza diagonali produce scalini
    N/E/N/E... su cui il bearing fra punti consecutivi oscilla fra 0° e 90°.

    Se i due estremi della finestra coincidono si usa il bearing del segmento non degenere
    più vicino (a parità, quello in avanti). Se tutti i punti coincidono, la direzione non è
    definita e si restituisce 0 (nord).
    """
    n = len(p_lat)
    seg_d = _haversine(p_lat[:-1], p_lon[:-1], p_lat[1:], p_lon[1:])  # (N-1,)
    seg_b = _bearing(p_lat[:-1], p_lon[:-1], p_lat[1:], p_lon[1:])    # (N-1,)
    cum = np.concatenate(([0.0], np.cumsum(seg_d)))                    # (N,)

    half = window_m / 2.0
    lo = np.searchsorted(cum, cum - half, side="left")       # primo punto a >= s-half
    hi = np.searchsorted(cum, cum + half, side="right") - 1  # ultimo punto a <= s+half
    hi = np.clip(hi, 0, n - 1)

    heading = _bearing(p_lat[lo], p_lon[lo], p_lat[hi], p_lon[hi])
    span = _haversine(p_lat[lo], p_lon[lo], p_lat[hi], p_lon[hi])

    degenerate = np.flatnonzero(span == 0.0)
    if degenerate.size:
        nz = np.flatnonzero(seg_d > 0.0)  # segmenti di lunghezza non nulla
        for i in degenerate:
            if nz.size == 0:
                heading[i] = 0.0
                continue
            d = np.abs(nz + 0.5 - i)       # distanza dal centro del segmento k (fra k e k+1)
            k = nz[d == d.min()].max()     # a parità, il segmento in avanti
            heading[i] = seg_b[k]
    return heading


def find_photo_stops(path: list[Coordinate], stops: list[PhotoStop],
                     radius_m: float = 40.0,
                     heading_window_m: float = 20.0) -> list[StopEvent]:
    """Trova gli stop fotografici entro radius_m dal percorso.

    Ogni stop compare al massimo una volta, al punto del percorso più vicino.
    Gli eventi sono ordinati per path_index (a parità, per distanza).
    La direzione di marcia è smussata su heading_window_m metri di percorso
    (vedi _travel_headings). Con meno di 2 punti la direzione non è definita: [].
    """
    stops = [s for s in stops if s.photos]
    if len(path) < 2 or not stops:
        return []

    p_lat = np.array([c.lat for c in path], dtype=float)
    p_lon = np.array([c.lon for c in path], dtype=float)
    s_lat = np.array([s.coo.lat for s in stops], dtype=float)
    s_lon = np.array([s.coo.lon for s in stops], dtype=float)

    # Distanze percorso × stop, (N, M)
    dist = _haversine(p_lat[:, None], p_lon[:, None], s_lat[None, :], s_lon[None, :])
    nearest = dist.argmin(axis=0)                       # (M,)
    dmin = dist[nearest, np.arange(len(stops))]          # (M,)

    travel = _travel_headings(p_lat, p_lon, heading_window_m)  # (N,)

    events: list[StopEvent] = []
    for j in np.flatnonzero(dmin <= radius_m):
        stop = stops[j]
        i = int(nearest[j])
        th = float(travel[i])
        rels = [relative_bearing(ph.heading_deg, th) for ph in stop.photos]
        k = int(np.argmin(np.abs(rels)))
        events.append(StopEvent(
            path_index=i,
            stop=stop,
            distance_m=float(dmin[j]),
            travel_heading_deg=th,
            best_photo=stop.photos[k],
            rel_bearing_deg=rels[k],
        ))
    events.sort(key=lambda e: (e.path_index, e.distance_m))
    return events
