import json
import math

import pytest

from src.basic import Coordinate
from src.landmarks.photo_stops import (
    MARS_RADIUS_M,
    Photo,
    PhotoStop,
    bearing_deg,
    distance_m,
    find_photo_stops,
    group_into_stops,
    load_photos,
    make_photo_stop,
    register_photo_stops,
    relative_bearing,
    turn_instruction,
    view_cone,
)
from src.landmarks.registry import LandmarkRegistry

# metri -> gradi (lungo un meridiano, o lungo l'equatore)
M_TO_DEG = 1.0 / (MARS_RADIUS_M * math.pi / 180.0)

CSV_HEADER = ("file,sol,camera,site,drive,date_utc,lat,lon,elev_m,rover_yaw_deg,mastAz_deg,"
              "mastEl_deg,view_dir_deg_from_N,match,full_res_url")


def _photo(name, lat, lon, heading=0.0, sol=1):
    return Photo(file=name, sol=sol, camera="NAVCAM_LEFT", lat=lat, lon=lon, elev_m=-2500.0,
                 heading_deg=heading, el_deg=0.0, url=f"https://example.org/{name}.png")


def _stop(coo, photos, title="test stop"):
    """PhotoStop costruito a mano, con tutti i campi obbligatori di Landmark."""
    return PhotoStop(coo=coo, title=title, description="", extra={"type": "photo_stop"},
                     photos=photos)


def _north_path(n=21, step_m=10.0):
    """Percorso rettilineo verso nord lungo lon=0, un punto ogni step_m metri."""
    return [Coordinate(i * step_m * M_TO_DEG, 0.0) for i in range(n)]


# 1
def test_distance_one_degree_latitude_at_equator():
    expected = 2 * math.pi * MARS_RADIUS_M / 360.0
    assert expected == pytest.approx(59_274, abs=1)
    assert distance_m(Coordinate(0, 0), Coordinate(1, 0)) == pytest.approx(expected, abs=1)


# 2
def test_bearing_cardinal_directions():
    o = Coordinate(0, 0)
    assert bearing_deg(o, Coordinate(1, 0)) == pytest.approx(0, abs=1e-6)
    assert bearing_deg(o, Coordinate(0, 1)) == pytest.approx(90, abs=1e-6)
    assert bearing_deg(o, Coordinate(-1, 0)) == pytest.approx(180, abs=1e-6)
    assert bearing_deg(o, Coordinate(0, -1)) == pytest.approx(270, abs=1e-6)


# 3
@pytest.mark.parametrize("photo, travel, expected", [
    (90, 0, 90),
    (350, 10, -20),
    (10, 350, 20),
    (180, 0, 180),
    (0, 180, 180),
    (270, 0, -90),
])
def test_relative_bearing(photo, travel, expected):
    assert relative_bearing(photo, travel) == pytest.approx(expected)


def test_turn_instruction():
    assert turn_instruction(0) == "davanti"
    assert turn_instruction(-15) == "davanti"
    assert turn_instruction(180) == "dietro"
    assert turn_instruction(78) == "a destra (~80°)"
    assert turn_instruction(-90) == "a sinistra (~90°)"
    assert turn_instruction(119) == "a destra (~120°)"
    assert turn_instruction(120) == "dietro a destra (~120°)"
    assert turn_instruction(150) == "dietro a destra (~150°)"
    assert turn_instruction(-137) == "dietro a sinistra (~140°)"
    assert turn_instruction(-159) == "dietro a sinistra (~160°)"
    assert turn_instruction(-160) == "dietro"


# 4
def test_load_photos_drops_ground_looking(tmp_path):
    rows = [
        "a.png,100,NAVCAM_LEFT,1,2,2021-05-01T00:00:00,18.44,77.45,-2560,10,20,-37,30,1,https://x/a",
        "b.png,100,MCZ_LEFT,1,2,2021-05-01T00:00:00,18.44,77.45,-2560,10,20,-89.6,30,1,https://x/b",
        "c.png,100,MCZ_LEFT,1,2,2021-05-01T00:00:00,18.44,77.45,-2560,10,20,-5,30,1,https://x/c",
    ]
    csv_path = tmp_path / "foto.csv"
    csv_path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    photos = load_photos(csv_path)
    assert [p.file for p in photos] == ["c.png"]
    p = photos[0]
    assert p.sol == 100 and p.camera == "MCZ_LEFT"
    assert p.lat == pytest.approx(18.44) and p.lon == pytest.approx(77.45)
    assert p.heading_deg == pytest.approx(30) and p.el_deg == pytest.approx(-5)
    assert p.url == "https://x/c"


# 5
def test_group_into_stops():
    photos = [_photo(f"p{i}", 0.0, 0.0, heading=90 * i) for i in range(3)]
    photos.append(_photo("far", 100 * M_TO_DEG, 0.0))
    stops = group_into_stops(photos)
    assert len(stops) == 2
    assert sorted(len(s.photos) for s in stops) == [1, 3]
    assert all(isinstance(s, PhotoStop) for s in stops)
    (big,) = [s for s in stops if len(s.photos) == 3]
    assert big.title == "Photo stop – sol 1"
    assert big.description == "3 Perseverance photos (NAVCAM_LEFT)"
    assert big.extra == {
        "type": "photo_stop", "sols": [1], "files": ["p0", "p1", "p2"],
        "cameras": ["NAVCAM_LEFT"], "best_heading_deg": pytest.approx(90.0),
    }
    for s in stops:
        json.dumps(s.extra, allow_nan=False)  # solo tipi JSON


def test_photo_stop_title_spans_sols_and_cameras():
    a = _photo("a", 0.0, 0.0, sol=7)
    b = _photo("b", 0.0, 0.0, sol=3)
    b.camera = "MCZ_LEFT"
    (stop,) = group_into_stops([a, b])
    assert stop.title == "Photo stop – sol 3–7"
    assert stop.description == "2 Perseverance photos (MCZ_LEFT, NAVCAM_LEFT)"
    assert stop.extra["sols"] == [3, 7]
    assert stop.extra["files"] == ["b", "a"]  # ordine (sol, file)
    assert stop.extra["cameras"] == ["MCZ_LEFT", "NAVCAM_LEFT"]


def test_best_heading_is_circular_mean():
    stop = make_photo_stop([_photo("a", 0, 0, heading=350), _photo("b", 0, 0, heading=10)])
    assert min(stop.extra["best_heading_deg"], 360 - stop.extra["best_heading_deg"]) \
        == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        make_photo_stop([])


def test_register_photo_stops():
    photos = [_photo("p0", 0.0, 0.0), _photo("p1", 0.0, 0.0, heading=90),
              _photo("far", 100 * M_TO_DEG, 0.0, sol=2)]
    stops = group_into_stops(photos)
    registry = LandmarkRegistry()
    ids = register_photo_stops(registry, stops)
    assert ids == ["photo_stop_000", "photo_stop_001"]

    available = registry.get_available()
    assert [lm.title for lm in available] == [s.title for s in stops]
    for i, s in zip(ids, stops):
        lm = registry.get(i)
        assert (lm.coo.lat, lm.coo.lon) == (s.coo.lat, s.coo.lon)
        assert lm.description == s.description
        assert lm.extra == s.extra
        assert json.loads(json.dumps(lm.extra, allow_nan=False)) == s.extra

    # rettangolo che contiene entrambi gli stop
    both = registry.get_in_region(Coordinate(200 * M_TO_DEG, -10 * M_TO_DEG),
                                  Coordinate(-10 * M_TO_DEG, 10 * M_TO_DEG))
    assert sorted(lm.title for lm in both) == sorted(s.title for s in stops)
    # rettangolo che contiene solo quello a 100 m a nord
    (north,) = registry.get_in_region(Coordinate(200 * M_TO_DEG, -10 * M_TO_DEG),
                                      Coordinate(50 * M_TO_DEG, 10 * M_TO_DEG))
    assert north.extra["files"] == ["far"]

    # stesso prefix -> id duplicati, il registry rifiuta l'intero batch
    with pytest.raises(ValueError):
        register_photo_stops(registry, stops)
    assert len(registry.get_available()) == 2
    assert register_photo_stops(registry, stops[:1], prefix="other") == ["other_000"]
    assert len(registry.get_available()) == 3


# 6
def test_find_photo_stops_radius_and_uniqueness():
    path = _north_path()
    # stop 30 m a est del punto 10 (100 m a nord dell'origine)
    stop = _stop(coo=Coordinate(100 * M_TO_DEG, 30 * M_TO_DEG), photos=[_photo("s", 0, 0)])

    # più punti del percorso sono entro 40 m (es. 9, 10, 11), ma l'evento è uno solo
    within = [c for c in path if distance_m(c, stop.coo) <= 40.0]
    assert len(within) > 1

    events = find_photo_stops(path, [stop], radius_m=40.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.path_index == 10
    assert ev.stop is stop
    assert ev.distance_m == pytest.approx(30, abs=0.1)
    assert ev.travel_heading_deg == pytest.approx(0, abs=1e-6)

    assert find_photo_stops(path, [stop], radius_m=20.0) == []


def test_find_photo_stops_last_point_heading_and_order():
    path = _north_path(n=5)
    near_end = _stop(coo=Coordinate(40 * M_TO_DEG, 5 * M_TO_DEG), photos=[_photo("e", 0, 0)])
    near_start = _stop(coo=Coordinate(0.0, -5 * M_TO_DEG), photos=[_photo("b", 0, 0)])
    events = find_photo_stops(path, [near_end, near_start], radius_m=10.0)
    assert [e.path_index for e in events] == [0, 4]
    # finestra troncata ai bordi: [0, 1] all'inizio, [3, 4] alla fine -> sempre nord
    assert events[0].travel_heading_deg == pytest.approx(0, abs=1e-6)
    assert events[1].travel_heading_deg == pytest.approx(0, abs=1e-6)


def test_find_photo_stops_too_short_path():
    stop = _stop(coo=Coordinate(0.0, 0.0), photos=[_photo("s", 0, 0)])
    assert find_photo_stops([Coordinate(0.0, 0.0)], [stop]) == []
    assert find_photo_stops([], [stop]) == []


def test_travel_heading_smoothed_on_staircase_path():
    # scalini N, E, N, E, ... da 1 m per 100 m, come il pathfinding senza diagonali
    lat, lon = 0.0, 0.0
    path = [Coordinate(lat, lon)]
    for k in range(100):
        if k % 2 == 0:
            lat += M_TO_DEG
        else:
            lon += M_TO_DEG
        path.append(Coordinate(lat, lon))
    center = 50
    # stop esattamente sul punto centrale (sugli scalini un offset laterale sposterebbe
    # il punto più vicino su un altro gradino)
    stop = _stop(coo=Coordinate(path[center].lat, path[center].lon),
                     photos=[_photo("s", 0, 0, heading=45)])
    (ev,) = find_photo_stops(path, [stop], radius_m=5.0)
    assert ev.path_index == center
    # fra punti consecutivi il bearing sarebbe 0° o 90°; con la finestra da 20 m è ~45°
    assert bearing_deg(path[center], path[center + 1]) in (pytest.approx(0, abs=1e-6),
                                                           pytest.approx(90, abs=1e-6))
    assert ev.travel_heading_deg == pytest.approx(45, abs=5)
    assert ev.rel_bearing_deg == pytest.approx(0, abs=5)


def test_travel_heading_fallback_when_window_ends_coincide():
    # andata e ritorno: 5 m a nord e di nuovo all'origine. Per il punto 1 la finestra
    # (10 m) va dal punto 0 al punto 2, che coincidono -> bearing del segmento più
    # vicino; a parità quello in avanti (1 -> 2, verso sud).
    a = Coordinate(0.0, 0.0)
    b = Coordinate(5 * M_TO_DEG, 0.0)
    stop = _stop(coo=Coordinate(b.lat, 1 * M_TO_DEG), photos=[_photo("s", 0, 0)])
    (ev,) = find_photo_stops([a, b, Coordinate(0.0, 0.0)], [stop], radius_m=3.0,
                             heading_window_m=10.0)
    assert ev.path_index == 1
    assert ev.travel_heading_deg == pytest.approx(180, abs=1e-6)


# 7
def test_best_photo_is_most_aligned_with_travel():
    path = _north_path()
    ahead = _photo("ahead", 0, 0, heading=0)
    behind = _photo("behind", 0, 0, heading=180)
    stop = _stop(coo=Coordinate(100 * M_TO_DEG, 10 * M_TO_DEG), photos=[behind, ahead])
    (ev,) = find_photo_stops(path, [stop])
    assert ev.best_photo is ahead
    assert ev.rel_bearing_deg == pytest.approx(0, abs=1e-6)
    assert turn_instruction(ev.rel_bearing_deg) == "davanti"


def test_view_cone_north_is_symmetric():
    apex = Coordinate(18.44, 77.45)
    cone = view_cone(apex, heading_deg=0.0, fov_deg=45.0, length_m=150.0)
    assert cone[0] == apex
    left, right = cone[1], cone[-1]
    # i due vertici laterali: stesso lat, lon simmetriche rispetto al vertice
    assert left.lat == pytest.approx(right.lat, abs=1e-12)
    assert left.lon - apex.lon == pytest.approx(-(right.lon - apex.lon), abs=1e-12)
    assert left.lon < apex.lon < right.lon
    # tutti i punti dell'arco a 150 m dal vertice, nella direzione giusta
    for c in cone[1:]:
        assert distance_m(apex, c) == pytest.approx(150.0, abs=1e-6)
    assert bearing_deg(apex, left) == pytest.approx(360 - 22.5, abs=1e-3)
    assert bearing_deg(apex, right) == pytest.approx(22.5, abs=1e-3)
