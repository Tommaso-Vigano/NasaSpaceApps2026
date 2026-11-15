import csv
import math

import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("pathfinding")

from src.basic import Coordinate  # noqa: E402
from src.costs.bridge import (  # noqa: E402
    CELL_EPS,
    lonlat_to_rc,
    path_to_coordinates,
    path_to_rc,
    to_gridgraph,
    write_path_csv,
)
from src.landmarks.photo_stops import (  # noqa: E402
    MARS_RADIUS_M,
    Photo,
    PhotoStop,
    find_photo_stops,
)
from src.models.budget import check_eva, dijkstra_grid, return_time_map  # noqa: E402
from src.models.constraints import Astronaut  # noqa: E402
from src.models.metabolic import calibrate_dM_suit, edge_costs  # noqa: E402
from src.models.terrain import OFFSETS, edge_steps  # noqa: E402
from src.routing.algorithm import DIRECTIONS  # noqa: E402

DX = 10.0                                    # m
STEP_DEG = DX / (MARS_RADIUS_M * math.pi / 180.0)
TOP_LEFT = (77.40, 18.45)                    # (lon, lat)
ROWS, COLS = 60, 80
WALL_COL = 40
GAP = range(22, 36)                          # rows where the wall is open
START_RC, END_RC = (5, 10), (5, 70)          # both far from the gap


def _rc_to_lonlat(rc):
    r, c = rc
    return TOP_LEFT[0] + c * STEP_DEG, TOP_LEFT[1] - r * STEP_DEG


def _synthetic_dem():
    """60x80 DEM at 10 m: gaussian hill + 5 m wall on column 40 with a gap."""
    r, c = np.mgrid[0:ROWS, 0:COLS]
    z = 25.0 * np.exp(-(((r - 45) * DX) ** 2 + ((c - 20) * DX) ** 2) / (2 * 120.0 ** 2))
    wall = np.zeros_like(z)
    wall[:, WALL_COL] = 5.0
    wall[list(GAP), WALL_COL] = 0.0
    return z + wall


@pytest.fixture(scope="module")
def scenario():
    z = _synthetic_dem()
    astro = Astronaut()
    t, e = edge_costs(z, DX, DX, 1.1, astro, calibrate_dM_suit(), steps=edge_steps(z))
    gg = to_gridgraph(t, TOP_LEFT, STEP_DEG)
    path, cost = gg.shortest_path(_rc_to_lonlat(START_RC), _rc_to_lonlat(END_RC))
    return dict(z=z, t=t, e=e, gg=gg, path=path, cost=cost, astro=astro)


def _model_cost(edge_cost, path):
    k_of = {d: k for k, d in enumerate(DIRECTIONS)}
    return [edge_cost[k_of[(x2 - x1, y2 - y1)], y1, x1]
            for (x1, y1), (x2, y2) in zip(path, path[1:])]


def test_direction_order_matches():
    assert len(OFFSETS) == len(DIRECTIONS) == 8
    for (drow, dcol), (dx, dy) in zip(OFFSETS, DIRECTIONS):
        assert (dx, dy) == (dcol, drow)


def test_synthetic_path_matches_dijkstra_and_avoids_wall(scenario):
    t, path, cost, gg = scenario["t"], scenario["path"], scenario["cost"], scenario["gg"]
    assert path, "no path found"
    assert path[0] == (START_RC[1], START_RC[0]) and path[-1] == (END_RC[1], END_RC[0])
    n_steps = len(path) - 1

    # (a) same cost as dijkstra_grid
    best = dijkstra_grid(t, [START_RC])[END_RC]
    assert np.isfinite(best)
    assert cost == pytest.approx(best, abs=1e-6 + n_steps * CELL_EPS * math.sqrt(2))

    # (b) every arc of the path is finite in the model -> wall only crossed at the gap
    arcs = _model_cost(t, path)
    assert all(np.isfinite(arcs))
    assert sum(arcs) == pytest.approx(best, abs=1e-6)
    wall_rows = {y for x, y in path if x == WALL_COL}
    assert wall_rows and wall_rows <= set(GAP)

    # (c) forbidden arcs were removed
    assert gg.removed_edges > 0


def test_forbidden_edge_not_in_graph(scenario):
    t, gg = scenario["t"], scenario["gg"]
    E = OFFSETS.index((0, 1))
    r = 5                                       # wall row, not in the gap
    assert not np.isfinite(t[E, r, WALL_COL - 1])
    a, b = (WALL_COL - 1, r), (WALL_COL, r)
    assert a in gg.graph and b in gg.graph      # both are nodes...
    assert not gg.graph.has_edge(a, b)          # ...but the climb is gone
    # and every remaining arc is finite in the model
    k_of = {d: k for k, d in enumerate(DIRECTIONS)}
    for (x1, y1), (x2, y2) in gg.graph.edges:
        assert np.isfinite(t[k_of[(x2 - x1, y2 - y1)], y1, x1])


def test_cell_without_finite_edges_is_not_a_node():
    z = np.zeros((7, 7))
    z[3, 3] = np.nan
    t, _ = edge_costs(z, DX, DX, 1.0, Astronaut(), calibrate_dM_suit())
    gg = to_gridgraph(t, TOP_LEFT, STEP_DEG)
    assert (3, 3) not in gg.graph
    assert (2, 2) in gg.graph
    with pytest.raises(ValueError):
        gg.shortest_path(_rc_to_lonlat((3, 3)), _rc_to_lonlat((0, 0)))


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_invalid_edge_cost_raises(bad):
    ec = np.ones((8, 4, 4))
    ec[2, 1, 1] = bad
    with pytest.raises(ValueError):
        to_gridgraph(ec, TOP_LEFT, STEP_DEG)
    with pytest.raises(ValueError):
        to_gridgraph(np.ones((4, 4)), TOP_LEFT, STEP_DEG)


def test_lonlat_to_rc_roundtrip(scenario):
    gg = scenario["gg"]
    shape = (ROWS, COLS)
    for rc in [(0, 0), (5, 10), (59, 79), (31, 40)]:
        (lon, lat), = gg.convert_idx_to_polar([(rc[1], rc[0])])
        assert lonlat_to_rc((lon, lat), TOP_LEFT, STEP_DEG, shape) == rc
        assert gg._to_node(lon, lat) == (rc[1], rc[0])
        # a point a bit off the centre rounds to the same cell
        assert lonlat_to_rc((lon + 0.4 * STEP_DEG, lat - 0.4 * STEP_DEG),
                            TOP_LEFT, STEP_DEG, shape) == rc
    with pytest.raises(ValueError):
        lonlat_to_rc((TOP_LEFT[0] - STEP_DEG, TOP_LEFT[1]), TOP_LEFT, STEP_DEG, shape)
    with pytest.raises(ValueError):
        lonlat_to_rc(_rc_to_lonlat((ROWS, 0)), TOP_LEFT, STEP_DEG, shape)


def test_path_to_rc_and_check_eva(scenario):
    path, t, e, astro = scenario["path"], scenario["t"], scenario["e"], scenario["astro"]
    rc = path_to_rc(path)
    assert rc[0] == START_RC and rc[-1] == END_RC
    t_ret = return_time_map(t, START_RC)
    rep = check_eva(rc, t, e, t_ret, astro)
    assert rep.t_elapsed[-1] == pytest.approx(sum(_model_cost(t, path)))
    assert np.all(np.isfinite(rep.energy_cum_J))


def test_path_to_coordinates_photo_stops_and_csv(scenario, tmp_path):
    gg, path = scenario["gg"], scenario["path"]
    coords = path_to_coordinates(gg, path)
    assert len(coords) == len(path)
    assert all(isinstance(c, Coordinate) for c in coords)
    lon0, lat0 = _rc_to_lonlat(START_RC)
    assert coords[0].lat == pytest.approx(lat0) and coords[0].lon == pytest.approx(lon0)

    mid = coords[len(coords) // 2]
    photo = Photo(file="p.png", sol=1, camera="NAVCAM_LEFT", lat=mid.lat, lon=mid.lon,
                  elev_m=-2500.0, heading_deg=90.0, el_deg=0.0, url="https://example.org/p.png")
    stop = PhotoStop(coo=Coordinate(lat=mid.lat, lon=mid.lon), title="t", description="",
                     extra={"type": "photo_stop"}, photos=[photo])
    events = find_photo_stops(coords, [stop], radius_m=15.0)
    assert len(events) == 1 and events[0].distance_m == pytest.approx(0.0, abs=1e-6)

    out = tmp_path / "path.csv"
    write_path_csv(coords, out)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["lat", "lon"]
    back = [Coordinate(float(r["lat"]), float(r["lon"])) for r in rows]  # as demo_map.py
    assert back == coords


def test_no_diagonal_steps():
    z = _synthetic_dem()
    t, _ = edge_costs(z, DX, DX, 1.1, Astronaut(), calibrate_dM_suit(), steps=edge_steps(z))
    gg = to_gridgraph(t, TOP_LEFT, STEP_DEG, allow_diagonal=False)
    path, cost = gg.shortest_path(_rc_to_lonlat(START_RC), _rc_to_lonlat(END_RC))
    assert path
    assert all(abs(x2 - x1) + abs(y2 - y1) == 1
               for (x1, y1), (x2, y2) in zip(path, path[1:]))
    assert not any(x1 != x2 and y1 != y2 for (x1, y1), (x2, y2) in gg.graph.edges)
    # optimal among orthogonal-only paths
    t_orth = t.copy()
    t_orth[[k for k, (dr, dc) in enumerate(OFFSETS) if dr and dc]] = np.inf
    best = dijkstra_grid(t_orth, [START_RC])[END_RC]
    assert cost == pytest.approx(best, abs=1e-6 + len(path) * CELL_EPS)
