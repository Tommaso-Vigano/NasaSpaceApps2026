"""Metabolic power, walking speed and per-edge time/energy costs for a suited
astronaut on Mars.

SI units throughout: masses in kg, speeds in m/s, power in W, energy in J,
time in s. The grade ``G`` is a percentage (100 * rise / run).
Per-edge arrays follow the direction convention of ``src.models.terrain``.
"""

import warnings

import numpy as np

from src.models.constraints import G_REL, Astronaut
from src.models.terrain import horizontal_lengths, neighbor_stack, step_masks

MINETTI_RANGE = 0.45  # validity of the Minetti polynomial, |i| <= 0.45 [R27]


def pandolf_W(W, L, V, G, eta):
    """Pandolf et al. load-carriage equation [R15], Earth gravity.

    W body mass (kg), L load (kg), V speed (m/s), G grade (%), eta terrain factor.
    M = 1.5 W + 2.0 (W + L)(L/W)^2 + eta (W + L)(1.5 V^2 + 0.35 V G)   [W]
    """
    W = np.asarray(W, dtype=np.float64)
    return (
        1.5 * W
        + 2.0 * (W + L) * (L / W) ** 2
        + eta * (W + L) * (1.5 * V ** 2 + 0.35 * V * G)
    )


def minetti_cost(i):
    """Cost of walking on a gradient [R27], J/(kg m).

    i = gradient as a fraction (tan of the slope angle), valid for
    -0.45 <= i <= 0.45. Values outside are clipped, with a single warning.
    Cw = 280.5 i^5 - 58.7 i^4 - 76.8 i^3 + 51.9 i^2 + 19.6 i + 2.5
    """
    i = np.asarray(i, dtype=np.float64)
    if np.any(np.abs(i[np.isfinite(i)]) > MINETTI_RANGE):
        warnings.warn(
            "minetti_cost: gradient outside [-0.45, 0.45], values clipped.",
            RuntimeWarning,
            stacklevel=2,
        )
    i = np.clip(i, -MINETTI_RANGE, MINETTI_RANGE)
    return 280.5 * i ** 5 - 58.7 * i ** 4 - 76.8 * i ** 3 + 51.9 * i ** 2 + 19.6 * i + 2.5


def mars_walking_power_W(V, G, eta, W, L, dM_suit=0.0):
    """Metabolic power (W) of a suited astronaut walking on Mars.

    - G >= 0: G_REL * Pandolf(W, L, V, G, eta) + dM_suit
    - G <  0: (G_REL * Pandolf(W, L, V, 0, eta) + dM_suit) * Cw(G/100) / Cw(0)
      (Pandolf underestimates downhill cost; Minetti [R27] gives the shape.)
    - never below the standing cost G_REL * 1.5 W + dM_suit.

    TEAM ASSUMPTION: scaling Pandolf linearly with G_REL (Mars/Earth gravity
    ratio) is our own assumption, not a validated model. dM_suit is the extra
    suit overhead, see ``calibrate_dM_suit``.
    """
    G = np.asarray(G, dtype=np.float64)
    up = G_REL * pandolf_W(W, L, V, G, eta) + dM_suit
    flat = G_REL * pandolf_W(W, L, V, 0.0, eta) + dM_suit
    # Minetti only on the downhill part, so the unused branch never warns.
    ratio = minetti_cost(np.minimum(G, 0.0) / 100.0) / minetti_cost(0.0)
    power = np.where(G >= 0, up, flat * ratio)
    standing = G_REL * 1.5 * np.asarray(W, dtype=np.float64) + dM_suit
    return np.maximum(power, standing)


def calibrate_dM_suit(target_W=498.0, V=3.5 / 3.6, W=80.0, L=104.0, eta=1.0):
    """Suit overhead dM (W) such that the flat-ground power equals ``target_W``.

    Reference: NASA average walkback metabolic rate 1700 BTU/h = 498 W [R13].
    """
    return float(target_W - G_REL * pandolf_W(W, L, V, 0.0, eta))


def tobler_speed_ms(G, v_flat, v_max):
    """Walking speed (m/s) from Tobler's hiking function [R28].

    Tobler: v = 6 exp(-3.5 |tan(theta) + 0.05|) km/h, rescaled so that v(0) is
    exactly ``v_flat``, then clipped at ``v_max``.
    TEAM ASSUMPTION: the shape is terrestrial; using it on Mars is our choice.
    """
    s = np.asarray(G, dtype=np.float64) / 100.0
    v = v_flat * np.exp(-3.5 * np.abs(s + 0.05)) / np.exp(-3.5 * 0.05)
    return np.minimum(v, v_max)


def edge_costs(z, dx, dy, eta_map, astronaut: Astronaut, dM_suit, steps=None):
    """Time (s) and energy (J) of every edge, each of shape (8, rows, cols).

    For each edge: d3D = sqrt(dh^2 + dz^2), G = 100 dz / dh,
    V = tobler_speed_ms(G, v_nominal, v_max),
    P = mars_walking_power_W(V, G, mean eta of the two cells, ...),
    t = d3D / V, E = P t.
    Edges are inf when: slope > astronaut.max_slope_deg, step beyond limits
    (if ``steps`` is given, see ``terrain.edge_steps``), NaN in z/eta, or the
    edge leaves the grid. The cost is asymmetric: A->B != B->A on a slope.
    """
    z = np.asarray(z, dtype=np.float64)
    eta = np.broadcast_to(np.asarray(eta_map, dtype=np.float64), z.shape)

    dh = horizontal_lengths(dx, dy)[:, None, None]
    dz = neighbor_stack(z) - z
    eta_e = 0.5 * (eta + neighbor_stack(eta))

    valid = np.isfinite(dz) & np.isfinite(eta_e)
    slope_deg = np.degrees(np.arctan(np.abs(np.where(valid, dz, 0.0)) / dh))
    valid &= slope_deg <= astronaut.max_slope_deg
    if steps is not None:
        valid &= ~step_masks(steps, astronaut.max_step_up_m, astronaut.max_step_down_m)

    dz_v = np.where(valid, dz, 0.0)
    eta_v = np.where(valid, eta_e, 1.0)
    G = 100.0 * dz_v / dh
    d3 = np.hypot(dh, dz_v)

    V = tobler_speed_ms(G, astronaut.v_nominal_ms, astronaut.v_max_ms)
    P = mars_walking_power_W(
        V, G, eta_v, astronaut.body_mass_kg, astronaut.suit_mass_kg, dM_suit
    )
    time_s = np.where(valid, d3 / V, np.inf)
    energy_J = np.where(valid, P * (d3 / V), np.inf)
    return time_s, energy_J


def cell_cost_from_edges(edge_cost, dx, dy, reduce="mean"):
    """Per-cell cost (rows, cols) for cell-based pathfinding.

    Mean (or max) of the FINITE outgoing edges, each divided by its horizontal
    length (cost per metre), multiplied by dx. inf if every outgoing edge is inf.

    Note: this is a bridge to cell-based pathfinding and loses the
    uphill/downhill asymmetry of ``edge_costs``.
    """
    if reduce not in ("mean", "max"):
        raise ValueError("reduce must be 'mean' or 'max'.")
    per_m = np.asarray(edge_cost, dtype=np.float64) / horizontal_lengths(dx, dy)[:, None, None]
    finite = np.isfinite(per_m)
    count = finite.sum(axis=0)
    if reduce == "mean":
        agg = np.full(count.shape, np.inf)
        np.divide(np.where(finite, per_m, 0.0).sum(axis=0), count, out=agg, where=count > 0)
    else:
        agg = np.where(finite, per_m, -np.inf).max(axis=0)
        agg[count == 0] = np.inf
    return agg * dx
