"""Rock abundance model of Golombek & Rapp [R5].

k is the total fraction of the surface covered by rocks (CFA, 0..~0.35),
diameters D are in metres. Every function is vectorised over k (maps of k).
"""

import numpy as np

_CHUNK = 512  # unique k values integrated at once (bounds memory use)


def q_of_k(k):
    """Exponent q(k) = 1.79 + 0.152 / k of the CFA model [R5], 1/m."""
    return 1.79 + 0.152 / np.asarray(k, dtype=np.float64)


def cfa(D, k):
    """Cumulative fractional area covered by rocks of diameter >= D [R5]:
    F_k(D) = k * exp(-q(k) * D)."""
    k = np.asarray(k, dtype=np.float64)
    return k * np.exp(-q_of_k(k) * np.asarray(D, dtype=np.float64))


def _trapezoid(f, x):
    """Trapezoidal rule along the last axis (np.trapz is gone in numpy 2.x,
    np.trapezoid does not exist in numpy 1.x)."""
    return np.sum(0.5 * (f[..., 1:] + f[..., :-1]) * np.diff(x), axis=-1)


def number_density_above(D0, k, D_max=10.0, n=2000):
    """Number of rocks per m^2 with diameter >= D0 (D0 scalar, m).

    Integral from D0 to D_max of  k q exp(-q x) / (pi x^2 / 4)  dx: the area
    density -dF/dx divided by the area of one rock. Trapezoidal rule on a
    logarithmic grid of ``n`` points. k <= 0 -> 0, NaN k -> NaN.
    """
    D0 = float(D0)
    if D0 <= 0:
        raise ValueError("D0 must be positive.")
    k = np.asarray(k, dtype=np.float64)
    out = np.full(k.shape, np.nan)
    finite = np.isfinite(k)
    if D0 >= D_max:
        out[finite] = 0.0
        return out[()]

    x = np.geomspace(D0, D_max, n)
    area = np.pi * x ** 2 / 4.0
    uk, inverse = np.unique(k[finite], return_inverse=True)
    values = np.zeros(uk.shape)
    positive = np.flatnonzero(uk > 0)
    for start in range(0, positive.size, _CHUNK):
        idx = positive[start:start + _CHUNK]
        kk = uk[idx][:, None]
        q = q_of_k(kk)
        values[idx] = _trapezoid(kk * q * np.exp(-q * x) / area, x)
    out[finite] = values[inverse.ravel()]
    return out[()]


def p_rock_taller_than(h_lim, k, area_m2, h_over_d=0.5):
    """Probability of at least one rock taller than ``h_lim`` (m) in an area
    of ``area_m2`` (Poisson): 1 - exp(-N(D >= h_lim / h_over_d) * area).

    TEAM ASSUMPTION: rock height = 0.5 * diameter (h_over_d).
    """
    N = number_density_above(h_lim / h_over_d, k)
    return 1.0 - np.exp(-N * np.asarray(area_m2, dtype=np.float64))
