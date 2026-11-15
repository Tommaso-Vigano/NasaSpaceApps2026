"""Aligned measurements along a computed route, ready for browser-side charts."""

import numpy as np


def sample_path(path, coordinates, layers, radius):
    """Sample aligned maps; each series and distance entry matches one path node.

    layers maps a stable series ID to (label, unit, Map). Missing measurements
    become JSON nulls, allowing future constraint series to contain gaps.
    Distances are cumulative horizontal distances on the routing body's sphere.
    """
    nodes = np.asarray(path, dtype=int).reshape(-1, 2)
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    if len(nodes) != len(coordinates):
        raise ValueError('Route nodes and coordinates must align.')
    distance = np.zeros(len(nodes))
    if len(nodes) > 1:
        longitude, latitude = np.radians(coordinates).T
        a = (np.sin(np.diff(latitude)/2)**2 + np.cos(latitude[:-1])*np.cos(latitude[1:])
             * np.sin(np.diff(longitude)/2)**2)
        distance[1:] = np.cumsum(2*radius*np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    series = []
    for id_, (label, unit, map_) in layers.items():
        values = map_.values[nodes[:, 1], nodes[:, 0]]
        series.append({'id': id_, 'label': label, 'unit': unit,
                       'values': [float(value) if np.isfinite(value) else None for value in values]})
    return {'distance_m': distance.tolist(), 'distance_kind': 'horizontal', 'series': series}
