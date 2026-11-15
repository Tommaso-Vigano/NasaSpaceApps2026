"""Array-backed A* for large regular grids, using the executor's slope policy."""

import heapq
import math
from time import monotonic

import numpy as np


def shortest_path(costs, elevation, start, end, slope_policy, radius,
                  allow_diagonal=True, timeout=30):
    deadline = monotonic() + timeout
    values = elevation.values
    rows, cols = values.shape
    valid = np.isfinite(values) & np.isfinite(costs) & (costs > 0) & (costs <= 1e9)
    if not valid[start[1], start[0]] or not valid[end[1], end[0]]:
        return [], None
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    if allow_diagonal:
        directions += [(1, -1), (1, 1), (-1, 1), (-1, -1)]
    dx, dy = elevation.granularity
    # Precompute slope-aware edges in arrays instead of millions of Python dicts.
    edges = []
    for dc, dr in directions:
        y0, y1 = max(0, -dr), min(rows, rows-dr)
        x0, x1 = max(0, -dc), min(cols, cols-dc)
        lat = np.radians(elevation.top_left[1] - np.arange(y0, y1)*dy)
        lat2 = lat - math.radians(dr*dy)
        a = np.sin((lat2-lat)/2)**2 + np.cos(lat)*np.cos(lat2)*math.sin(math.radians(dc*dx)/2)**2
        distance = 2*radius*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        difference = values[y0+dr:y1+dr, x0+dc:x1+dc] - values[y0:y1, x0:x1]
        slopes = np.degrees(np.arctan2(difference, distance[:, None]))
        penalties = np.asarray(slope_policy(slopes))
        weights = costs[y0+dr:y1+dr, x0+dc:x1+dc] * (math.sqrt(2) if dc and dr else 1) + penalties
        allowed = valid[y0:y1, x0:x1] & valid[y0+dr:y1+dr, x0+dc:x1+dc]
        allowed &= (distance[:, None] > 1e-9) & np.isfinite(weights) & (penalties <= 1e9)
        if dc and dr:
            # Never cut diagonally between blocked cells.
            allowed &= valid[y0:y1, x0+dc:x1+dc] & valid[y0+dr:y1+dr, x0:x1]
        grid = np.full((rows, cols), np.inf, dtype=np.float32)
        grid[y0:y1, x0:x1] = np.where(allowed, weights, np.inf)
        edges.append((dc, dr, grid))
    if monotonic() > deadline:
        raise TimeoutError("Routing took too long. Try fewer subdivisions.")
    source, target = start[1]*cols + start[0], end[1]*cols + end[0]
    distance = np.full(rows*cols, np.inf)
    previous = np.full(rows*cols, -1, dtype=np.int32)
    closed = np.zeros(rows*cols, dtype=bool)
    distance[source] = 0
    cheapest = float(costs[valid].min())
    def heuristic(x, y):
        a, b = abs(x-end[0]), abs(y-end[1])
        return cheapest * (a+b + (math.sqrt(2)-2)*min(a,b) if allow_diagonal else a+b)
    queue = [(heuristic(*start), 0.0, source)]
    visits = 0
    while queue:
        _, score, node = heapq.heappop(queue)
        if closed[node] or score > distance[node]:
            continue
        if node == target:
            path = []
            while node != -1:
                path.append((int(node % cols), int(node // cols)))
                node = int(previous[node])
            return path[::-1], float(score)
        closed[node] = True
        y, x = divmod(node, cols)
        visits += 1
        if visits % 1024 == 0 and monotonic() > deadline:
            raise TimeoutError("Routing took too long. Try fewer subdivisions.")
        for dc, dr, weights in edges:
            weight = float(weights[y, x])
            if not math.isfinite(weight):
                continue
            nx, ny = x+dc, y+dr
            neighbour = ny*cols + nx
            candidate = score + weight
            if not closed[neighbour] and candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate+heuristic(nx, ny), candidate, neighbour))
    return [], None
