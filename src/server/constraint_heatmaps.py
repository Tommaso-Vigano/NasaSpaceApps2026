"""Lossless grid previews: one PNG pixel per routing cell, never resampled."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


COLORS = ['#3c91b5', '#efd074', '#d35245']
BLOCKED_COLOR = '#e063cf'


def encode_heatmaps(maps, elevation, projected, inverse):
    rows, columns = elevation.values.shape
    dx, dy = elevation.granularity
    # A small mesh places the geographic grid over the source TIFF, including
    # projected/rotated rasters. Mesh interpolation affects placement, not costs.
    xs = np.linspace(0, columns, min(17, columns + 1))
    ys = np.linspace(0, rows, min(17, rows + 1))
    x, y = np.meshgrid(xs, ys)
    lon = elevation.top_left[0] + (x - .5) * dx
    lat = elevation.top_left[1] - (y - .5) * dy
    wx, wy = projected.transform(lon, lat, errcheck=True)
    pixel_mesh = np.stack((inverse.a*wx + inverse.b*wy + inverse.c,
                           inverse.d*wx + inverse.e*wy + inverse.f), axis=-1)
    layers = []
    for index, (title, values) in enumerate(maps.items()):
        values = np.asarray(values)
        if values.shape != (rows, columns):
            raise ValueError('Constraint heatmaps must match the routing grid.')
        finite = np.isfinite(values)
        blocked = np.isinf(values) | (finite & (values > 1e9))
        finite &= ~blocked
        low = float(values[finite].min()) if finite.any() else None
        high = float(values[finite].max()) if finite.any() else None
        normalized = np.zeros(values.shape)
        if low is not None and high > low:
            normalized[finite] = (values[finite] - low) / (high - low)
        palette = np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)] for color in COLORS])
        rgba = np.zeros((rows, columns, 4), dtype=np.uint8)
        for channel in range(3):
            rgba[..., channel] = np.interp(normalized, [0, .5, 1], palette[:, channel]).astype(np.uint8)
        rgba[finite, 3] = 255
        rgba[blocked] = [224, 99, 207, 255]
        buffer = BytesIO()
        Image.fromarray(rgba).save(buffer, format='PNG')
        layers.append({'id': str(index), 'title': title, 'minimum': low, 'maximum': high,
                       'colors': COLORS, 'blocked_color': BLOCKED_COLOR,
                       'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')})
    return {'shape': [rows, columns], 'pixel_mesh': pixel_mesh.tolist(), 'layers': layers}
