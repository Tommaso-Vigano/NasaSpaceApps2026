"""Bounded, windowed elevation geometry for the read-only 3D preview."""

import base64
import math

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window

from src.server.terrain_fill import fill_small_gaps, read_preview_heights


def terrain_preview(path, x, y, width, height, samples=192):
    if (not all(math.isfinite(v) for v in (x, y, width, height))
            or width <= 0 or height <= 0
            or not isinstance(samples, int) or not 32 <= samples <= 512):
        raise ValueError('Invalid 3D viewport or mesh detail (32–512).')
    with rasterio.open(path) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError('3D terrain requires a georeferenced elevation TIFF.')
        left, top = max(0, x), max(0, y)
        right, bottom = min(source.width, x + width), min(source.height, y + height)
        if right-left < 2 or bottom-top < 2:
            raise ValueError('Zoom out to include at least two terrain pixels per axis.')
        window = Window(left, top, right-left, bottom-top)
        ratio = min(1, samples/window.width, samples/window.height)
        columns, rows = max(2, round(window.width*ratio)), max(2, round(window.height*ratio))
        values, masked_zero_padding = read_preview_heights(source, window, rows, columns)
        values, imputed_cells = fill_small_gaps(values)
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError('No valid elevations in this area. Move to covered terrain.')
        offset = float(np.median(values[valid]))
        # Coordinates describe sample centres, matching Rasterio's window reads.
        px, py = np.meshgrid(left + (np.arange(columns)+.5)*window.width/columns,
                             top + (np.arange(rows)+.5)*window.height/rows)
        affine = source.transform
        wx, wy = affine.a*px + affine.b*py + affine.c, affine.d*px + affine.e*py + affine.f
        crs = CRS.from_wkt(source.crs.to_wkt())
        geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        lon, lat = geographic.transform(wx, wy, errcheck=True)
        centre_x, centre_y = left+window.width/2, top+window.height/2
        cx = affine.a*centre_x + affine.b*centre_y + affine.c
        cy = affine.d*centre_x + affine.e*centre_y + affine.f
        centre_lon, centre_lat = geographic.transform(cx, cy, errcheck=True)
        radius = crs.ellipsoid.semi_major_metre
        # Local east/south distances on the source body's sphere. Suitable for
        # these small patches; heights stay in metres with no exaggeration.
        east = np.radians((lon-centre_lon+180) % 360-180)*radius*math.cos(math.radians(centre_lat))
        south = np.radians(centre_lat-lat)*radius
        valid &= np.isfinite(east) & np.isfinite(south)
        positions = np.stack((east, np.where(valid, values-offset, 0), south), axis=-1)
        positions[~np.isfinite(positions)] = 0
        return {'shape': [rows, columns],
                'positions': base64.b64encode(positions.astype('<f4').tobytes()).decode('ascii'),
                'valid': base64.b64encode(valid.astype('uint8').tobytes()).decode('ascii'),
                'viewport': {'x': left, 'y': top, 'width': window.width, 'height': window.height},
                'height_offset': offset, 'units': 'm', 'vertical_exaggeration': 1,
                'imputed_cells': imputed_cells, 'masked_zero_padding': masked_zero_padding}
