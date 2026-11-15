"""A local photo-stop collection bound to the unique Jezero sector."""

import base64
import json
import math
from pathlib import Path

from pyproj import CRS, Transformer
import rasterio


class PerseverancePhotos:
    def __init__(self, regions, directory):
        self.directory = Path(directory).resolve()
        self.regions = regions
        matches = [sector['id'] for sector in regions.sectors.values()
                   if sector['title'].casefold() == 'jezero']
        # Resolve the configured dataset once, then gate requests by its full ID.
        self.sector_id = matches[0] if len(matches) == 1 else None

    def available(self, sector_id):
        return (self.sector_id is not None and sector_id == self.sector_id
                and (self.directory / 'perseverance_26_stop.json').is_file())

    def collection(self, sector_id):
        if not self.available(sector_id):
            raise KeyError(sector_id)
        index = json.loads((self.directory / 'perseverance_26_stop.json').read_text(encoding='utf-8'))
        groups = {}
        with rasterio.open(self.regions.get(sector_id)['path']) as source:
            crs = CRS.from_wkt(source.crs.to_wkt())
            project = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
            inverse = ~source.transform
            for entry in index['stops']:
                lat, lon = float(entry['lat']), float(entry['lon'])
                if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError('Invalid photo-stop coordinates.')
                wx, wy = project.transform(lon, lat, errcheck=True)
                x = inverse.a*wx + inverse.b*wy + inverse.c
                y = inverse.d*wx + inverse.e*wy + inverse.f
                if not (0 <= x < source.width and 0 <= y < source.height):
                    continue
                images = []
                for name in dict.fromkeys([entry['image'], *entry.get('other_photos_same_stop', [])]):
                    path = (self.directory / name).resolve()
                    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                            '.webp': 'image/webp'}.get(path.suffix.lower())
                    # Only indexed, locally installed image files are exposed.
                    if (Path(name).name != name or not path.is_relative_to(self.directory)
                            or not mime or not path.is_file()):
                        continue
                    images.append({'name': name, 'lat': entry.get('photo_lat', lat),
                                   'lon': entry.get('photo_lon', lon),
                                   'image': f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')})
                if not images:
                    continue
                group = groups.setdefault((lat, lon), {
                    'id': str(entry['id']), 'title': entry.get('title', 'Perseverance photo stop'),
                    'lat': lat, 'lon': lon, 'x': x, 'y': y, 'images': [],
                })
                existing = {image['name'] for image in group['images']}
                group['images'].extend(image for image in images if image['name'] not in existing)
        return {'sector_id': sector_id, 'stops': list(groups.values())}
