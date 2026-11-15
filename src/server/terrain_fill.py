"""Conservative missing-data repair for display meshes, never routing grids."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import find_objects, label
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@lru_cache(maxsize=32)
def _has_zero_padding(path, modified, size):
    """Recognize untagged zero fill only when well separated from terrain.

    Requiring many zeros and a narrow, distant elevation range avoids treating
    ordinary zero-elevation crossings or constant zero rasters as missing.
    A bounded overview is read once per file version.
    """
    with rasterio.open(path) as source:
        sample = source.read(1, out_shape=(min(128, source.height), min(128, source.width)),
                             masked=True, resampling=Resampling.nearest).astype(float).filled(np.nan)
    finite = np.isfinite(sample) & (np.abs(sample) < 1e30)
    nonzero = sample[finite & (sample != 0)]
    if np.count_nonzero(finite & (sample == 0)) < 8 or nonzero.size < 32:
        return False
    low, high = np.percentile(nonzero, [5, 95])
    return bool((low > 0 or high < 0) and min(abs(low), abs(high)) > max(100, 4*(high-low)))


def read_preview_heights(source, window, rows, columns):
    # Sample a bounded support grid before averaging, so untagged fill cannot
    # contaminate otherwise valid elevations at missing-data boundaries.
    factor = max(1, min(4, 1024//max(rows, columns),
                        int(window.width/columns), int(window.height/rows)))
    raw = source.read(1, window=window, out_shape=(rows*factor, columns*factor),
                      masked=True, resampling=Resampling.nearest).astype(float).filled(np.nan)
    raw[np.abs(raw) >= 1e30] = np.nan
    zero_padding = False
    if (source.nodata is None and np.issubdtype(np.dtype(source.dtypes[0]), np.floating)
            and np.any(raw == 0)):
        path = Path(source.name).resolve()
        stat = path.stat()
        zero_padding = _has_zero_padding(str(path), stat.st_mtime_ns, stat.st_size)
        if zero_padding:
            raw[raw == 0] = np.nan
    with np.errstate(over='ignore', invalid='ignore'):
        values = raw*source.scales[0] + source.offsets[0]
    finite = np.isfinite(values)
    blocks = (rows, factor, columns, factor)
    support = finite.reshape(blocks).sum(axis=(1, 3))
    total = np.where(finite, values, 0).reshape(blocks).sum(axis=(1, 3))
    values = np.divide(total, support, out=np.full((rows, columns), np.nan),
                       where=support >= factor*factor*.5)
    return values, zero_padding


def fill_small_gaps(values):
    """Harmonic interpolation inside small holes; preserve all measured heights.

    Solve each missing cell as the mean of its four neighbours. This produces
    smooth fills bounded by surrounding heights, without flattening real relief.
    Edge-connected regions, large holes, and extensive missing coverage stay empty.
    """
    missing = ~np.isfinite(values)
    components, _ = label(missing, structure=np.ones((3, 3)))
    candidates = np.zeros(values.shape, dtype=bool)
    rows, cols = values.shape
    for index, region in enumerate(find_objects(components), 1):
        if region is None:
            continue
        y, x = region
        if y.start == 0 or x.start == 0 or y.stop == rows or x.stop == cols:
            continue
        hole = components[region] == index
        if hole.sum() <= 64 and max(hole.shape) <= 12:
            candidates[region] |= hole
    count = int(candidates.sum())
    if not count or count > max(4, int(values.size*.1)):
        return values, 0
    indices = np.full(values.shape, -1, dtype=np.int32)
    indices[candidates] = np.arange(count)
    rr, cc = np.nonzero(candidates)
    matrix_rows, matrix_cols, weights = [np.arange(count)], [np.arange(count)], [np.full(count, 4.)]
    rhs = np.zeros(count)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbours = indices[rr+dr, cc+dc]
        unknown = neighbours >= 0
        matrix_rows.append(np.flatnonzero(unknown)); matrix_cols.append(neighbours[unknown])
        weights.append(np.full(int(unknown.sum()), -1.))
        rhs[~unknown] += values[rr[~unknown]+dr, cc[~unknown]+dc]
    matrix = coo_matrix((np.concatenate(weights), (np.concatenate(matrix_rows), np.concatenate(matrix_cols))),
                        shape=(count, count)).tocsr()
    result = values.copy()
    result[candidates] = spsolve(matrix, rhs)
    return result, count
