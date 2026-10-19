"""Outcome-independent spatial query and local affine initialization rules."""
import numpy as np

SETTINGS = {'grid_side': 16, 'neighbors': 32, 'neighbor_radius_pixels': 64.,
            'minimum_neighbors': 8, 'source_normal_condition_limit': 1e6,
            'affine_condition_limit': 100., 'cache_to_array_offset': -0.5}


def select_and_initialize(x1, x2, inliers, width, height):
    x1, x2 = np.asarray(x1, dtype=np.float64), np.asarray(x2, dtype=np.float64)
    mask = np.asarray(inliers)
    if (x1.ndim != 2 or x1.shape[1] != 2 or x2.shape != x1.shape
            or mask.shape != (len(x1),) or mask.dtype != np.bool_
            or not np.isfinite(x1).all() or not np.isfinite(x2).all()
            or type(width) is not int or type(height) is not int
            or width < 32 or height < 32):
        raise ValueError('invalid coordinates, mask or image dimensions')
    ids = np.flatnonzero(mask)
    if np.any(x1[ids] < 0) or np.any(x1[ids] >= [width, height]):
        raise ValueError('cached source coordinates outside paired image extent')
    if not len(ids):
        return []
    extent = np.array([width, height], dtype=np.float64)
    cells_xy = np.floor(16. * x1[ids] / extent).astype(np.int64)
    cells = 16 * cells_xy[:, 1] + cells_xy[:, 0]
    centers = (cells_xy + .5) * extent / 16.
    distances = np.sum((x1[ids] - centers)**2, axis=1)
    order = np.lexsort((ids, distances, cells))
    chosen = []
    previous = None
    for pos in order:
        cell = int(cells[pos])
        if cell != previous:
            chosen.append((cell, int(ids[pos])))
            previous = cell
    rows = []
    for cell, index in chosen:
        row = {'cell': cell, 'index': index, 'status': 'initialization_failed'}
        d = x1[ids] - x1[index]
        distances = np.sum(d*d, axis=1)
        eligible = (ids != index) & (distances <= 64.**2)
        neighbors = ids[eligible]
        nd = distances[eligible]
        neighbors = neighbors[np.lexsort((neighbors, nd))[:32]]
        row['neighbor_indices'] = neighbors.tolist()
        row['neighbor_count'] = len(neighbors)
        if len(neighbors) < 8:
            row['reason'] = 'fewer_than_eight_local_inlier_neighbors'
            rows.append(row); continue
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                X, Y = x1[neighbors]-x1[index], x2[neighbors]-x2[index]
                H = np.einsum('ni,nj->ij', X, X, optimize=False)
                cross = np.einsum('ni,nj->ij', X, Y, optimize=False)
                condition = float(np.linalg.cond(H))
                if not np.isfinite(condition) or condition > 1e6:
                    raise ValueError('source_neighbor_geometry_ill_conditioned')
                A = np.linalg.solve(H, cross).T
                acondition = float(np.linalg.cond(A))
                if (not np.isfinite(A).all() or not np.isfinite(acondition)
                        or np.linalg.det(A) <= 0 or acondition > 100.):
                    raise ValueError('affine_initialization_degenerate_or_reflected')
            row.update(status='initialized', A0=A.tolist(),
                       source_normal_condition=condition, affine_condition=acondition)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            row['reason'] = str(exc)
        rows.append(row)
    return rows
