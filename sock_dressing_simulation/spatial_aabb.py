"""Conservative spatial broadphase for large registered convex-part groups."""
import numpy as np
from scipy.spatial import cKDTree


def candidate_indices(lo_a, hi_a, lo_b, hi_b, margin, cache=None, key=None,
                      minimum_product=65536):
    """Find possible part pairs without dropping any pair inside the margin.

    Radius buckets use an enclosing sphere, followed by the original exact
    AABB distance. A cache is valid only for one set of world-frame bounds.
    """
    if len(lo_a)*len(lo_b) <= minimum_product:
        d = np.linalg.norm(np.maximum(np.maximum(lo_a[:, None]-hi_b[None],
                                                lo_b[None]-hi_a[:, None]), 0.), axis=2)
        return np.flatnonzero(d.ravel() <= margin)
    index = None if cache is None else cache.get(key)
    if index is None:
        centers_b = (lo_b+hi_b)/2
        radii_b = np.linalg.norm((hi_b-lo_b)/2, axis=1)
        index = []
        for ids in np.array_split(np.argsort(radii_b), 4):
            if len(ids):
                index.append((ids, cKDTree(centers_b[ids]), float(radii_b[ids].max())))
        if cache is not None:
            cache[key] = index
    centers_a = (lo_a+hi_a)/2
    radii_a = np.linalg.norm((hi_a-lo_a)/2, axis=1)
    found = []
    for ids_b, tree, maximum_radius in index:
        hits = tree.query_ball_point(centers_a, radii_a+maximum_radius+margin+1e-11)
        counts = np.fromiter((len(v) for v in hits), dtype=int, count=len(hits))
        if counts.sum() == 0:
            continue
        ia = np.repeat(np.arange(len(lo_a)), counts)
        ib = ids_b[np.concatenate([np.asarray(v, dtype=int) for v in hits if len(v)])]
        distance = np.linalg.norm(np.maximum(np.maximum(lo_a[ia]-hi_b[ib], lo_b[ib]-hi_a[ia]), 0.), axis=1)
        valid = distance <= margin+1e-12
        found.append(ia[valid]*len(lo_b)+ib[valid])
    return np.sort(np.concatenate(found)) if found else np.empty(0, dtype=int)
