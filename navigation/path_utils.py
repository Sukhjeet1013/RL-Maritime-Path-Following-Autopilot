import numpy as np


def cross_track_error(p, a, b):
    """
    Compute signed cross-track error from point p to line segment a-b.

    p : ship position (x,y)
    a : previous waypoint
    b : next waypoint

    Returns:
        signed cross-track error
    """

    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    ap = p - a

    ab_len = np.linalg.norm(ab)

    if ab_len < 1e-8:
        return 0.0

    ab_unit = ab / ab_len

    # projection length of p onto segment
    projection = np.dot(ap, ab_unit)

    # clamp projection to segment
    projection = np.clip(projection, 0.0, ab_len)

    closest_point = a + projection * ab_unit

    error_vec = p - closest_point

    cte = np.linalg.norm(error_vec)

    # signed error using cross product
    cross = ab[0] * ap[1] - ab[1] * ap[0]

    if cross < 0:
        cte = -cte

    return float(cte)