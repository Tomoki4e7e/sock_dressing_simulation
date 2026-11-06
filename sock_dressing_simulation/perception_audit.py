"""Image-space tolerance for comparing SAM masks to rasterized mesh support."""
from __future__ import annotations

import numpy as np


def projection_precision(mask, reference, tolerance_px=2):
    import cv2
    mask = np.asarray(mask, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if mask.shape != reference.shape or mask.ndim != 2:
        raise ValueError('Mask/reference image shapes differ')
    if tolerance_px < 0 or int(tolerance_px) != tolerance_px:
        raise ValueError('Tolerance must be a nonnegative integer')
    support = cv2.dilate(reference.astype('uint8'), np.ones((2 * tolerance_px + 1,) * 2, dtype='uint8')) > 0
    area = max(1, int(mask.sum()))
    return dict(strict=float((mask & reference).sum() / area),
                tolerant=float((mask & support).sum() / area), tolerance_px=int(tolerance_px))
