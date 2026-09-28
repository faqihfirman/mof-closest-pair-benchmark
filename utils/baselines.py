"""Baseline modern: KD-tree (ground truth cepat)."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

Pair = tuple[int, int]


def closest_pair_kdtree(points: np.ndarray) -> tuple[float, Pair]:
    """Query tetangga terdekat (k=2, k=1 adalah titik itu sendiri) tiap titik, ambil minimum."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return math.inf, (-1, -1)
    dists, idx = cKDTree(pts).query(pts, k=2)
    i = int(np.argmin(dists[:, 1]))
    j = int(idx[i, 1])
    return float(dists[i, 1]), (min(i, j), max(i, j))
