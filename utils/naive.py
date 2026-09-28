"""Algo. 1 (paper): pencarian naive O(n^2)."""
from __future__ import annotations

import math

import numpy as np

Pair = tuple[int, int]


def closest_pair_naive(points: np.ndarray) -> tuple[float, Pair]:
    """Double loop murni Python sesuai Algo. 1. Mengembalikan (jarak, (i, j)) berupa indeks.

    Perbandingan memakai jarak kuadrat (monoton terhadap jarak), akar dihitung sekali di akhir.
    """
    pts = np.asarray(points, dtype=float).tolist()
    n = len(pts)
    best = math.inf
    pair: Pair = (-1, -1)
    for i in range(n):
        xi, yi, zi = pts[i]
        for j in range(i + 1, n):
            xj, yj, zj = pts[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            if d2 < best:
                best, pair = d2, (i, j)
    return math.sqrt(best), pair


def closest_pair_naive_vectorized(
    points: np.ndarray, chunk_elems: int = 4_000_000
) -> tuple[float, Pair]:
    """Versi NumPy dari naive: masih O(n^2) tetapi tanpa loop Python (pembanding adil).

    Diproses per blok baris agar memori aman (~chunk_elems elemen per blok). Jarak dihitung
    lewat |a|^2+|b|^2-2ab (matmul), lalu pasangan terbaik dihitung ulang eksak.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return math.inf, (-1, -1)
    sq = np.einsum("ij,ij->i", pts, pts)
    rows_per_chunk = max(1, chunk_elems // n)
    best = math.inf
    pair: Pair = (-1, -1)
    for s in range(0, n - 1, rows_per_chunk):
        e = min(s + rows_per_chunk, n - 1)
        rows = np.arange(s, e)
        # hanya kolom > s; sisanya (j <= i) diberi inf
        d2 = sq[s:e, None] + sq[None, s + 1 :] - 2.0 * (pts[s:e] @ pts[s + 1 :].T)
        cols = np.arange(s + 1, n)
        d2[cols[None, :] <= rows[:, None]] = np.inf
        k = int(np.argmin(d2))
        r, c = divmod(k, d2.shape[1])
        if d2[r, c] < best:
            best, pair = float(d2[r, c]), (s + r, s + 1 + c)
    i, j = pair
    return float(np.linalg.norm(pts[i] - pts[j])), pair
