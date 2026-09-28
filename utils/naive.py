"""Algo. 1 (paper): pencarian naive O(n^2)."""
from __future__ import annotations

import math

import numpy as np

Pair = tuple[int, int]


def closest_pair_naive(points: np.ndarray) -> tuple[float, Pair]:
    """Double loop murni Python sesuai Algo. 1. Mengembalikan (jarak, (i, j)) berupa indeks.

    Perbandingan memakai jarak kuadrat (monoton terhadap jarak), akar dihitung sekali di akhir.
    """
    point_list = np.asarray(points, dtype=float).tolist()
    num_points = len(point_list)
    best_squared = math.inf
    best_pair: Pair = (-1, -1)
    for first in range(num_points):
        x1, y1, z1 = point_list[first]
        for second in range(first + 1, num_points):
            x2, y2, z2 = point_list[second]
            squared = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if squared < best_squared:
                best_squared, best_pair = squared, (first, second)
    return math.sqrt(best_squared), best_pair


def closest_pair_naive_vectorized(
    points: np.ndarray, chunk_elems: int = 4_000_000
) -> tuple[float, Pair]:
    """Versi NumPy dari naive: masih O(n^2) tetapi tanpa loop Python (pembanding adil).

    Diproses per blok baris agar memori aman (~chunk_elems elemen per blok). Jarak dihitung
    lewat |a|^2+|b|^2-2ab (matmul), lalu pasangan terbaik dihitung ulang eksak.
    """
    points_array = np.asarray(points, dtype=float)
    num_points = len(points_array)
    if num_points < 2:
        return math.inf, (-1, -1)
    squared_norms = np.einsum("ij,ij->i", points_array, points_array)  # |p|^2 tiap titik
    rows_per_chunk = max(1, chunk_elems // num_points)
    best_squared = math.inf
    best_pair: Pair = (-1, -1)

    for row_start in range(0, num_points - 1, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, num_points - 1)
        column_start = row_start + 1  # kolom sebelum ini pasti j <= i, tidak perlu dihitung
        row_indices = np.arange(row_start, row_end)
        column_indices = np.arange(column_start, num_points)

        # Matriks jarak kuadrat blok: |a|^2 + |b|^2 - 2 a·b
        block_squared = (
            squared_norms[row_start:row_end, None]
            + squared_norms[None, column_start:]
            - 2.0 * (points_array[row_start:row_end] @ points_array[column_start:].T)
        )
        block_squared[column_indices[None, :] <= row_indices[:, None]] = np.inf  # buang j <= i

        flat_position = int(np.argmin(block_squared))
        block_row, block_column = divmod(flat_position, block_squared.shape[1])
        if block_squared[block_row, block_column] < best_squared:
            best_squared = float(block_squared[block_row, block_column])
            best_pair = (row_start + block_row, column_start + block_column)

    index_a, index_b = best_pair
    exact_distance = float(np.linalg.norm(points_array[index_a] - points_array[index_b]))
    return exact_distance, best_pair
