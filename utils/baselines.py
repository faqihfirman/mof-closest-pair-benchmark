"""Baseline modern: KD-tree (ground truth cepat)."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

Pair = tuple[int, int]


def closest_pair_kdtree(points: np.ndarray) -> tuple[float, Pair]:
    """Query tetangga terdekat (k=2, k=1 adalah titik itu sendiri) tiap titik, ambil minimum."""
    points_array = np.asarray(points, dtype=float)
    if len(points_array) < 2:
        return math.inf, (-1, -1)

    tree = cKDTree(points_array)
    # Kolom 0 = titik itu sendiri (jarak 0), kolom 1 = tetangga terdekat sebenarnya.
    neighbor_distances, neighbor_indices = tree.query(points_array, k=2)
    nearest_distance_per_point = neighbor_distances[:, 1]

    point_index = int(np.argmin(nearest_distance_per_point))
    partner_index = int(neighbor_indices[point_index, 1])
    shortest_distance = float(nearest_distance_per_point[point_index])
    return shortest_distance, (min(point_index, partner_index), max(point_index, partner_index))


def k_closest_pairs(points: np.ndarray, k: int = 10) -> list[tuple[float, Pair]]:
    """k pasangan terdekat (eksak), terurut naik.

    Cukup memeriksa k tetangga terdekat tiap titik: jika v bukan salah satunya bagi u,
    maka u sudah punya k pasangan yang lebih dekat sehingga (u, v) tidak masuk k teratas.
    """
    points_array = np.asarray(points, dtype=float)
    num_points = len(points_array)
    if num_points < 2:
        return []

    # +1 karena hasil query ikut memuat titik itu sendiri.
    neighbors_per_point = min(num_points, k + 1)
    neighbor_distances, neighbor_indices = cKDTree(points_array).query(points_array, k=neighbors_per_point)

    # Ratakan jadi daftar kandidat pasangan (titik_asal, tetangga, jarak).
    source_indices = np.repeat(np.arange(num_points), neighbors_per_point)
    neighbor_flat = neighbor_indices.ravel()
    distance_flat = neighbor_distances.ravel()

    # Buang pasangan titik dengan dirinya sendiri (aman juga bila ada titik kembar).
    not_self = source_indices != neighbor_flat
    source_indices = source_indices[not_self]
    neighbor_flat = neighbor_flat[not_self]
    distance_flat = distance_flat[not_self]

    # Pasangan (u, v) dan (v, u) sama: normalisasi jadi (kecil, besar) lalu hapus duplikat.
    smaller_index = np.minimum(source_indices, neighbor_flat)
    larger_index = np.maximum(source_indices, neighbor_flat)
    pair_keys = smaller_index.astype(np.int64) * num_points + larger_index
    _, first_occurrence = np.unique(pair_keys, return_index=True)

    # Urutkan pasangan unik menurut jarak, ambil k teratas.
    order_by_distance = np.argsort(distance_flat[first_occurrence], kind="stable")
    selected = first_occurrence[order_by_distance[:k]]
    return [
        (float(distance_flat[s]), (int(smaller_index[s]), int(larger_index[s])))
        for s in selected
    ]
