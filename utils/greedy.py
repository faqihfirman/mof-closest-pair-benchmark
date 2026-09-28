"""Algo. 2 (paper): pencarian greedy, plus versi berulang."""
from __future__ import annotations

import math

import numpy as np

Pair = tuple[int, int]


def _greedy_walk(
    points_array: np.ndarray, start_index: int, path: list | None = None
) -> tuple[float, Pair]:
    """Satu jalan greedy dari titik `start_index`. Mengembalikan (jarak kuadrat, (i, j)).

    Jika `path` berupa list, indeks titik yang dikunjungi ditambahkan berurutan.
    """
    num_points = len(points_array)
    is_unvisited = np.ones(num_points, dtype=bool)
    is_unvisited[start_index] = False  # "hapus" titik awal dari himpunan
    unvisited_count = num_points - 1
    current_index = start_index
    best_squared = math.inf
    best_pair: Pair = (-1, -1)
    if path is not None:
        path.append(start_index)

    while unvisited_count > 0:
        # Kandidat: titik belum dikunjungi di dalam kubus berpusat di titik saat ini,
        # setengah sisi = jarak terpendek saat ini (inf pada langkah pertama -> semua titik).
        cube_half_side = math.sqrt(best_squared)
        offsets = np.abs(points_array - points_array[current_index])
        inside_cube = is_unvisited & np.all(offsets <= cube_half_side, axis=1)
        candidate_indices = np.flatnonzero(inside_cube)
        if len(candidate_indices) == 0:
            break  # tidak ada titik di sekitar: berhenti (langkah 4 paper)

        # Cari kandidat terdekat dari titik saat ini.
        differences = points_array[candidate_indices] - points_array[current_index]
        candidate_squared = np.einsum("ij,ij->i", differences, differences)
        nearest_rank = int(np.argmin(candidate_squared))
        nearest_index = int(candidate_indices[nearest_rank])
        nearest_squared = float(candidate_squared[nearest_rank])

        if nearest_squared < best_squared:  # pasangan terbaik hanya diperbarui jika lebih pendek
            best_squared, best_pair = nearest_squared, (current_index, nearest_index)

        current_index = nearest_index  # pindah ke kandidat terdekat
        if path is not None:
            path.append(current_index)
        is_unvisited[current_index] = False
        unvisited_count -= 1
    return best_squared, best_pair


def closest_pair_greedy(
    points: np.ndarray, seed: int | None = None, path: list | None = None
) -> tuple[float, Pair]:
    """Greedy setia ke Algo. 2, satu kali jalan dari titik awal acak.

    Asumsi interpretasi pseudocode paper (ambigu):
    - Pada langkah pertama shortestDis = inf, jadi kubus mencakup seluruh titik.
    - Pasangan terbaik hanya diperbarui saat jarak < shortestDis (pseudocode menimpa
      closestPair tiap iterasi, yang jelas bukan maksudnya).
    - Titik kandidat terdekat selalu jadi titik berikutnya, walau jaraknya tidak
      memperbaiki shortestDis. Jalan berhenti jika kubus kosong atau titik habis.
    - Perbandingan memakai jarak kuadrat; akar diambil di akhir.
    Hasil tidak dijamin optimal. `path` (opsional) diisi urutan titik yang dikunjungi.
    """
    points_array = np.asarray(points, dtype=float)
    if len(points_array) < 2:
        return math.inf, (-1, -1)
    random_generator = np.random.default_rng(seed)
    start_index = int(random_generator.integers(len(points_array)))
    best_squared, (index_a, index_b) = _greedy_walk(points_array, start_index, path)
    return math.sqrt(best_squared), (min(index_a, index_b), max(index_a, index_b))


def closest_pair_greedy_repeated(
    points: np.ndarray, repeats: int, seed: int = 0
) -> tuple[float, Pair, np.ndarray]:
    """Ulangi greedy dari titik awal berbeda, ambil minimum.

    Mengembalikan (jarak, pasangan, riwayat) dengan riwayat[k] = jarak terbaik
    setelah pengulangan ke-(k+1), untuk plot konvergensi.
    """
    points_array = np.asarray(points, dtype=float)
    num_points = len(points_array)
    if num_points < 2:
        return math.inf, (-1, -1), np.full(repeats, math.inf)
    random_generator = np.random.default_rng(seed)
    start_indices = random_generator.choice(num_points, size=repeats, replace=repeats > num_points)

    best_squared = math.inf
    best_pair: Pair = (-1, -1)
    best_so_far_history = np.empty(repeats)
    for repeat_number, start_index in enumerate(start_indices):
        walk_squared, walk_pair = _greedy_walk(points_array, int(start_index))
        if walk_squared < best_squared:
            best_squared, best_pair = walk_squared, walk_pair
        best_so_far_history[repeat_number] = math.sqrt(best_squared)

    index_a, index_b = best_pair
    return math.sqrt(best_squared), (min(index_a, index_b), max(index_a, index_b)), best_so_far_history
