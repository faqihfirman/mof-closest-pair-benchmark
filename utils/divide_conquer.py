"""Algo. 3 (paper): divide-and-conquer iteratif dengan stack buatan."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .naive import closest_pair_naive_vectorized

Pair = tuple[int, int]
# Hasil satu sub-masalah: (jarak kuadrat terbaik, (posisi_a, posisi_b) pada array terurut X).
SubResult = tuple[float, Pair]


def _squared_distance(point_a: list[float], point_b: list[float]) -> float:
    """Jarak Euclidean kuadrat 3D (tanpa akar: cukup untuk membandingkan)."""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return dx * dx + dy * dy + dz * dz


def _brute_force_block(sorted_points: list[list[float]], start: int, end: int) -> tuple[SubResult, int]:
    """Base case: bandingkan semua pasangan di sorted_points[start:end] (maks 3 titik).

    Mengembalikan (hasil, jumlah perhitungan jarak).
    """
    best_squared = math.inf
    best_pair: Pair = (-1, -1)
    distance_count = 0
    for first in range(start, end):
        for second in range(first + 1, end):
            squared = _squared_distance(sorted_points[first], sorted_points[second])
            distance_count += 1
            if squared < best_squared:
                best_squared, best_pair = squared, (first, second)
    return (best_squared, best_pair), distance_count


def _scan_strip_standard(
    sorted_points: list[list[float]], y_values: np.ndarray, strip_start: int, strip_end: int,
    best_squared: float, best_pair: Pair,
) -> tuple[SubResult, int]:
    """Strip "standard": urutkan strip menurut Y, bandingkan tiap titik dengan titik-titik
    berikutnya selama selisih Y masih < d (d ikut mengecil saat pasangan lebih dekat ditemukan).
    """
    strip_by_y = (strip_start + np.argsort(y_values[strip_start:strip_end], kind="stable")).tolist()
    strip_size = len(strip_by_y)
    distance_count = 0
    for current_rank in range(strip_size - 1):
        current_point = sorted_points[strip_by_y[current_rank]]
        for other_rank in range(current_rank + 1, strip_size):
            other_point = sorted_points[strip_by_y[other_rank]]
            y_gap = other_point[1] - current_point[1]
            if y_gap * y_gap >= best_squared:
                break  # titik berikutnya lebih jauh lagi di Y: tidak mungkin lebih dekat
            squared = _squared_distance(current_point, other_point)
            distance_count += 1
            if squared < best_squared:
                best_squared = squared
                best_pair = (strip_by_y[current_rank], strip_by_y[other_rank])
    return (best_squared, best_pair), distance_count


def _isolation_filter(candidates: np.ndarray, axis_values: np.ndarray, width: float) -> np.ndarray:
    """Buang titik yang tidak punya tetangga dengan selisih koordinat < width pada satu sumbu.

    Titik yang terisolasi pada satu sumbu tidak mungkin berpasangan dengan jarak < width,
    jadi pembuangannya aman (tidak mengubah hasil).
    """
    if len(candidates) < 2:
        return candidates[:0]
    order_on_axis = np.argsort(axis_values[candidates], kind="stable")
    sorted_values = axis_values[candidates][order_on_axis]
    close_to_next = np.diff(sorted_values) < width
    keep = np.zeros(len(candidates), dtype=bool)
    keep[:-1] |= close_to_next  # dekat dengan tetangga sesudahnya
    keep[1:] |= close_to_next   # dekat dengan tetangga sebelumnya
    return candidates[order_on_axis[keep]]


def _scan_strip_paper(
    sorted_array: np.ndarray, sorted_points: list[list[float]], y_values: np.ndarray, z_values: np.ndarray,
    strip_start: int, strip_end: int, strip_width: float, best_squared: float, best_pair: Pair,
) -> tuple[SubResult, int]:
    """Strip "paper": filter X (sudah lewat batas strip), lalu Y, lalu Z, lalu brute force sisa titik."""
    candidates = np.arange(strip_start, strip_end)
    candidates = _isolation_filter(candidates, y_values, strip_width)
    candidates = _isolation_filter(candidates, z_values, strip_width)
    remaining = len(candidates)
    if remaining < 2:
        return (best_squared, best_pair), 0
    distance_count = remaining * (remaining - 1) // 2
    _, (local_a, local_b) = closest_pair_naive_vectorized(sorted_array[candidates])
    position_a, position_b = int(candidates[local_a]), int(candidates[local_b])
    squared = _squared_distance(sorted_points[position_a], sorted_points[position_b])
    if squared < best_squared:
        best_squared, best_pair = squared, (position_a, position_b)
    return (best_squared, best_pair), distance_count


def _summarize_stats(merge_records: list[tuple[int, int, int]]) -> tuple[dict, dict]:
    """Ringkas ukuran strip per level rekursi dan keseluruhan."""
    records_by_level: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for level, subset_size, strip_size in merge_records:
        records_by_level[level].append((subset_size, strip_size))
    strip_by_level = {}
    for level, level_records in sorted(records_by_level.items()):
        strip_sizes = np.array([strip for _, strip in level_records])
        strip_by_level[level] = {
            "min": int(strip_sizes.min()),
            "mean": float(strip_sizes.mean()),
            "max": int(strip_sizes.max()),
            "mean_subset_size": float(np.mean([subset for subset, _ in level_records])),
            "count": len(level_records),
        }
    all_strip_sizes = np.array([strip for _, _, strip in merge_records]) if merge_records else np.zeros(1)
    strip_overall = {
        "min": int(all_strip_sizes.min()),
        "mean": float(all_strip_sizes.mean()),
        "max": int(all_strip_sizes.max()),
    }
    return strip_by_level, strip_overall


def closest_pair_dc(
    points: np.ndarray,
    split_ratio: float = 0.5,
    strip_mode: str = "standard",
    return_stats: bool = False,
    trace: list | None = None,
):
    """Divide-and-conquer closest pair 3D, iteratif (tanpa rekursi Python).

    Langkah: urutkan menurut X, bagi di indeks int(n*split_ratio), selesaikan kiri dan
    kanan secara iteratif dengan stack buatan (post-order), d = min(d1, d2), lalu periksa
    strip |x - x_mid| < d di sekitar garis pembagi. Base case: <= 3 titik (brute force).

    strip_mode:
    - "standard": strip diurutkan menurut Y, tiap titik dibandingkan dengan titik berikutnya
      selama selisih Y < d (jarak penuh 3D dicek, jadi Z ikut terjaga).
    - "paper": filter berurutan X, lalu Y, lalu Z, lalu brute force pada sisa titik.
      Asumsi: paper tidak menyebut pusat filter Y/Z. Dipakai filter isolasi (titik dibuang
      jika tak punya tetangga dengan selisih < d pada sumbu itu), satu-satunya pemotongan
      yang tetap menjamin hasil benar.

    Jika `trace` berupa list, tiap tahap gabung (urutan post-order) ditambahkan sebagai dict
    (depth, lo, hi, mid, x_mid, d_children, d_after, strip_lo, strip_hi, pair) untuk
    visualisasi langkah demi langkah; lo/hi/mid/strip_* adalah posisi pada array terurut X
    (np.argsort(kind="stable")), pair berupa indeks asli.

    Mengembalikan (jarak, (i, j)) berupa indeks asli. Jika return_stats=True, mengembalikan
    ((jarak, pasangan), stats) dengan stats: max_depth, n_distance_computations,
    strip_by_level {level: min/mean/max/mean_subset_size/count}, strip_overall.
    """
    if strip_mode not in ("standard", "paper"):
        raise ValueError(f"strip_mode tidak dikenal: {strip_mode!r}")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio harus di antara 0 dan 1")
    points_array = np.asarray(points, dtype=float)
    num_points = len(points_array)
    stats: dict = {
        "n_points": num_points,
        "strip_mode": strip_mode,
        "max_depth": 0,
        "n_distance_computations": 0,
        "strip_by_level": {},
        "strip_overall": {"min": 0, "mean": 0.0, "max": 0},
    }
    if num_points < 2:
        no_result = (math.inf, (-1, -1))
        return (no_result, stats) if return_stats else no_result

    # 1. Urutkan menurut X. sorted_to_original[p] = indeks asli titik di posisi terurut p.
    sorted_to_original = np.argsort(points_array[:, 0], kind="stable")
    sorted_array = points_array[sorted_to_original]
    x_values = sorted_array[:, 0].copy()
    y_values = sorted_array[:, 1].copy()
    z_values = sorted_array[:, 2].copy()
    sorted_points = sorted_array.tolist()  # list Python: lebih cepat untuk loop skalar kecil

    total_distance_count = 0
    max_depth = 0
    merge_records: list[tuple[int, int, int]] = []  # (level, ukuran subset, ukuran strip)

    # 2. Rekursi ditiru dengan dua stack:
    #    - pending_tasks: frame (start, end, depth, split_index). split_index None = belum
    #      dipecah; split_index terisi = kedua anak sudah dijadwalkan, tinggal digabung.
    #    - solved_results: hasil sub-masalah yang sudah selesai (dipakai tahap gabung).
    solved_results: list[SubResult] = []
    pending_tasks: list[tuple[int, int, int, int | None]] = [(0, num_points, 0, None)]

    while pending_tasks:
        start, end, depth, split_index = pending_tasks.pop()
        subset_size = end - start

        if split_index is None:
            max_depth = max(max_depth, depth)
            if subset_size <= 3:  # base case: brute force
                block_result, count = _brute_force_block(sorted_points, start, end)
                total_distance_count += count
                solved_results.append(block_result)
                continue
            # DIVIDE: tentukan titik bagi. Frame gabung didorong lebih dulu agar diproses
            # setelah kedua anaknya (stack = LIFO, jadi kiri diproses dulu, lalu kanan).
            split_index = start + min(max(int(subset_size * split_ratio), 1), subset_size - 1)
            pending_tasks.append((start, end, depth, split_index))
            pending_tasks.append((split_index, end, depth + 1, None))   # kanan
            pending_tasks.append((start, split_index, depth + 1, None))  # kiri
            continue

        # COMBINE: kedua anak sudah selesai. Hasil kanan ada di puncak stack.
        right_result = solved_results.pop()
        left_result = solved_results.pop()
        best_squared, best_pair = left_result if left_result[0] <= right_result[0] else right_result
        children_distance = math.sqrt(best_squared)  # d = min(d1, d2)

        # Strip: titik dengan |x - x_mid| < d (dicari cepat karena x_values sudah terurut).
        x_mid = x_values[split_index]
        subset_x = x_values[start:end]
        strip_start = start + int(np.searchsorted(subset_x, x_mid - children_distance, side="right"))
        strip_end = start + int(np.searchsorted(subset_x, x_mid + children_distance, side="left"))
        strip_size = strip_end - strip_start

        if strip_size >= 2:
            if strip_mode == "standard":
                (best_squared, best_pair), count = _scan_strip_standard(
                    sorted_points, y_values, strip_start, strip_end, best_squared, best_pair)
            else:
                (best_squared, best_pair), count = _scan_strip_paper(
                    sorted_array, sorted_points, y_values, z_values, strip_start, strip_end,
                    children_distance, best_squared, best_pair)
            total_distance_count += count

        merge_records.append((depth, subset_size, strip_size))
        if trace is not None:
            original_a = int(sorted_to_original[best_pair[0]])
            original_b = int(sorted_to_original[best_pair[1]])
            trace.append({
                "depth": depth, "lo": start, "hi": end, "mid": split_index, "x_mid": float(x_mid),
                "d_children": children_distance, "d_after": math.sqrt(best_squared),
                "strip_lo": strip_start, "strip_hi": strip_end,
                "pair": (min(original_a, original_b), max(original_a, original_b)),
            })
        solved_results.append((best_squared, best_pair))

    # 3. Sisa satu hasil = solusi seluruh himpunan. Kembalikan ke indeks asli.
    best_squared, (position_a, position_b) = solved_results.pop()
    original_a = int(sorted_to_original[position_a])
    original_b = int(sorted_to_original[position_b])
    result = (math.sqrt(best_squared), (min(original_a, original_b), max(original_a, original_b)))
    if not return_stats:
        return result

    stats["strip_by_level"], stats["strip_overall"] = _summarize_stats(merge_records)
    stats["max_depth"] = max_depth
    stats["n_distance_computations"] = total_distance_count
    return result, stats
