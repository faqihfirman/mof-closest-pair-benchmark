"""Algo. 2 (paper): pencarian greedy, plus versi berulang."""
from __future__ import annotations

import math

import numpy as np

Pair = tuple[int, int]


def _greedy_walk(pts: np.ndarray, start: int, path: list | None = None) -> tuple[float, Pair]:
    """Satu jalan greedy dari titik `start`. Mengembalikan (jarak kuadrat, (i, j)).

    Jika `path` berupa list, indeks titik yang dikunjungi ditambahkan berurutan.
    """
    n = len(pts)
    alive = np.ones(n, dtype=bool)
    alive[start] = False
    remaining = n - 1
    cur = start
    best_d2 = math.inf
    best_pair: Pair = (-1, -1)
    if path is not None:
        path.append(start)
    while remaining > 0:
        # Kandidat: titik tersisa dalam kubus berpusat di titik saat ini,
        # setengah sisi = jarak terpendek saat ini (inf pada langkah pertama -> semua titik).
        half = math.sqrt(best_d2)
        in_cube = alive & np.all(np.abs(pts - pts[cur]) <= half, axis=1)
        cand = np.flatnonzero(in_cube)
        if len(cand) == 0:
            break  # tidak ada titik di sekitar: berhenti (langkah 4 paper)
        diff = pts[cand] - pts[cur]
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = int(np.argmin(d2))
        nxt = int(cand[k])
        if d2[k] < best_d2:  # pasangan terbaik hanya diperbarui jika lebih pendek
            best_d2, best_pair = float(d2[k]), (cur, nxt)
        cur = nxt  # pindah ke kandidat terdekat
        if path is not None:
            path.append(cur)
        alive[cur] = False
        remaining -= 1
    return best_d2, best_pair


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
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return math.inf, (-1, -1)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(pts)))
    d2, (i, j) = _greedy_walk(pts, start, path)
    return math.sqrt(d2), (min(i, j), max(i, j))


def closest_pair_greedy_repeated(
    points: np.ndarray, repeats: int, seed: int = 0
) -> tuple[float, Pair, np.ndarray]:
    """Ulangi greedy dari titik awal berbeda, ambil minimum.

    Mengembalikan (jarak, pasangan, riwayat) dengan riwayat[k] = jarak terbaik
    setelah pengulangan ke-(k+1), untuk plot konvergensi.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return math.inf, (-1, -1), np.full(repeats, math.inf)
    rng = np.random.default_rng(seed)
    starts = rng.choice(n, size=repeats, replace=repeats > n)
    best_d2 = math.inf
    best_pair: Pair = (-1, -1)
    history = np.empty(repeats)
    for r, s in enumerate(starts):
        d2, pair = _greedy_walk(pts, int(s))
        if d2 < best_d2:
            best_d2, best_pair = d2, pair
        history[r] = math.sqrt(best_d2)
    i, j = best_pair
    return math.sqrt(best_d2), (min(i, j), max(i, j)), history
