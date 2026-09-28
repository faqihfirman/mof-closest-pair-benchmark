"""Algo. 3 (paper): divide-and-conquer iteratif dengan stack buatan."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .naive import closest_pair_naive_vectorized

Pair = tuple[int, int]


def _isolation_filter(cand: np.ndarray, values: np.ndarray, d: float) -> np.ndarray:
    """Buang titik yang tidak punya tetangga dengan selisih koordinat < d.

    Titik yang terisolasi pada satu sumbu tidak mungkin berpasangan dengan jarak < d,
    jadi pembuangannya aman (tidak mengubah hasil).
    """
    if len(cand) < 2:
        return cand[:0]
    order = np.argsort(values[cand], kind="stable")
    sv = values[cand][order]
    close = np.diff(sv) < d
    keep = np.zeros(len(cand), dtype=bool)
    keep[:-1] |= close
    keep[1:] |= close
    return cand[order[keep]]


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
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    stats: dict = {
        "n_points": n,
        "strip_mode": strip_mode,
        "max_depth": 0,
        "n_distance_computations": 0,
        "strip_by_level": {},
        "strip_overall": {"min": 0, "mean": 0.0, "max": 0},
    }
    if n < 2:
        out = (math.inf, (-1, -1))
        return (out, stats) if return_stats else out

    order = np.argsort(pts[:, 0], kind="stable")
    sp = pts[order]
    X, Y, Z = sp[:, 0].copy(), sp[:, 1].copy(), sp[:, 2].copy()
    pl = sp.tolist()  # daftar Python: lebih cepat untuk loop skalar kecil
    n_dist = 0
    max_depth = 0
    records: list[tuple[int, int, int]] = []  # (level, ukuran subset, ukuran strip)

    # Hasil sub-masalah: (jarak kuadrat, (posisi_a, posisi_b)) pada array terurut.
    results: list[tuple[float, Pair]] = []
    # Frame stack: (lo, hi, depth, mid). mid None = belum dipecah; mid ada = tahap gabung.
    work: list[tuple[int, int, int, int | None]] = [(0, n, 0, None)]

    while work:
        lo, hi, depth, mid = work.pop()
        size = hi - lo
        if mid is None:
            max_depth = max(max_depth, depth)
            if size <= 3:  # base case: brute force
                best, pair = math.inf, (-1, -1)
                for a in range(lo, hi):
                    for b in range(a + 1, hi):
                        pa, pb = pl[a], pl[b]
                        d2 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
                        n_dist += 1
                        if d2 < best:
                            best, pair = d2, (a, b)
                results.append((best, pair))
                continue
            m = lo + min(max(int(size * split_ratio), 1), size - 1)
            work.append((lo, hi, depth, m))  # frame gabung ada di bawah anak-anaknya
            work.append((m, hi, depth + 1, None))
            work.append((lo, m, depth + 1, None))
            continue

        # Tahap gabung: kedua anak sudah selesai (LIFO menjamin urutan ini).
        right = results.pop()
        left = results.pop()
        best_d2, best_pair = left if left[0] <= right[0] else right
        d = math.sqrt(best_d2)
        xm = X[mid]
        a = lo + int(np.searchsorted(X[lo:hi], xm - d, side="right"))
        b = lo + int(np.searchsorted(X[lo:hi], xm + d, side="left"))
        strip_size = b - a
        if strip_size >= 2:
            if strip_mode == "standard":
                idx = (a + np.argsort(Y[a:b], kind="stable")).tolist()
                for k in range(strip_size - 1):
                    pk = pl[idx[k]]
                    for l in range(k + 1, strip_size):
                        pq = pl[idx[l]]
                        dy = pq[1] - pk[1]
                        if dy * dy >= best_d2:
                            break
                        d2 = (pk[0] - pq[0]) ** 2 + dy * dy + (pk[2] - pq[2]) ** 2
                        n_dist += 1
                        if d2 < best_d2:
                            best_d2, best_pair = d2, (idx[k], idx[l])
            else:
                cand = np.arange(a, b)
                cand = _isolation_filter(cand, Y, d)
                cand = _isolation_filter(cand, Z, d)
                m_left = len(cand)
                if m_left >= 2:
                    n_dist += m_left * (m_left - 1) // 2
                    _, (u, v) = closest_pair_naive_vectorized(sp[cand])
                    pa, pb = pl[int(cand[u])], pl[int(cand[v])]
                    d2 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
                    if d2 < best_d2:
                        best_d2, best_pair = d2, (int(cand[u]), int(cand[v]))
        records.append((depth, size, strip_size))
        if trace is not None:
            pa_, pb_ = int(order[best_pair[0]]), int(order[best_pair[1]])
            trace.append({
                "depth": depth, "lo": lo, "hi": hi, "mid": mid, "x_mid": float(xm),
                "d_children": d, "d_after": math.sqrt(best_d2),
                "strip_lo": a, "strip_hi": b, "pair": (min(pa_, pb_), max(pa_, pb_)),
            })
        results.append((best_d2, best_pair))

    best_d2, (pa, pb) = results.pop()
    i, j = int(order[pa]), int(order[pb])
    out = (math.sqrt(best_d2), (min(i, j), max(i, j)))
    if not return_stats:
        return out

    by_level: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for level, size, strip in records:
        by_level[level].append((size, strip))
    for level, vals in sorted(by_level.items()):
        strips = np.array([s for _, s in vals])
        stats["strip_by_level"][level] = {
            "min": int(strips.min()),
            "mean": float(strips.mean()),
            "max": int(strips.max()),
            "mean_subset_size": float(np.mean([sz for sz, _ in vals])),
            "count": len(vals),
        }
    all_strips = np.array([s for _, _, s in records]) if records else np.zeros(1)
    stats["strip_overall"] = {
        "min": int(all_strips.min()),
        "mean": float(all_strips.mean()),
        "max": int(all_strips.max()),
    }
    stats["max_depth"] = max_depth
    stats["n_distance_computations"] = n_dist
    return out, stats
