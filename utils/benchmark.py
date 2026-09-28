"""Timing, verifikasi kebenaran, dan eksperimen skala."""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from .baselines import closest_pair_kdtree
from .data import box_for, generate_points
from .divide_conquer import closest_pair_dc
from .greedy import closest_pair_greedy, closest_pair_greedy_repeated
from .naive import closest_pair_naive, closest_pair_naive_vectorized

Algo = Callable[[np.ndarray], tuple[float, tuple[int, int]]]

DEFAULT_RATIOS = [(1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)]


class Timing(NamedTuple):
    median: float
    mean: float
    std: float
    result: object  # hasil pemanggilan terakhir


def time_function(
    fn: Callable, *args, repeats: int = 5, warmup: int = 1,
    max_total_s: float | None = None, **kw,
) -> Timing:
    """Ukur waktu `fn(*args, **kw)` dengan `time.perf_counter`.

    `max_total_s`: berhenti mengulang jika total waktu terukur melewati batas (minimal
    1 sampel), agar algoritma lambat tidak memakan waktu berlebihan.
    """
    result = None
    for _ in range(warmup):
        t0 = time.perf_counter()
        result = fn(*args, **kw)
        dt = time.perf_counter() - t0
        if max_total_s is not None and dt > max_total_s:
            return Timing(dt, dt, 0.0, result)  # terlalu lambat: pemanasan jadi satu-satunya sampel
    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn(*args, **kw)
        times.append(time.perf_counter() - t0)
        if max_total_s is not None and sum(times) > max_total_s:
            break
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return Timing(statistics.median(times), statistics.fmean(times), std, result)


def default_algos(greedy_repeats: int = 20, seed: int = 0) -> dict[str, Algo]:
    """Kumpulan algoritma standar untuk eksperimen (semua: points -> (dist, pair))."""
    return {
        "naive": closest_pair_naive,
        "naive_vec": closest_pair_naive_vectorized,
        "greedy_1x": lambda p: closest_pair_greedy(p, seed=seed),
        "greedy_rep": lambda p: closest_pair_greedy_repeated(p, greedy_repeats, seed=seed)[:2],
        "dc_standard": lambda p: closest_pair_dc(p, strip_mode="standard"),
        "dc_paper": lambda p: closest_pair_dc(p, strip_mode="paper"),
        "kdtree": closest_pair_kdtree,
    }


def verify_correctness(
    points: np.ndarray, algos: dict[str, Algo], atol: float = 1e-9
) -> pd.DataFrame:
    """Bandingkan jarak tiap algoritma dengan KD-tree. Cek juga pasangan yang dilaporkan."""
    pts = np.asarray(points, dtype=float)
    truth, _ = closest_pair_kdtree(pts)
    rows = []
    for name, fn in algos.items():
        d, (i, j) = fn(pts)
        actual = float(np.linalg.norm(pts[i] - pts[j])) if i >= 0 else math.inf
        rows.append({
            "algorithm": name,
            "distance": d,
            "true_distance": truth,
            "abs_diff": abs(d - truth),
            "correct": bool(abs(d - truth) <= atol),
            "pair_consistent": bool(abs(actual - d) <= atol),
        })
    return pd.DataFrame(rows)


def run_scaling_experiment(
    sizes: list[int], dist: str, algos: dict[str, Algo], repeats: int = 3,
    seed: int = 0, max_naive_n: int = 6000, max_total_s: float = 3.0,
) -> pd.DataFrame:
    """Waktu tiap algoritma untuk tiap N. Kerapatan atom dijaga konstan (`box_for`).

    Kolom: n, dist, algorithm, median_time, std_time, correct. Algoritma "naive"
    (Python murni) dilewati untuk n > max_naive_n.
    """
    rows = []
    for n in sizes:
        pts = generate_points(n, dist, box=box_for(n), seed=seed)
        truth, _ = closest_pair_kdtree(pts)
        for name, fn in algos.items():
            if name == "naive" and n > max_naive_n:
                continue
            t = time_function(fn, pts, repeats=repeats, warmup=1, max_total_s=max_total_s)
            d = t.result[0]
            rows.append({
                "n": n, "dist": dist, "algorithm": name,
                "median_time": t.median, "std_time": t.std,
                "correct": bool(abs(d - truth) <= 1e-9),
            })
    return pd.DataFrame(rows)


def run_greedy_accuracy(
    points: np.ndarray, repeat_list: list[int], trials: int = 20
) -> pd.DataFrame:
    """Galat relatif (%) greedy berulang terhadap jarak sebenarnya, plus waktunya.

    Kolom: repeats, mean_rel_error_pct, std_rel_error_pct, frac_exact, mean_time.
    """
    pts = np.asarray(points, dtype=float)
    truth, _ = closest_pair_kdtree(pts)
    rows = []
    for r in repeat_list:
        errs, times = [], []
        for t in range(trials):
            t0 = time.perf_counter()
            d, _, _ = closest_pair_greedy_repeated(pts, r, seed=1000 + t)
            times.append(time.perf_counter() - t0)
            errs.append((d - truth) / truth * 100.0)
        errs_a = np.array(errs)
        rows.append({
            "repeats": r,
            "mean_rel_error_pct": float(errs_a.mean()),
            "std_rel_error_pct": float(errs_a.std()),
            "frac_exact": float(np.mean(errs_a <= 1e-9)),
            "mean_time": float(np.mean(times)),
        })
    return pd.DataFrame(rows)


def run_split_ratio_experiment(
    points_by_size: dict[int, np.ndarray],
    ratios: list[tuple[int, int]] | None = None,
    repeats: int = 3,
) -> pd.DataFrame:
    """Waktu DnC untuk rasio kiri:kanan 1:6 ... 1:1 (analog Fig. 8 paper).

    Kolom: n, ratio (label "a:b"), split_ratio (a/(a+b)), median_time, std_time.
    """
    ratios = ratios or DEFAULT_RATIOS
    rows = []
    for n, pts in points_by_size.items():
        for a, b in ratios:
            t = time_function(closest_pair_dc, pts, split_ratio=a / (a + b),
                              repeats=repeats, warmup=1)
            rows.append({"n": n, "ratio": f"{a}:{b}", "split_ratio": a / (a + b),
                         "median_time": t.median, "std_time": t.std})
    return pd.DataFrame(rows)
