import math
from pathlib import Path

import numpy as np
import pytest

from utils.baselines import closest_pair_kdtree
from utils.benchmark import (default_algos, run_greedy_accuracy, run_scaling_experiment,
                             run_split_ratio_experiment, time_function, verify_correctness)
from utils.data import generate_points, load_cif, plant_close_pair
from utils.divide_conquer import closest_pair_dc
from utils.greedy import closest_pair_greedy, closest_pair_greedy_repeated
from utils.naive import closest_pair_naive, closest_pair_naive_vectorized

DISTS = ["uniform", "clustered", "lattice_jitter"]
CIF = Path(__file__).parent.parent / "data" / "pcl_v1-on15-3D_1-oe107.cif"


def pair_dist(pts, pair):
    return float(np.linalg.norm(pts[pair[0]] - pts[pair[1]]))


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 17, 200, 700])
def test_exact_algorithms_match_kdtree(dist, n):
    pts = generate_points(n, dist, seed=n)
    truth, _ = closest_pair_kdtree(pts)
    exact = {
        "naive": closest_pair_naive,
        "naive_vec": closest_pair_naive_vectorized,
        "dc_standard": lambda p: closest_pair_dc(p),
        "dc_paper": lambda p: closest_pair_dc(p, strip_mode="paper"),
    }
    for name, fn in exact.items():
        d, pair = fn(pts)
        assert d == pytest.approx(truth, abs=1e-9), name
        assert pair_dist(pts, pair) == pytest.approx(d, abs=1e-9), name
        assert pair[0] < pair[1]


@pytest.mark.parametrize("ratio", [1 / 7, 1 / 4, 0.5, 0.8])
@pytest.mark.parametrize("mode", ["standard", "paper"])
def test_dc_split_ratios(ratio, mode):
    pts = generate_points(500, "clustered", seed=3)
    truth, _ = closest_pair_kdtree(pts)
    d, _ = closest_pair_dc(pts, split_ratio=ratio, strip_mode=mode)
    assert d == pytest.approx(truth, abs=1e-9)


def test_duplicate_points_and_ties():
    pts = generate_points(50, seed=1)
    pts = np.vstack([pts, pts[7]])  # titik kembar -> jarak 0
    for mode in ("standard", "paper"):
        d, pair = closest_pair_dc(pts, strip_mode=mode)
        assert d == 0.0 and set(pair) == {7, 50}
    same_x = np.array([[1.0, y, z] for y in range(6) for z in range(6)])  # semua X sama
    assert closest_pair_dc(same_x)[0] == pytest.approx(1.0)


def test_small_inputs():
    one = np.zeros((1, 3))
    for fn in (closest_pair_naive, closest_pair_naive_vectorized, closest_pair_dc,
               closest_pair_greedy, closest_pair_kdtree):
        assert fn(one)[0] == math.inf


def test_dc_is_iterative_deep_split():
    # rasio ekstrem + recursion limit rendah: hanya lolos jika tanpa rekursi Python
    import sys
    pts = generate_points(3000, "uniform", box=(60, 60, 60), seed=2)
    truth, _ = closest_pair_kdtree(pts)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(60)
    try:
        d, _ = closest_pair_dc(pts, split_ratio=0.02)
    finally:
        sys.setrecursionlimit(old)
    assert d == pytest.approx(truth, abs=1e-9)


def test_dc_stats():
    pts = generate_points(1000, "uniform", box=(45, 45, 45), seed=4)
    (d, pair), stats = closest_pair_dc(pts, return_stats=True)
    assert stats["max_depth"] >= 8
    assert stats["n_distance_computations"] > 0
    assert 0 in stats["strip_by_level"]
    lv = stats["strip_by_level"][0]
    assert lv["min"] <= lv["mean"] <= lv["max"]
    assert stats["n_distance_computations"] < 1000 * 999 // 2


def test_greedy_valid_and_never_below_truth():
    pts = generate_points(400, "uniform", seed=5)
    truth, _ = closest_pair_kdtree(pts)
    for s in range(10):
        d, pair = closest_pair_greedy(pts, seed=s)
        assert d >= truth - 1e-12
        assert pair_dist(pts, pair) == pytest.approx(d)


def test_greedy_repeated_history_and_reproducible():
    pts = generate_points(400, "uniform", seed=5)
    truth, _ = closest_pair_kdtree(pts)
    d, pair, hist = closest_pair_greedy_repeated(pts, 30, seed=1)
    assert len(hist) == 30 and np.all(np.diff(hist) <= 0) and hist[-1] == d
    assert d >= truth - 1e-12
    assert closest_pair_greedy_repeated(pts, 30, seed=1)[0] == d
    # start dari semua titik -> pasti menemukan pasangan terdekat
    d_all, _, _ = closest_pair_greedy_repeated(pts, len(pts), seed=0)
    assert d_all == pytest.approx(truth)


def test_generators():
    for dist in DISTS:
        p = generate_points(300, dist, seed=0)
        assert p.shape == (300, 3) and np.isfinite(p).all()
        assert closest_pair_kdtree(p)[0] > 0
        assert np.array_equal(p, generate_points(300, dist, seed=0))
    with pytest.raises(ValueError):
        generate_points(10, "bogus")


@pytest.mark.parametrize("dist", DISTS)
def test_plant_close_pair(dist):
    pts = generate_points(300, dist, seed=0)
    new, (i, j) = plant_close_pair(pts, 0.96, seed=1)
    assert new.shape == (301, 3)
    d, pair = closest_pair_kdtree(new)
    assert d == pytest.approx(0.96, abs=1e-6)
    assert set(pair) == {i, j}


def test_benchmark_helpers():
    pts = generate_points(300, seed=0)
    t = time_function(closest_pair_dc, pts, repeats=3)
    assert t.median > 0 and t.std >= 0
    df = verify_correctness(pts, default_algos())
    assert df.set_index("algorithm").loc["dc_standard", "correct"]
    assert df["pair_consistent"].all()
    sc = run_scaling_experiment([100, 200], "uniform", default_algos(), repeats=1, max_naive_n=100)
    assert set(sc.columns) == {"n", "dist", "algorithm", "median_time", "std_time", "correct"}
    assert not ((sc.n == 200) & (sc.algorithm == "naive")).any()
    acc = run_greedy_accuracy(pts, [1, 10], trials=3)
    assert list(acc.repeats) == [1, 10]
    sr = run_split_ratio_experiment({300: pts}, [(1, 6), (1, 1)], repeats=1)
    assert list(sr.ratio) == ["1:6", "1:1"]


def test_load_cif():
    pts, sym = load_cif(CIF, return_symbols=True)
    assert pts.shape[1] == 3 and len(pts) == len(sym) > 1000
    # ukuran sel ~ 88 x 93 x 84 Å
    assert 0 < pts.max() < 400
    assert closest_pair_kdtree(pts)[0] > 0


def test_dc_trace():
    pts = generate_points(200, "uniform", seed=6)
    events: list = []
    d, pair = closest_pair_dc(pts, trace=events)
    assert events and events[-1]["depth"] == 0 and events[-1]["lo"] == 0 and events[-1]["hi"] == 200
    assert events[-1]["pair"] == pair and events[-1]["d_after"] == pytest.approx(d)
    assert all(e["d_after"] <= e["d_children"] + 1e-12 for e in events)
    assert closest_pair_dc(pts)[0] == d  # trace tidak mengubah hasil


def test_k_closest_pairs_matches_bruteforce():
    from itertools import combinations
    from utils.baselines import k_closest_pairs
    pts = generate_points(120, "clustered", seed=8)
    pts = np.vstack([pts, pts[3]])  # titik kembar
    brute = sorted((float(np.linalg.norm(pts[i] - pts[j])), (i, j)) for i, j in combinations(range(len(pts)), 2))
    got = k_closest_pairs(pts, k=10)
    assert [d for d, _ in got] == pytest.approx([d for d, _ in brute[:10]])


def test_greedy_path_records_visits():
    pts = generate_points(300, seed=9)
    path: list = []
    d, pair = closest_pair_greedy(pts, seed=3, path=path)
    assert len(path) == len(set(path)) >= 2
    assert set(pair) <= set(path)


def test_paper_algos_only_three():
    from utils.benchmark import paper_algos
    algos = paper_algos()
    assert list(algos) == ["naive", "greedy_1x", "dc_standard"]
    pts = generate_points(150, seed=2)
    df = verify_correctness(pts, algos).set_index("algorithm")
    assert df.loc["naive", "correct"] and df.loc["dc_standard", "correct"]
