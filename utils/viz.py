"""Semua fungsi plotting (matplotlib). Tiap fungsi mengembalikan Figure."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .baselines import closest_pair_kdtree

# Warna & label konsisten antar plot
COLORS = {
    "naive": "#c0392b", "naive_vec": "#e67e22", "greedy_1x": "#95a5a6",
    "greedy_rep": "#7f8c8d", "dc_standard": "#2471a3", "dc_paper": "#7d3c98",
    "kdtree": "#1e8449",
}
LABELS = {
    "naive": "Naive (Python)", "naive_vec": "Naive (NumPy)", "greedy_1x": "Greedy 1x",
    "greedy_rep": "Greedy berulang", "dc_standard": "DnC (standard)",
    "dc_paper": "DnC (paper)", "kdtree": "KD-tree",
}
DIST_COLORS = {"uniform": "#2471a3", "clustered": "#c0392b", "lattice_jitter": "#1e8449"}
HIGHLIGHT = "#e74c3c"

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
})


def _c(name: str) -> str:
    return COLORS.get(name, "#333333")


def _l(name: str) -> str:
    return LABELS.get(name, name)


def plot_points_3d(points: np.ndarray, pair: tuple[int, int] | None = None, title: str = "") -> Figure:
    """Scatter 3D; pasangan terdekat diberi warna dan garis mencolok."""
    pts = np.asarray(points)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c="#7fb3d5", alpha=0.6, label="Atom")
    if pair is not None:
        a, b = pts[pair[0]], pts[pair[1]]
        d = float(np.linalg.norm(a - b))
        ax.plot(*zip(a, b), color=HIGHLIGHT, lw=3, label=f"Pasangan terdekat (d = {d:.3f} Å)")
        ax.scatter(*zip(a, b), s=60, c=HIGHLIGHT, depthshade=False)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return fig


def plot_dc_split_2d(points: np.ndarray, split_ratio: float = 0.5, title: str = "") -> Figure:
    """Ilustrasi satu level DnC (mirip Fig. 5): proyeksi XY, garis pembagi, strip lebar 2d.

    d1, d2 = jarak terdekat 3D di kiri dan kanan; d = min(d1, d2). Strip: |x - x_mid| < d.
    """
    pts = np.asarray(points)
    sp = pts[np.argsort(pts[:, 0], kind="stable")]
    m = min(max(int(len(sp) * split_ratio), 2), len(sp) - 2)
    left, right = sp[:m], sp[m:]
    d1, p1 = closest_pair_kdtree(left)
    d2, p2 = closest_pair_kdtree(right)
    d = min(d1, d2)
    xm = sp[m, 0]
    in_strip = np.abs(sp[:, 0] - xm) < d
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(xm - d, xm + d, color="#f9e79f", alpha=0.6, label=f"Strip |x − x_mid| < d  ({in_strip.sum()} titik)")
    ax.scatter(left[:, 0], left[:, 1], s=14, c="#2471a3", label=f"Kiri ({len(left)})")
    ax.scatter(right[:, 0], right[:, 1], s=14, c="#1e8449", label=f"Kanan ({len(right)})")
    ax.axvline(xm, color="k", ls="--", lw=1.2, label=f"Garis pembagi x_mid = {xm:.2f}")
    for pr, base, col, name, dd in ((p1, left, "#154360", "d1", d1), (p2, right, "#0b5345", "d2", d2)):
        a, b = base[pr[0]], base[pr[1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=2.5)
        ax.annotate(f"{name} = {dd:.2f}", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                    textcoords="offset points", xytext=(6, 6), color=col, weight="bold")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)  [proyeksi XY]")
    ax.set_title(title or f"Satu level pembagian DnC   d = min(d1, d2) = {d:.2f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    return fig


def _anchor(sub: pd.DataFrame, algo: str) -> tuple[float, float] | None:
    s = sub[sub.algorithm == algo].sort_values("n")
    return None if s.empty else (float(s.n.iloc[0]), float(s.median_time.iloc[0]))


def plot_scaling(df: pd.DataFrame, logy: bool = True,
                 ref_algos: tuple[str, str] = ("naive", "dc_standard")) -> Figure:
    """Waktu vs N per algoritma (log-log), satu panel per distribusi data.

    Garis putus-putus: acuan teoritis O(n^2) dan O(n log n), dijangkar di titik terkecil
    algoritma `ref_algos` (naive dan DnC).
    """
    dists = list(dict.fromkeys(df["dist"]))
    fig, axes = plt.subplots(1, len(dists), figsize=(5.2 * len(dists), 4.6), sharey=True, squeeze=False)
    for ax, dist in zip(axes[0], dists):
        sub = df[df["dist"] == dist]
        for algo in dict.fromkeys(sub.algorithm):
            s = sub[sub.algorithm == algo].sort_values("n")
            # batas bawah dijepit agar tetap positif di skala log
            lower = np.minimum(s.std_time, 0.8 * s.median_time)
            ax.errorbar(s.n, s.median_time, yerr=[lower, s.std_time], marker="o", ms=4, lw=1.6,
                        color=_c(algo), label=_l(algo), capsize=2, elinewidth=0.8, alpha=0.9)
        nmin, nmax = sub.n.min(), sub.n.max()
        xs = np.geomspace(nmin, nmax, 50)
        a = _anchor(sub, ref_algos[0])
        if a:
            ax.plot(xs, a[1] * (xs / a[0]) ** 2, ":", color="#c0392b", alpha=0.7, label="O(n²) acuan")
        b = _anchor(sub, ref_algos[1])
        if b:
            ax.plot(xs, b[1] * (xs * np.log(xs)) / (b[0] * math.log(b[0])), ":", color="#2471a3",
                    alpha=0.7, label="O(n log n) acuan")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Jumlah atom N"); ax.set_title(f"Distribusi: {dist}")
    axes[0][0].set_ylabel("Waktu median (detik)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_speedup(df: pd.DataFrame, numerator: str = "naive", denominator: str = "dc_standard") -> Figure:
    """Bar chart rasio waktu numerator/denominator per ukuran (analog rasio efisiensi paper)."""
    piv = df.pivot_table(index=["dist", "n"], columns="algorithm", values="median_time")
    if numerator not in piv or denominator not in piv:
        raise ValueError("df harus memuat kedua algoritma")
    ratio = (piv[numerator] / piv[denominator]).dropna().rename("ratio").reset_index()
    dists = list(dict.fromkeys(ratio["dist"]))
    ns = sorted(ratio.n.unique())
    w = 0.8 / len(dists)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k, dist in enumerate(dists):
        r = ratio[ratio["dist"] == dist].set_index("n").reindex(ns)["ratio"]
        ax.bar(np.arange(len(ns)) + k * w, r.values, w, label=dist, color=DIST_COLORS.get(dist))
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_xticks(np.arange(len(ns)) + w * (len(dists) - 1) / 2, [str(n) for n in ns])
    ax.set_xlabel("Jumlah atom N"); ax.set_ylabel(f"Rasio waktu {_l(numerator)} / {_l(denominator)}")
    ax.set_title("Rasio efisiensi (>1: DnC lebih cepat)"); ax.legend(title="Distribusi")
    return fig


def plot_greedy_convergence(acc_df: pd.DataFrame, dc_time: float | None = None) -> Figure:
    """Galat relatif greedy vs jumlah pengulangan (analog Fig. 6), panel kanan: waktu."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    r = acc_df.repeats
    m, s = acc_df.mean_rel_error_pct, acc_df.std_rel_error_pct
    a1.plot(r, m, "o-", color=_c("greedy_rep"), label="Galat relatif rata-rata")
    a1.fill_between(r, np.maximum(m - s, 0), m + s, color=_c("greedy_rep"), alpha=0.2)
    a1.set_xscale("log"); a1.set_xlabel("Jumlah pengulangan"); a1.set_ylabel("Galat relatif (%)")
    a1.set_title("Akurasi greedy vs pengulangan")
    ax2 = a1.twinx(); ax2.grid(False)
    ax2.plot(r, acc_df.frac_exact * 100, "s--", color="#1e8449", label="% percobaan eksak")
    ax2.set_ylabel("% percobaan tepat optimal", color="#1e8449"); ax2.set_ylim(-3, 103)
    a1.legend(loc="upper center"); ax2.legend(loc="center right")
    a2.plot(r, acc_df.mean_time, "o-", color=_c("greedy_rep"), label="Greedy berulang")
    if dc_time is not None:
        a2.axhline(dc_time, color=_c("dc_standard"), ls="--", label=f"DnC ({dc_time:.3f} s)")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("Jumlah pengulangan"); a2.set_ylabel("Waktu rata-rata (detik)")
    a2.set_title("Biaya waktu"); a2.legend()
    fig.tight_layout()
    return fig


def plot_split_ratio(df: pd.DataFrame) -> Figure:
    """Waktu DnC vs rasio pembagian kiri:kanan per ukuran data (analog Fig. 8)."""
    ratios = list(dict.fromkeys(df.ratio))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in sorted(df.n.unique()):
        s = df[df.n == n].set_index("ratio").reindex(ratios)
        ax.errorbar(range(len(ratios)), s.median_time, yerr=s.std_time, marker="o", capsize=2, label=f"N = {n}")
    ax.set_xticks(range(len(ratios)), ratios)
    ax.set_xlabel("Rasio pembagian (kiri:kanan)"); ax.set_ylabel("Waktu median (detik)")
    ax.set_title("Pengaruh rasio pembagian pada DnC"); ax.legend()
    return fig


def plot_strip_sizes(stats_dict: dict[str, dict]) -> Figure:
    """Ukuran strip per level rekursi (0 = akar) untuk beberapa distribusi data.

    Kiri: rata-rata (garis) dan min-max (area) ukuran strip. Kanan: strip / ukuran subset.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, st in stats_dict.items():
        lv = st["strip_by_level"]
        levels = sorted(lv)
        mean = np.array([lv[k]["mean"] for k in levels])
        lo = np.array([lv[k]["min"] for k in levels])
        hi = np.array([lv[k]["max"] for k in levels])
        frac = mean / np.array([lv[k]["mean_subset_size"] for k in levels])
        col = DIST_COLORS.get(name)
        a1.plot(levels, np.maximum(mean, 0.5), "o-", ms=3, color=col, label=name)
        a1.fill_between(levels, np.maximum(lo, 0.5), np.maximum(hi, 0.5), color=col, alpha=0.15)
        a2.plot(levels, frac, "o-", ms=3, color=col, label=name)
    a1.set_yscale("log"); a1.set_xlabel("Level rekursi"); a1.set_ylabel("Ukuran strip (titik)")
    a1.set_title("Ukuran strip per level"); a1.legend()
    a2.set_xlabel("Level rekursi"); a2.set_ylabel("Strip / ukuran subset")
    a2.set_title("Porsi subset yang masuk strip"); a2.legend()
    fig.tight_layout()
    return fig


def plot_correctness_table(df: pd.DataFrame) -> Figure:
    """Tabel benar/salah berwarna (hijau benar, merah salah)."""
    show = df.copy()
    show["distance"] = show["distance"].map("{:.5f}".format)
    show["abs_diff"] = show["abs_diff"].map("{:.2e}".format)
    show["correct"] = show["correct"].map({True: "BENAR", False: "SALAH"})
    cols = [c for c in ("dataset", "algorithm", "distance", "abs_diff", "correct") if c in show]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(show) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=show[cols].values, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.3)
    ci = cols.index("correct")
    for i, ok in enumerate(show["correct"], start=1):
        tbl[i, ci].set_facecolor("#abebc6" if ok == "BENAR" else "#f5b7b1")
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#d6eaf8")
    ax.set_title("Verifikasi kebenaran terhadap KD-tree")
    return fig
