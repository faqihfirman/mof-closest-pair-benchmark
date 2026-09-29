"""Chart 2D interaktif: proyeksi DnC, histogram, benchmark, skala, greedy, strip, dan rasio."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import (ACCENT, ALERT, BODY_TEXT, BRIGHT_TEXT, DIST_COLORS, INACTIVE_ATOM, MUTED_TEXT, OK, TRANSPARENT,
                    VIOLET, WARN, color, label, rgba, style, style_subplot_titles)

CORAL = "#fb7185"
NEUTRAL_POINT = "#a1a1aa"


def dnc_step_2d(points: np.ndarray, sorted_to_original: np.ndarray, merge_step: dict,
                height: int = 600) -> go.Figure:
    """Satu tahap gabung DnC dalam proyeksi XY."""
    points_array = np.asarray(points)
    sorted_points = points_array[sorted_to_original]
    start, end, split_index = merge_step["lo"], merge_step["hi"], merge_step["mid"]
    outside_subset = np.ones(len(sorted_points), dtype=bool)
    outside_subset[start:end] = False
    x_mid, strip_half_width = merge_step["x_mid"], merge_step["d_children"]

    fig = go.Figure()
    fig.add_vrect(x0=x_mid - strip_half_width, x1=x_mid + strip_half_width, fillcolor=rgba(WARN, 0.10),
                  line=dict(color=rgba(WARN, 0.5), width=1))
    fig.add_vline(x=x_mid, line=dict(color=BRIGHT_TEXT, dash="dash", width=1))
    faded = sorted_points[outside_subset]
    fig.add_trace(go.Scatter(x=faded[:, 0], y=faded[:, 1], mode="markers", name="Di luar subset",
                             marker=dict(size=5, color=INACTIVE_ATOM), hoverinfo="skip"))
    for index_range, side_name, side_color in ((slice(start, split_index), "Kiri", ACCENT),
                                               (slice(split_index, end), "Kanan", OK)):
        side_points = sorted_points[index_range]
        fig.add_trace(go.Scatter(x=side_points[:, 0], y=side_points[:, 1], mode="markers",
                                 name=f"{side_name} · {len(side_points)}",
                                 marker=dict(size=8, color=side_color, line=dict(width=0))))
    strip_points = sorted_points[merge_step["strip_lo"]:merge_step["strip_hi"]]
    if len(strip_points):
        fig.add_trace(go.Scatter(x=strip_points[:, 0], y=strip_points[:, 1], mode="markers",
                                 name=f"Di strip · {len(strip_points)}",
                                 marker=dict(size=14, color=TRANSPARENT, line=dict(color=WARN, width=1.5)),
                                 hoverinfo="skip"))
    point_a, point_b = points_array[list(merge_step["pair"])]
    fig.add_trace(go.Scatter(x=[point_a[0], point_b[0]], y=[point_a[1], point_b[1]], mode="lines+markers",
                             name="Terbaik saat ini", line=dict(color=ALERT, width=3), marker=dict(size=9, color=ALERT)))
    fig.update_yaxes(scaleanchor="x", title_text="y (Å) · proyeksi XY")
    fig.update_xaxes(title_text="x (Å)")
    return style(fig, height, legend_bottom=True)


def demo_step_2d(points: dict[str, tuple[float, float]], algo: str, state: dict, height: int = 460) -> go.Figure:
    """Scatter 2D dataset demo (P1..P7) pada satu langkah, gaya seragam lintas tab.

    `state` = keluaran `utils.demo_2d.state_at()`: pasangan yang dibandingkan, terbaik saat
    ini, grup kiri/kanan (DnC), garis pembagi + strip (DnC), titik `current` (Greedy).
    """
    ids = list(points)
    xs = [points[p][0] for p in ids]
    ys = [points[p][1] for p in ids]
    compare_pair = set(state["compare_pair"] or [])
    skip_points = set(state["skip_points"] or [])
    best_pair = set(state["best_pair"] or [])
    group = set(state["group"]) if state["group"] else None
    left = set(state["left"]) if state["left"] else None
    remaining = set(state["remaining"]) if state["remaining"] is not None else None
    current = state["current"]

    fig = go.Figure()
    if algo == "dnc" and state["strip"] is not None:
        lo, hi = state["strip"]
        fig.add_vrect(x0=lo, x1=hi, fillcolor=rgba(WARN, 0.10), line=dict(color=rgba(WARN, 0.5), width=1))
    if algo == "dnc" and state["divider"] is not None:
        fig.add_vline(x=state["divider"], line=dict(color="#52525b", dash="dash", width=1.5))

    marker_colors, marker_lines, opacities = [], [], []
    for pid in ids:
        if pid in best_pair:
            marker_colors.append(OK)
        elif pid in compare_pair:
            marker_colors.append(WARN)
        elif algo == "dnc" and left is not None:
            marker_colors.append(ACCENT if pid in left else CORAL)
        else:
            marker_colors.append(NEUTRAL_POINT)
        marker_lines.append(ALERT if pid == current else "rgba(0,0,0,0)")
        if algo == "greedy" and remaining is not None:
            opacities.append(1.0 if (pid == current or pid in remaining or pid in best_pair) else 0.3)
        elif algo == "dnc" and group is not None:
            opacities.append(1.0 if pid in group or pid in best_pair else 0.3)
        else:
            opacities.append(1.0)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=ids, textposition="top center",
        textfont=dict(size=12, color=BRIGHT_TEXT),
        marker=dict(size=22, color=marker_colors, opacity=opacities,
                    line=dict(width=3, color=marker_lines)),
        hovertemplate="%{text} (%{x}, %{y})<extra></extra>", showlegend=False))

    if state["best_pair"]:
        a, b = state["best_pair"]
        fig.add_trace(go.Scatter(x=[points[a][0], points[b][0]], y=[points[a][1], points[b][1]],
                                 mode="lines", line=dict(color=OK, width=3), hoverinfo="skip", showlegend=False))
    if state["compare_pair"] and len(state["compare_pair"]) == 2:
        a, b = state["compare_pair"]
        fig.add_trace(go.Scatter(x=[points[a][0], points[b][0]], y=[points[a][1], points[b][1]],
                                 mode="lines", line=dict(color=WARN, width=2, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
    if skip_points and current:
        fig.add_trace(go.Scatter(x=[points[current][0]], y=[points[current][1]], mode="markers",
                                 marker=dict(size=32, color=TRANSPARENT, line=dict(color=MUTED_TEXT, width=2, dash="dot")),
                                 hoverinfo="skip", showlegend=False))

    fig.update_xaxes(title_text="x", range=[0, 10])
    fig.update_yaxes(title_text="y", range=[0, 7], scaleanchor="x")
    return style(fig, height, show_legend=False)


def d_timeline(merge_steps: list[dict], current_step: int) -> go.Figure:
    """Nilai d setelah tiap tahap gabung (post-order); titik merah = strip memperbaiki d."""
    step_numbers = np.arange(1, len(merge_steps) + 1)
    distance_after = np.array([step["d_after"] for step in merge_steps])
    strip_improved = np.array([step["d_after"] < step["d_children"] - 1e-12 for step in merge_steps])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_numbers, y=distance_after, mode="lines", name="d setelah gabung",
                             line=dict(color=ACCENT, width=1.5, shape="hv"),
                             fill="tozeroy", fillcolor=rgba(ACCENT, 0.08),
                             hovertemplate="tahap %{x}<br>d = %{y:.3f} Å<extra></extra>"))
    fig.add_trace(go.Scatter(x=step_numbers[strip_improved], y=distance_after[strip_improved], mode="markers",
                             name="strip memperbaiki d", marker=dict(color=ALERT, size=7),
                             hovertemplate="tahap %{x}<extra></extra>"))
    fig.add_vline(x=current_step, line=dict(color=WARN, width=1.5))
    fig.update_xaxes(title_text="tahap penggabungan")
    fig.update_yaxes(type="log", title_text="d (Å)")
    return style(fig, 230)


# ---------------------------------------------------------------- 2D
def nn_histogram(nn_distance: np.ndarray, threshold: float, height: int = 240) -> go.Figure:
    """Histogram jarak tetangga terdekat dengan garis threshold."""
    fig = go.Figure(go.Histogram(x=nn_distance, nbinsx=60, marker=dict(color=ACCENT, line=dict(width=0)),
                                 opacity=0.75, hovertemplate="%{x} Å<br>%{y} atom<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=ALERT, dash="dash", width=1.5),
                  annotation_text=f"threshold {threshold:g} Å", annotation_position="top right",
                  annotation_font=dict(color=ALERT, size=10))
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title_text="jarak tetangga terdekat (Å)")
    fig.update_yaxes(title_text="atom")
    return style(fig, height, show_legend=False)


def compare_bars(timing_table: pd.DataFrame) -> go.Figure:
    """Bar horizontal waktu per algoritma. Kolom: key, algoritma, time, correct."""
    fastest_first = timing_table.sort_values("time")
    bar_text = [f"{seconds * 1e3:,.2f} ms · {'benar' if is_correct else 'salah'}"
                for seconds, is_correct in zip(fastest_first["time"], fastest_first["correct"])]
    fig = go.Figure(go.Bar(
        y=fastest_first["algoritma"], x=fastest_first["time"], orientation="h",
        marker=dict(color=[color(key) for key in fastest_first["key"]], line=dict(width=0)),
        text=bar_text, textposition="outside", cliponaxis=False, textfont=dict(size=10, color=BRIGHT_TEXT),
        hovertemplate="%{y}<br>%{x:.5f} s<extra></extra>"))
    fig.update_xaxes(type="log", title_text="waktu median (s, log)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = style(fig, 90 + 38 * len(fastest_first), show_legend=False)
    fig.update_layout(margin=dict(l=120, r=120))  # ruang untuk label di ujung bar
    return fig


def _first_measurement(results: pd.DataFrame, algo_key: str) -> tuple[float, float] | None:
    """(N terkecil, waktu di N itu) untuk menjangkar garis acuan teoritis."""
    rows = results[results.algorithm == algo_key].sort_values("n")
    return None if rows.empty else (float(rows.n.iloc[0]), float(rows.median_time.iloc[0]))


def scaling_chart(results: pd.DataFrame) -> go.Figure:
    """Waktu vs N (log-log) per distribusi, dengan acuan O(n²) dan O(n log n)."""
    distributions = list(dict.fromkeys(results["dist"]))
    fig = make_subplots(rows=1, cols=len(distributions), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=distributions)
    algos_in_legend: set[str] = set()
    for column, distribution in enumerate(distributions, start=1):
        dist_results = results[results["dist"] == distribution]
        n_grid = np.geomspace(dist_results.n.min(), dist_results.n.max(), 40)

        naive_anchor = _first_measurement(dist_results, "naive")
        if naive_anchor:
            anchor_n, anchor_time = naive_anchor
            fig.add_trace(go.Scatter(x=n_grid, y=anchor_time * (n_grid / anchor_n) ** 2, mode="lines",
                                     name="acuan O(n²)", line=dict(color=rgba(ALERT, 0.35), dash="dot", width=1),
                                     legendgroup="ref_n2", showlegend=column == 1, hoverinfo="skip"), 1, column)
        dnc_anchor = _first_measurement(dist_results, "dc_standard")
        if dnc_anchor:
            anchor_n, anchor_time = dnc_anchor
            nlogn_curve = anchor_time * n_grid * np.log(n_grid) / (anchor_n * math.log(anchor_n))
            fig.add_trace(go.Scatter(x=n_grid, y=nlogn_curve, mode="lines", name="acuan O(n log n)",
                                     line=dict(color=rgba(ACCENT, 0.35), dash="dot", width=1),
                                     legendgroup="ref_nlogn", showlegend=column == 1, hoverinfo="skip"), 1, column)

        for algo_key in dict.fromkeys(dist_results.algorithm):
            algo_rows = dist_results[dist_results.algorithm == algo_key].sort_values("n")
            # batas bawah error bar dijepit agar tetap positif di skala log
            lower_error = np.minimum(algo_rows.std_time, 0.8 * algo_rows.median_time)
            fig.add_trace(go.Scatter(
                x=algo_rows.n, y=algo_rows.median_time, mode="lines+markers", name=label(algo_key),
                legendgroup=algo_key, showlegend=algo_key not in algos_in_legend,
                line=dict(color=color(algo_key), width=2), marker=dict(size=6),
                error_y=dict(type="data", array=algo_rows.std_time, arrayminus=lower_error,
                             thickness=1, width=2, color=color(algo_key)),
                hovertemplate=f"{label(algo_key)}<br>N = %{{x}}<br>%{{y:.4f}} s<extra></extra>"), 1, column)
            algos_in_legend.add(algo_key)
    style_subplot_titles(fig, len(distributions))
    fig.update_xaxes(type="log", title_text="N atom")
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="waktu median (s)", row=1, col=1)
    return style(fig, 440, legend_bottom=True)


def speedup_chart(results: pd.DataFrame, numerator: str = "naive",
                  denominator: str = "dc_standard") -> go.Figure | None:
    """Rasio waktu numerator/denominator per N (analog rasio efisiensi paper)."""
    time_table = results.pivot_table(index=["dist", "n"], columns="algorithm", values="median_time")
    if numerator not in time_table or denominator not in time_table:
        return None
    speedup = (time_table[numerator] / time_table[denominator]).dropna().rename("ratio").reset_index()
    sizes = sorted(speedup.n.unique())
    fig = go.Figure()
    for distribution in dict.fromkeys(speedup["dist"]):
        ratios = speedup[speedup["dist"] == distribution].set_index("n").reindex(sizes)["ratio"]
        fig.add_trace(go.Bar(x=[str(n) for n in sizes], y=ratios.values, name=distribution,
                             marker_color=DIST_COLORS.get(distribution, ACCENT), marker_line_width=0,
                             hovertemplate=f"{distribution}<br>N = %{{x}}<br>%{{y:.1f}}×<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=MUTED_TEXT, dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(type="category", title_text="N atom")
    fig.update_yaxes(title_text="× lebih cepat")
    return style(fig, 340)


def greedy_chart(accuracy: pd.DataFrame, dnc_time: float | None = None) -> go.Figure:
    """Galat relatif greedy vs pengulangan (analog Fig. 6) dan biaya waktunya."""
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {}]], horizontal_spacing=0.1,
                        subplot_titles=("Galat relatif terhadap pengulangan", "Biaya waktu"))
    repeats = accuracy.repeats
    mean_error = accuracy.mean_rel_error_pct
    std_error = accuracy.std_rel_error_pct
    # Pita ±1 simpangan baku: garis atas, lalu garis bawah diisi sampai garis atas.
    fig.add_trace(go.Scatter(x=repeats, y=mean_error + std_error, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=repeats, y=np.maximum(mean_error - std_error, 0), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(WARN, 0.10), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=repeats, y=mean_error, mode="lines+markers", name="galat rata-rata (%)",
                             line=dict(color=WARN, width=2), hovertemplate="%{x}× → %{y:.1f}%<extra></extra>"),
                  1, 1, secondary_y=False)
    fig.add_trace(go.Scatter(x=repeats, y=accuracy.frac_exact * 100, mode="lines+markers", name="% percobaan eksak",
                             line=dict(color=OK, dash="dot", width=2), marker=dict(symbol="square"),
                             hovertemplate="%{x}× → %{y:.0f}% eksak<extra></extra>"), 1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=repeats, y=accuracy.mean_time, mode="lines+markers", name="waktu greedy",
                             line=dict(color=BODY_TEXT, width=2), hovertemplate="%{x}× → %{y:.4f} s<extra></extra>"),
                  1, 2)
    style_subplot_titles(fig, 2)
    if dnc_time is not None:
        fig.add_hline(y=dnc_time, line=dict(color=ACCENT, dash="dash", width=1.5), row=1, col=2,
                      annotation_text=f"DnC {dnc_time * 1e3:.1f} ms", annotation_font=dict(color=ACCENT, size=10))
    fig.update_xaxes(type="log", title_text="pengulangan")
    fig.update_yaxes(title_text="galat (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="% eksak", range=[-3, 103], row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(type="log", title_text="detik", row=1, col=2)
    return style(fig, 380, legend_bottom=True)


def strip_chart(stats_by_distribution: dict[str, dict]) -> go.Figure:
    """Ukuran strip per level rekursi dan porsinya terhadap ukuran subset."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("Ukuran strip per level (min–maks)", "Porsi subset yang masuk strip"))
    for distribution, stats in stats_by_distribution.items():
        per_level = stats["strip_by_level"]
        levels = sorted(per_level)
        # minimal 0.5 agar tetap tampil di skala log (strip kosong = 0 titik)
        mean_strip = np.maximum([per_level[lv]["mean"] for lv in levels], 0.5)
        min_strip = np.maximum([per_level[lv]["min"] for lv in levels], 0.5)
        max_strip = np.maximum([per_level[lv]["max"] for lv in levels], 0.5)
        strip_fraction = np.array([per_level[lv]["mean"] / per_level[lv]["mean_subset_size"] for lv in levels])
        line_color = DIST_COLORS.get(distribution, ACCENT)
        fig.add_trace(go.Scatter(x=levels, y=max_strip, mode="lines", line=dict(width=0), legendgroup=distribution,
                                 showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=levels, y=min_strip, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=rgba(line_color, 0.12), legendgroup=distribution, showlegend=False,
                                 hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=levels, y=mean_strip, mode="lines+markers", name=distribution,
                                 legendgroup=distribution, line=dict(color=line_color, width=2), marker=dict(size=5),
                                 hovertemplate=f"{distribution}<br>level %{{x}}<br>rata-rata %{{y:.1f}} titik<extra></extra>"),
                      1, 1)
        fig.add_trace(go.Scatter(x=levels, y=strip_fraction, mode="lines+markers", name=distribution,
                                 legendgroup=distribution, showlegend=False,
                                 line=dict(color=line_color, width=2), marker=dict(size=5),
                                 hovertemplate=f"{distribution}<br>level %{{x}}<br>%{{y:.0%}}<extra></extra>"), 1, 2)
    style_subplot_titles(fig, 2)
    fig.update_xaxes(title_text="level rekursi (0 = akar)")
    fig.update_yaxes(type="log", title_text="titik di strip", row=1, col=1)
    fig.update_yaxes(title_text="strip / subset", tickformat=".0%", row=1, col=2)
    return style(fig, 380, legend_bottom=True)


def ratio_chart(results: pd.DataFrame) -> go.Figure:
    """Waktu DnC vs rasio pembagian kiri:kanan (analog Fig. 8)."""
    ratio_labels = list(dict.fromkeys(results.ratio))
    sizes = sorted(results.n.unique())
    line_palette = [ACCENT, OK, WARN, ALERT, VIOLET]
    fig = go.Figure()
    for position, size in enumerate(sizes):
        size_rows = results[results.n == size].set_index("ratio").reindex(ratio_labels)
        line_color = line_palette[position % len(line_palette)]
        fig.add_trace(go.Scatter(x=ratio_labels, y=size_rows.median_time, mode="lines+markers", name=f"N = {size}",
                                 line=dict(color=line_color, width=2), marker=dict(size=7),
                                 error_y=dict(type="data", array=size_rows.std_time, thickness=1, width=2,
                                              color=line_color),
                                 hovertemplate=f"N = {size}<br>rasio %{{x}}<br>%{{y:.4f}} s<extra></extra>"))
    fig.update_xaxes(type="category", title_text="rasio pembagian kiri:kanan")
    fig.update_yaxes(title_text="waktu median (s)")
    return style(fig, 360)
