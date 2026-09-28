"""Chart Plotly interaktif bertema gelap (gaya command-center) untuk dashboard Streamlit."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- palet (gaya shadcn/ui: zinc + aksen Tailwind)
TRANSPARENT = "rgba(0,0,0,0)"
PANEL_COLOR = "#ffffff"
GRID_COLOR = "#f4f4f5"
AXIS_COLOR = "#e4e4e7"
MUTED_TEXT = "#a1a1aa"
BODY_TEXT = "#71717a"
BRIGHT_TEXT = "#09090b"  # teks utama
ACCENT = "#2563eb"       # blue-600: warna data utama
ALERT = "#ef4444"        # merah: pasangan terdekat / pelanggaran
WARN = "#f59e0b"         # oranye: strip / peringatan
OK = "#16a34a"           # hijau
VIOLET = "#7c3aed"
FONT_FAMILY = "Poppins, ui-sans-serif, system-ui, sans-serif"

ALGO_COLORS = {
    "naive": ALERT, "naive_vec": WARN, "greedy_1x": "#eab308", "greedy_rep": "#a16207",
    "dc_standard": ACCENT, "dc_paper": VIOLET, "kdtree": OK,
}
ALGO_LABELS = {
    "naive": "Naive", "naive_vec": "Naive (NumPy)", "greedy_1x": "Greedy",
    "greedy_rep": "Greedy berulang", "dc_standard": "Divide & Conquer", "dc_paper": "DnC (strip paper)",
    "kdtree": "KD-tree",
}
DIST_COLORS = {"uniform": ACCENT, "clustered": WARN, "lattice_jitter": OK}
ELEMENT_COLORS = {
    "H": "#38bdf8", "C": "#0d9488", "N": "#4f46e5", "O": "#ef4444", "F": "#22c55e", "B": "#f97316",
    "Si": "#eab308", "Al": "#a855f7", "V": "#ec4899", "Cu": "#d97706", "In": "#e11d48",
}
ELEMENT_SIZES = {"H": 2.0, "C": 3.0, "N": 3.2, "O": 3.2, "F": 3.0, "B": 3.2}
METAL_SIZE = 6.0  # elemen yang tidak ada di ELEMENT_SIZES (logam) digambar lebih besar
# Warna jarak tetangga: dekat = merah/oranye (bahaya), jauh = biru tua yang tenang.
# Transisi oranye -> biru dibuat sangat pendek: campuran RGB keduanya menghasilkan cokelat keabuan.
NEAREST_NEIGHBOR_SCALE = [[0.0, ALERT], [0.2, WARN], [0.23, "#93c5fd"], [0.55, "#3b82f6"], [1.0, "#1e3a8a"]]
BACKGROUND_ATOM = "rgba(59,130,246,0.75)"   # atom latar (greedy): biru muda, bukan abu-abu
INACTIVE_ATOM = "rgba(165,180,252,0.85)"    # atom di luar subset (DnC): lavender
# scrollZoom mati: scroll halaman tidak boleh "dibajak" chart (zoom lewat toolbar / pinch).
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
# 3D: scroll / pinch untuk zoom masuk ke struktur (juga saat fullscreen).
PLOTLY_CONFIG_3D = {**PLOTLY_CONFIG, "scrollZoom": True}


def label(algo_key: str) -> str:
    return ALGO_LABELS.get(algo_key, algo_key)


def color(algo_key: str) -> str:
    return ALGO_COLORS.get(algo_key, BODY_TEXT)


def _rgba(hex_color: str, alpha: float) -> str:
    """'#22d3ee' + alpha -> 'rgba(34,211,238,alpha)'."""
    hex_digits = hex_color.lstrip("#")
    red, green, blue = (int(hex_digits[offset:offset + 2], 16) for offset in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


MODEBAR_STYLE = dict(bgcolor="rgba(255,255,255,0.9)", color="#a1a1aa", activecolor=BRIGHT_TEXT)


def _hover_style() -> dict:
    return dict(bgcolor=PANEL_COLOR, bordercolor=AXIS_COLOR,
                font=dict(family=FONT_FAMILY, color=BRIGHT_TEXT, size=11))


def style(fig: go.Figure, height: int = 360, title: str | None = None,
          show_legend: bool = True, legend_bottom: bool = False) -> go.Figure:
    """Tema gelap transparan untuk chart 2D."""
    title_settings = {}
    if title:
        title_settings["title"] = dict(text=title, font=dict(size=14, color=BRIGHT_TEXT),
                                       x=0.005, xanchor="left", y=0.97)
    if legend_bottom:
        legend_position = dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    else:
        legend_position = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    fig.update_layout(
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT, height=height, showlegend=show_legend,
        font=dict(family=FONT_FAMILY, size=12, color=BODY_TEXT),
        margin=dict(l=56, r=20, t=50 if title else 20, b=48),
        legend=dict(bgcolor=TRANSPARENT, font=dict(size=12, color=BRIGHT_TEXT), **legend_position),
        hoverlabel=_hover_style(), modebar=MODEBAR_STYLE, **title_settings,
    )
    axis_style = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=AXIS_COLOR, automargin=True,
                      tickfont=dict(size=11), title_font=dict(size=12, color=BODY_TEXT), title_standoff=8)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _style_subplot_titles(fig: go.Figure, title_count: int) -> None:
    """Judul subplot dibuat make_subplots sebagai anotasi pertama; samakan gayanya."""
    for annotation in fig.layout.annotations[:title_count]:
        annotation.font = dict(size=12, color=BODY_TEXT, family=FONT_FAMILY)


# ---------------------------------------------------------------- 3D
def _axis_3d(title: str) -> dict:
    return dict(title=dict(text=title, font=dict(size=10, color=MUTED_TEXT)), showbackground=False,
                gridcolor="#e4e4e7", zeroline=False, showspikes=False,
                color=MUTED_TEXT, tickfont=dict(size=9), linecolor=AXIS_COLOR)


def style_3d(fig: go.Figure, height: int = 640) -> go.Figure:
    """Scene 3D tanpa latar: sumbu tipis, kamera orbit, sudut kamera dipertahankan antar-rerun."""
    fig.update_layout(
        paper_bgcolor=TRANSPARENT, height=height, margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",  # jangan reset kamera saat slider digeser
        modebar=MODEBAR_STYLE,
        font=dict(family=FONT_FAMILY, size=12, color=BODY_TEXT), hoverlabel=_hover_style(),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7", borderwidth=1, x=0.015, y=0.975,
                    font=dict(size=12, color=BRIGHT_TEXT), itemsizing="constant"),
        scene=dict(bgcolor=TRANSPARENT, aspectmode="data", dragmode="orbit",
                   xaxis=_axis_3d("x (Å)"), yaxis=_axis_3d("y (Å)"), zaxis=_axis_3d("z (Å)"),
                   camera=dict(eye=dict(x=1.45, y=1.45, z=0.85))),
    )
    return fig


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Jarak tiap atom ke tetangga terdekatnya (kolom 0 hasil query = atom itu sendiri)."""
    neighbor_distances, _ = cKDTree(points).query(points, k=2)
    return neighbor_distances[:, 1]


def _line_segments(points: np.ndarray, index_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gabungkan banyak segmen garis jadi satu trace; antar-segmen dipisah NaN agar tidak tersambung."""
    if len(index_pairs) == 0:
        empty = np.array([])
        return empty, empty, empty
    # Tiap segmen = 3 baris: titik awal, titik akhir, NaN (pemutus garis).
    segment_rows = np.full((len(index_pairs), 3, 3), np.nan)
    segment_rows[:, 0] = points[index_pairs[:, 0]]
    segment_rows[:, 1] = points[index_pairs[:, 1]]
    segment_rows = segment_rows.reshape(-1, 3)
    return segment_rows[:, 0], segment_rows[:, 1], segment_rows[:, 2]


def _highlighted_pair_traces(points: np.ndarray, pair: tuple[int, int], name: str, pair_color: str) -> list:
    """Pasangan disorot: halo lembut, garis tebal, label jarak di tengah."""
    point_a, point_b = points[pair[0]], points[pair[1]]
    distance = float(np.linalg.norm(point_a - point_b))
    xs, ys, zs = [point_a[0], point_b[0]], [point_a[1], point_b[1]], [point_a[2], point_b[2]]
    midpoint = (point_a + point_b) / 2
    halo = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=26, color=pair_color, opacity=0.12),
                        hoverinfo="skip", showlegend=False)
    link = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", name=f"{name} · {distance:.3f} Å",
                        line=dict(color=pair_color, width=9),
                        marker=dict(size=7, color=pair_color, line=dict(color="#ffffff", width=1.5)),
                        customdata=[pair[0], pair[1]],
                        hovertemplate=f"<b>{name}</b><br>atom #%{{customdata}}<br>d = {distance:.4f} Å<extra></extra>")
    distance_label = go.Scatter3d(x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]], mode="text",
                                  text=[f"   {distance:.3f} Å"], textposition="middle right",
                                  textfont=dict(color=pair_color, size=12, family=FONT_FAMILY),
                                  hoverinfo="skip", showlegend=False)
    return [halo, link, distance_label]


def atoms_3d(points: np.ndarray, pair: tuple[int, int] | None = None, symbols: list[str] | None = None,
             color_mode: str = "nn", bond_radius: float | None = None, threshold: float | None = None,
             size_scale: float = 1.0, height: int = 640) -> go.Figure:
    """Model atom 3D.

    color_mode: "nn" (warna = jarak tetangga terdekat), "element" (butuh symbols), "flat".
    bond_radius: gambar ikatan/tautan antar-atom yang lebih dekat dari nilai ini.
    threshold: atom yang tetangganya lebih dekat dari threshold diberi cincin peringatan.
    """
    points_array = np.asarray(points, dtype=float)
    nn_distance = nearest_neighbor_distances(points_array)
    atom_indices = np.arange(len(points_array))
    fig = go.Figure()

    if bond_radius:
        bonded_pairs = cKDTree(points_array).query_pairs(bond_radius, output_type="ndarray")
        bond_x, bond_y, bond_z = _line_segments(points_array, bonded_pairs)
        fig.add_trace(go.Scatter3d(x=bond_x, y=bond_y, z=bond_z, mode="lines", name="Ikatan",
                                   line=dict(color="rgba(72,72,74,0.28)", width=2), hoverinfo="skip"))

    hover_text = "atom #%{customdata[0]}<br>d_nn = %{customdata[1]:.3f} Å<extra></extra>"
    if color_mode == "element" and symbols is not None:
        symbol_array = np.asarray(symbols)
        elements, element_counts = np.unique(symbol_array, return_counts=True)
        for element in elements[np.argsort(-element_counts)]:  # elemen terbanyak di atas legend
            is_element = symbol_array == element
            fig.add_trace(go.Scatter3d(
                x=points_array[is_element, 0], y=points_array[is_element, 1], z=points_array[is_element, 2],
                mode="markers", name=f"{element} · {int(is_element.sum())}",
                marker=dict(size=ELEMENT_SIZES.get(element, METAL_SIZE) * size_scale,
                            color=ELEMENT_COLORS.get(element, "#cbd5e1"), opacity=0.92, line=dict(width=0)),
                customdata=np.c_[atom_indices[is_element], nn_distance[is_element]],
                hovertemplate=f"<b>{element}</b> " + hover_text))
    else:
        marker_style = dict(size=3.2 * size_scale, opacity=0.9, line=dict(width=0))
        if color_mode == "flat":
            marker_style["color"] = "#3b82f6"
        else:
            marker_style.update(
                color=nn_distance, colorscale=NEAREST_NEIGHBOR_SCALE,
                cmin=float(nn_distance.min()), cmax=float(np.percentile(nn_distance, 95)),
                colorbar=dict(title=dict(text="d_nn (Å)", font=dict(size=10, color=MUTED_TEXT)), thickness=8,
                              len=0.5, x=0.98, tickfont=dict(size=9, color=MUTED_TEXT), outlinewidth=0))
        fig.add_trace(go.Scatter3d(x=points_array[:, 0], y=points_array[:, 1], z=points_array[:, 2],
                                   mode="markers", name="Atom", marker=marker_style,
                                   customdata=np.c_[atom_indices, nn_distance], hovertemplate=hover_text))

    if threshold is not None:
        is_too_close = nn_distance < threshold
        flagged_count = int(is_too_close.sum())
        if flagged_count:
            # Banyak atom ditandai -> cincin kecil & tipis agar model tetap terlihat.
            ring_size, ring_opacity = (12, 0.9) if flagged_count <= 150 else (6, 0.35)
            fig.add_trace(go.Scatter3d(
                x=points_array[is_too_close, 0], y=points_array[is_too_close, 1], z=points_array[is_too_close, 2],
                mode="markers", name=f"d_nn < {threshold:g} Å · {flagged_count}",
                marker=dict(size=ring_size, symbol="circle-open", color=WARN, opacity=ring_opacity),
                hoverinfo="skip"))

    if pair is not None:
        fig.add_traces(_highlighted_pair_traces(points_array, pair, "Pasangan terdekat", ALERT))
    return style_3d(fig, height)


def dnc_step_3d(points: np.ndarray, sorted_to_original: np.ndarray, merge_step: dict,
                height: int = 600) -> go.Figure:
    """Satu tahap gabung DnC di 3D: bidang pembagi x_mid, slab strip ±d, subset kiri/kanan.

    merge_step = satu elemen `trace` dari closest_pair_dc (posisi pada array terurut X).
    """
    points_array = np.asarray(points)
    sorted_points = points_array[sorted_to_original]
    start, end, split_index = merge_step["lo"], merge_step["hi"], merge_step["mid"]
    outside_subset = np.ones(len(sorted_points), dtype=bool)
    outside_subset[start:end] = False

    fig = go.Figure()
    faded = sorted_points[outside_subset]
    fig.add_trace(go.Scatter3d(x=faded[:, 0], y=faded[:, 1], z=faded[:, 2], mode="markers", name="Di luar subset",
                               marker=dict(size=2.5, color=INACTIVE_ATOM), hoverinfo="skip"))
    for index_range, side_name, side_color in ((slice(start, split_index), "Kiri", ACCENT),
                                               (slice(split_index, end), "Kanan", OK)):
        side_points = sorted_points[index_range]
        fig.add_trace(go.Scatter3d(x=side_points[:, 0], y=side_points[:, 1], z=side_points[:, 2], mode="markers",
                                   name=f"{side_name} · {len(side_points)}",
                                   marker=dict(size=5, color=side_color, opacity=0.95, line=dict(width=0)),
                                   hovertemplate=f"{side_name}<extra></extra>"))

    # Bidang vertikal x = konstan, selebar bounding box subset pada sumbu Y dan Z.
    subset_points = sorted_points[start:end]
    y_min, z_min = subset_points[:, 1:].min(axis=0) - 0.5
    y_max, z_max = subset_points[:, 1:].max(axis=0) + 0.5
    x_mid, strip_half_width = merge_step["x_mid"], merge_step["d_children"]

    def vertical_plane(x_position: float, plane_color: str, opacity: float, name: str, show_in_legend: bool):
        return go.Surface(x=[[x_position, x_position], [x_position, x_position]],
                          y=[[y_min, y_max], [y_min, y_max]], z=[[z_min, z_min], [z_max, z_max]],
                          colorscale=[[0, plane_color], [1, plane_color]], showscale=False, opacity=opacity,
                          hoverinfo="skip", name=name, showlegend=show_in_legend)

    fig.add_trace(vertical_plane(x_mid, BRIGHT_TEXT, 0.08, "Bidang pembagi x_mid", True))
    fig.add_trace(vertical_plane(x_mid - strip_half_width, WARN, 0.12, "Batas strip ±d", True))
    fig.add_trace(vertical_plane(x_mid + strip_half_width, WARN, 0.12, "Batas strip ±d", False))

    strip_points = sorted_points[merge_step["strip_lo"]:merge_step["strip_hi"]]
    if len(strip_points):
        fig.add_trace(go.Scatter3d(x=strip_points[:, 0], y=strip_points[:, 1], z=strip_points[:, 2],
                                   mode="markers", name=f"Di strip · {len(strip_points)}",
                                   marker=dict(size=10, symbol="circle-open", color=WARN), hoverinfo="skip"))
    fig.add_traces(_highlighted_pair_traces(points_array, merge_step["pair"], "Terbaik saat ini", ALERT))
    return style_3d(fig, height)


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
    fig.add_vrect(x0=x_mid - strip_half_width, x1=x_mid + strip_half_width, fillcolor=_rgba(WARN, 0.10),
                  line=dict(color=_rgba(WARN, 0.5), width=1))
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


def d_timeline(merge_steps: list[dict], current_step: int) -> go.Figure:
    """Nilai d setelah tiap tahap gabung (post-order); titik merah = strip memperbaiki d."""
    step_numbers = np.arange(1, len(merge_steps) + 1)
    distance_after = np.array([step["d_after"] for step in merge_steps])
    strip_improved = np.array([step["d_after"] < step["d_children"] - 1e-12 for step in merge_steps])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_numbers, y=distance_after, mode="lines", name="d setelah gabung",
                             line=dict(color=ACCENT, width=1.5, shape="hv"),
                             fill="tozeroy", fillcolor=_rgba(ACCENT, 0.08),
                             hovertemplate="tahap %{x}<br>d = %{y:.3f} Å<extra></extra>"))
    fig.add_trace(go.Scatter(x=step_numbers[strip_improved], y=distance_after[strip_improved], mode="markers",
                             name="strip memperbaiki d", marker=dict(color=ALERT, size=7),
                             hovertemplate="tahap %{x}<extra></extra>"))
    fig.add_vline(x=current_step, line=dict(color=WARN, width=1.5))
    fig.update_xaxes(title_text="tahap penggabungan")
    fig.update_yaxes(type="log", title_text="d (Å)")
    return style(fig, 230)


def greedy_path_3d(points: np.ndarray, visit_order: list[int], greedy_pair: tuple[int, int],
                   true_pair: tuple[int, int], height: int = 620) -> go.Figure:
    """Jalur kunjungan greedy (gradasi hijau→ungu) vs pasangan terdekat sebenarnya."""
    points_array = np.asarray(points)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points_array[:, 0], y=points_array[:, 1], z=points_array[:, 2], mode="markers",
                               name="Atom", marker=dict(size=3.5, color=BACKGROUND_ATOM), hoverinfo="skip"))
    path_points = points_array[visit_order]
    step_numbers = np.arange(len(path_points))
    path_gradient = [[0, OK], [1, VIOLET]]
    fig.add_trace(go.Scatter3d(x=path_points[:, 0], y=path_points[:, 1], z=path_points[:, 2], mode="lines+markers",
                               name="Jalur greedy", showlegend=False,
                               line=dict(color=step_numbers, colorscale=path_gradient, width=4),
                               marker=dict(size=2.5, color=step_numbers, colorscale=path_gradient),
                               customdata=step_numbers, hovertemplate="langkah %{customdata}<extra></extra>"))
    # Garis bergradasi tampil hitam di legend -> pakai entri legend pengganti berwarna.
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines+markers",
                               name=f"Jalur greedy · {len(path_points)} titik",
                               line=dict(color=OK, width=4), marker=dict(size=4, color=OK)))
    fig.add_trace(go.Scatter3d(x=[path_points[0, 0]], y=[path_points[0, 1]], z=[path_points[0, 2]], mode="markers",
                               name="Titik awal",
                               marker=dict(size=9, symbol="diamond", color=OK, line=dict(color="#ffffff", width=1.5))))
    fig.add_traces(_highlighted_pair_traces(points_array, greedy_pair, "Hasil greedy", WARN))
    fig.add_traces(_highlighted_pair_traces(points_array, true_pair, "Terdekat sebenarnya", ALERT))
    return style_3d(fig, height)


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
                                     name="acuan O(n²)", line=dict(color=_rgba(ALERT, 0.35), dash="dot", width=1),
                                     legendgroup="ref_n2", showlegend=column == 1, hoverinfo="skip"), 1, column)
        dnc_anchor = _first_measurement(dist_results, "dc_standard")
        if dnc_anchor:
            anchor_n, anchor_time = dnc_anchor
            nlogn_curve = anchor_time * n_grid * np.log(n_grid) / (anchor_n * math.log(anchor_n))
            fig.add_trace(go.Scatter(x=n_grid, y=nlogn_curve, mode="lines", name="acuan O(n log n)",
                                     line=dict(color=_rgba(ACCENT, 0.35), dash="dot", width=1),
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
    _style_subplot_titles(fig, len(distributions))
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
                             fill="tonexty", fillcolor=_rgba(WARN, 0.10), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=repeats, y=mean_error, mode="lines+markers", name="galat rata-rata (%)",
                             line=dict(color=WARN, width=2), hovertemplate="%{x}× → %{y:.1f}%<extra></extra>"),
                  1, 1, secondary_y=False)
    fig.add_trace(go.Scatter(x=repeats, y=accuracy.frac_exact * 100, mode="lines+markers", name="% percobaan eksak",
                             line=dict(color=OK, dash="dot", width=2), marker=dict(symbol="square"),
                             hovertemplate="%{x}× → %{y:.0f}% eksak<extra></extra>"), 1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=repeats, y=accuracy.mean_time, mode="lines+markers", name="waktu greedy",
                             line=dict(color=BODY_TEXT, width=2), hovertemplate="%{x}× → %{y:.4f} s<extra></extra>"),
                  1, 2)
    _style_subplot_titles(fig, 2)
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
                                 fillcolor=_rgba(line_color, 0.12), legendgroup=distribution, showlegend=False,
                                 hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=levels, y=mean_strip, mode="lines+markers", name=distribution,
                                 legendgroup=distribution, line=dict(color=line_color, width=2), marker=dict(size=5),
                                 hovertemplate=f"{distribution}<br>level %{{x}}<br>rata-rata %{{y:.1f}} titik<extra></extra>"),
                      1, 1)
        fig.add_trace(go.Scatter(x=levels, y=strip_fraction, mode="lines+markers", name=distribution,
                                 legendgroup=distribution, showlegend=False,
                                 line=dict(color=line_color, width=2), marker=dict(size=5),
                                 hovertemplate=f"{distribution}<br>level %{{x}}<br>%{{y:.0%}}<extra></extra>"), 1, 2)
    _style_subplot_titles(fig, 2)
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
