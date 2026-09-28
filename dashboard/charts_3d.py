"""Chart 3D interaktif: model atom, penelusuran DnC, dan jalur greedy."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree

from utils.baselines import nearest_neighbor_distances
from .theme import (ACCENT, ALERT, BACKGROUND_ATOM, BRIGHT_TEXT, ELEMENT_COLORS, ELEMENT_SIZES, FONT_FAMILY,
                    INACTIVE_ATOM, METAL_SIZE, MUTED_TEXT, NEAREST_NEIGHBOR_SCALE, OK, VIOLET, WARN, style_3d)


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
