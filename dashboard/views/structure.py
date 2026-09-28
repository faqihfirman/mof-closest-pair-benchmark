"""Tampilan Struktur 3D: viewer atom, panel pasangan terdekat, peringkat, dan histogram jarak."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts_2d, charts_3d, theme
from dashboard.components import card, detail_rows, note, show_chart
from dashboard.config import MOF_BOND_RADIUS, SOURCE_MOF
from dashboard.context import AppContext


def render(ctx: AppContext) -> None:
    """Gambar tampilan ini untuk dataset aktif."""
    source = ctx.source
    threshold = ctx.threshold
    points = ctx.points
    symbols = ctx.symbols
    analysis = ctx.analysis
    shortest_distance = ctx.shortest_distance
    closest_pair = ctx.closest_pair
    is_invalid = ctx.is_invalid
    viewer_column, inspector_column = st.columns([3, 1.15], gap="medium")
    with viewer_column:
        with card("Viewer 3D", "Seret untuk memutar · scroll atau pinch untuk zoom · klik legend untuk "
                               "menyembunyikan lapisan"):
            control_columns = st.columns([1.7, 0.9, 0.9, 1.2])
            color_options = {"Jarak tetangga": "nn", "Polos": "flat"}
            if symbols is not None:
                color_options = {"Elemen": "element", **color_options}
            color_choice = control_columns[0].segmented_control("Warna atom", list(color_options),
                                                                default=list(color_options)[0], key=f"color_{source}")
            color_mode = color_options[color_choice or list(color_options)[0]]
            if source == SOURCE_MOF:
                show_links = control_columns[1].toggle("Ikatan", value=True, help=f"Garis antar-atom < {MOF_BOND_RADIUS} Å")
                bond_radius = MOF_BOND_RADIUS if show_links else None
            else:
                show_links = control_columns[1].toggle("Tautan", value=False, help="Garis antar-atom yang lebih dekat dari threshold")
                bond_radius = threshold if show_links else None
            # Di MOF, ikatan C–H (~1.09 Å) sudah di bawah 1.2 Å: tanda default mati agar model tak tertutup.
            mark_violations = control_columns[2].toggle("Tandai", value=source != SOURCE_MOF, key=f"mark_{source}",
                                                        help="Cincin oranye pada atom yang tetangganya < threshold")
            size_scale = control_columns[3].slider("Ukuran atom", 0.5, 2.5, 1.0, 0.1)
            show_chart(charts_3d.atoms_3d(points, closest_pair, symbols=symbols, color_mode=color_mode,
                                   bond_radius=bond_radius, threshold=threshold if mark_violations else None,
                                   size_scale=size_scale, height=680), key="structure3d", is_3d=True)
    with inspector_column:
        with card("Pasangan terdekat", "Hasil divide-and-conquer"):
            pair_elements = f"{symbols[closest_pair[0]]} – {symbols[closest_pair[1]]}" if symbols else "—"
            value_color = theme.ALERT if is_invalid else theme.OK
            st.markdown(f'<div class="big-value" style="color:{value_color}">{shortest_distance:.3f} Å</div>'
                        + detail_rows([("Atom A", f"#{closest_pair[0]}"), ("Atom B", f"#{closest_pair[1]}"),
                                       ("Elemen", pair_elements), ("Threshold", f"{threshold:.2f} Å"),
                                       ("Keputusan", "Dibuang" if is_invalid else "Diterima")]),
                        unsafe_allow_html=True)
        with card("Peringkat pasangan", "10 pasangan atom terdekat"):
            ranking_table = pd.DataFrame(
                [{"A": pair[0], "B": pair[1], "Jarak (Å)": distance} for distance, pair in analysis["top_pairs"]])
            max_distance = float(ranking_table["Jarak (Å)"].max()) if len(ranking_table) else 1.0
            st.dataframe(ranking_table, hide_index=True, width="stretch", height=250,
                         column_config={"Jarak (Å)": st.column_config.ProgressColumn(
                             "Jarak (Å)", format="%.3f", min_value=0.0, max_value=max_distance)})
        if source == SOURCE_MOF:
            with card("Catatan dataset"):
                note("Ikatan C–H (~1.09 Å) wajar berada di bawah threshold 1.2 Å; threshold paper ditujukan "
                     "untuk jarak C–C. Yang anomali adalah pasangan ≪ 1 Å di peringkat atas.")
    with card("Distribusi jarak tetangga terdekat", "Jumlah atom per jarak ke tetangga terdekatnya; "
                                                    "garis merah = threshold"):
        show_chart(charts_2d.nn_histogram(analysis["nn_distance"], threshold, height=230), key="nnhist")
