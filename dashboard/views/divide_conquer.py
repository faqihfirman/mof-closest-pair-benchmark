"""Tampilan Divide & Conquer: penelusuran tahap penggabungan dalam 3D atau 2D."""
from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard import charts_2d, charts_3d, theme
from dashboard.cache import synthetic_points
from dashboard.components import card, detail_rows, note, show_chart
from dashboard.config import DISTRIBUTIONS
from dashboard.context import AppContext
from utils.divide_conquer import closest_pair_dc


def render(ctx: AppContext) -> None:
    """Gambar tampilan ini untuk dataset aktif."""
    seed = ctx.seed
    with card("Parameter penelusuran", "Dataset kecil terpisah agar setiap tahap mudah diamati"):
        control_columns = st.columns([1.2, 1.2, 1.2, 1.1, 0.9])
        trace_size = control_columns[0].slider("N (demo)", 20, 400, 80, 10)
        trace_distribution = control_columns[1].selectbox("Distribusi", DISTRIBUTIONS, key="trace_dist")
        trace_split_ratio = control_columns[2].slider("Rasio pembagian (kiri)", 0.1, 0.9, 0.5, 0.05)
        trace_strip_mode = control_columns[3].segmented_control("Mode strip", ["standard", "paper"], default="standard",
                                                                key="trace_mode") or "standard"
        trace_view = control_columns[4].segmented_control("Tampilan", ["3D", "2D"], default="3D",
                                                          key="trace_view") or "3D"

    trace_points = synthetic_points(trace_size, trace_distribution, seed)
    merge_steps: list[dict] = []
    trace_distance, trace_pair = closest_pair_dc(trace_points, split_ratio=trace_split_ratio,
                                                 strip_mode=trace_strip_mode, trace=merge_steps)
    sorted_to_original = np.argsort(trace_points[:, 0], kind="stable")

    viewer_column, inspector_column = st.columns([3, 1.15], gap="medium")
    with viewer_column:
        with card("Tahap penggabungan", "Urutan post-order: sub-masalah kecil lebih dulu, akar paling akhir"):
            current_step = st.slider("Tahap", 1, len(merge_steps), len(merge_steps), label_visibility="collapsed")
            step = merge_steps[current_step - 1]
            step_chart = charts_3d.dnc_step_3d if trace_view == "3D" else charts_2d.dnc_step_2d
            show_chart(step_chart(trace_points, sorted_to_original, step, height=600), key="dncstep",
                       is_3d=trace_view == "3D")
    strip_improved = step["d_after"] < step["d_children"] - 1e-12
    with inspector_column:
        with card(f"Tahap {current_step} dari {len(merge_steps)}", "Kondisi setelah strip diperiksa"):
            st.markdown(f'<div class="big-value" style="color:{theme.ALERT if strip_improved else theme.BRIGHT_TEXT}">'
                        f'{step["d_after"]:.3f} Å</div>' + detail_rows([
                            ("Level rekursi", str(step["depth"])),
                            ("Ukuran subset", str(step["hi"] - step["lo"])),
                            ("Kiri : kanan", f"{step['mid'] - step['lo']} : {step['hi'] - step['mid']}"),
                            ("d = min(d1, d2)", f"{step['d_children']:.3f} Å"),
                            ("Titik di strip", str(step["strip_hi"] - step["strip_lo"])),
                            ("Strip memperbaiki d", "Ya" if strip_improved else "Tidak"),
                        ]), unsafe_allow_html=True)
        with card("Cara kerja"):
            note("<b>Divide</b>: urutkan menurut X, bagi di <code>x_mid</code>.<br>"
                 "<b>Conquer</b>: selesaikan kiri dan kanan → <code>d = min(d1, d2)</code>.<br>"
                 "<b>Combine</b>: pasangan lintas bidang yang lebih dekat dari d pasti berada di slab "
                 "<code>|x − x_mid| &lt; d</code>, jadi hanya titik bercincin oranye yang diperiksa.")
            note(f"<br>Hasil akhir: <b>{trace_distance:.4f} Å</b>, pasangan #{trace_pair[0]} ↔ #{trace_pair[1]}.")
    with card("Evolusi d per tahap", "Titik merah = tahap ketika strip menemukan pasangan lintas yang lebih dekat"):
        show_chart(charts_2d.d_timeline(merge_steps, current_step), key="dtimeline")
