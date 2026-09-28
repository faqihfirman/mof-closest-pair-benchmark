"""Tampilan Strip & rasio: ukuran strip per level dan pengaruh rasio pembagian."""
from __future__ import annotations

import streamlit as st

from dashboard import charts_2d
from dashboard.cache import synthetic_points
from dashboard.components import card, show_chart
from dashboard.config import DISTRIBUTIONS
from dashboard.context import AppContext
from utils.benchmark import run_split_ratio_experiment
from utils.divide_conquer import closest_pair_dc


def render(ctx: AppContext) -> None:
    """Gambar tampilan ini untuk dataset aktif."""
    seed = ctx.seed
    with card("Ukuran strip per level rekursi", "Strip besar = lebih banyak pekerjaan; penyebab waktu DnC "
                                                "tidak monoton di paper"):
        strip_size = st.slider("N untuk statistik strip", 1000, 20000, 5000, 1000)
        stats_by_distribution = {}
        for name in DISTRIBUTIONS:
            _, stats_by_distribution[name] = closest_pair_dc(synthetic_points(strip_size, name, seed),
                                                             return_stats=True)
        show_chart(charts_2d.strip_chart(stats_by_distribution), key="strip")
        st.markdown('<div class="stats" style="grid-template-columns:repeat(3,minmax(0,1fr));margin:4px 0 0">' + "".join(
            f'<div class="stat"><div class="label">{name}</div>'
            f'<div class="value">{stats["n_distance_computations"]:,}</div>'
            f'<div class="hint">perhitungan jarak · strip rata-rata {stats["strip_overall"]["mean"]:.1f} · '
            f'maks {stats["strip_overall"]["max"]}</div></div>'
            for name, stats in stats_by_distribution.items()) + "</div>", unsafe_allow_html=True)

    with card("Rasio pembagian kiri:kanan", "Waktu DnC untuk rasio 1:6 sampai 1:1 (analog Fig. 8 paper)"):
        ratio_columns = st.columns([4, 0.8])
        ratio_sizes = ratio_columns[0].multiselect("Ukuran N", [1000, 3000, 6000, 10000], default=[1000, 3000, 6000],
                                                   key="ratio_sizes")
        ratio_columns[1].write("")
        if ratio_columns[1].button("Jalankan", key="run_ratio", width="stretch") and ratio_sizes:
            with st.spinner("Menghitung..."):
                st.session_state["ratio"] = run_split_ratio_experiment(
                    {size: synthetic_points(size, "uniform", seed) for size in ratio_sizes}, repeats=3)
        if "ratio" in st.session_state:
            show_chart(charts_2d.ratio_chart(st.session_state["ratio"]), key="ratio")
