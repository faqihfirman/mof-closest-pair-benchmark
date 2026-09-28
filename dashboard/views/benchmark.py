"""Tampilan Benchmark: perbandingan Naive, Greedy, dan Divide & Conquer, plus eksperimen skala."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts_2d, theme
from dashboard.components import card, note, show_chart
from dashboard.config import DISTRIBUTIONS
from dashboard.context import AppContext
from utils.benchmark import paper_algos, run_scaling_experiment, time_function, verify_correctness


def render(ctx: AppContext) -> None:
    """Gambar tampilan ini untuk dataset aktif."""
    seed = ctx.seed
    points = ctx.points
    atom_count = ctx.atom_count
    with card("Perbandingan algoritma", "Naive, Greedy, dan Divide & Conquer: waktu median dan kebenaran "
                                        "pada dataset aktif"):
        compare_columns = st.columns([5, 0.8], vertical_alignment="center")
        with compare_columns[0]:
            note("Kebenaran dinilai terhadap jarak sebenarnya. Naive Python murni dilewati untuk N &gt; 4000.")
        if compare_columns[1].button("Bandingkan", key="run_compare", width="stretch"):
            algorithms_to_run = paper_algos(seed)
            if atom_count > 4000:
                algorithms_to_run.pop("naive")
                st.info("Naive Python murni dilewati untuk N > 4000.")
            with st.spinner("Menghitung..."):
                correctness = verify_correctness(points, algorithms_to_run).set_index("algorithm")
                timing_rows = []
                for key, algorithm in algorithms_to_run.items():
                    timing = time_function(algorithm, points, repeats=3, warmup=1, max_total_s=3.0)
                    timing_rows.append({"key": key, "algoritma": theme.label(key), "time": timing.median,
                                        "correct": bool(correctness.loc[key, "correct"])})
            st.session_state["comparison"] = pd.DataFrame(timing_rows)
        if "comparison" in st.session_state:
            show_chart(charts_2d.compare_bars(st.session_state["comparison"]), key="comparebars")

    with card("Eksperimen skala", "Waktu terhadap N (log-log) dengan acuan teoritis O(n²) dan O(n log n) · naive dibatasi N ≤ 6000"):
        scale_columns = st.columns([3, 2, 1, 0.8])
        scale_sizes = scale_columns[0].multiselect("Ukuran N", [300, 600, 1000, 2000, 3000, 4000, 6000, 10000, 20000],
                                                   default=[300, 1000, 3000, 6000, 10000])
        scale_distributions = scale_columns[1].multiselect("Distribusi", DISTRIBUTIONS, default=["uniform"],
                                                           key="scale_dist")
        scale_repeats = scale_columns[2].slider("Ulangan", 1, 5, 3)
        scale_columns[3].write("")
        if (scale_columns[3].button("Jalankan", key="run_scale", width="stretch")
                and scale_sizes and scale_distributions):
            with st.spinner("Menjalankan (naive Python dibatasi N ≤ 6000)..."):
                st.session_state["scaling"] = pd.concat(
                    [run_scaling_experiment(sorted(scale_sizes), name, paper_algos(seed),
                                            repeats=scale_repeats, seed=seed) for name in scale_distributions],
                    ignore_index=True)
        if "scaling" in st.session_state:
            show_chart(charts_2d.scaling_chart(st.session_state["scaling"]), key="scaling")
            speedup_figure = charts_2d.speedup_chart(st.session_state["scaling"])
            if speedup_figure is not None:
                st.markdown('<div class="card-title" style="margin-top:8px">Speedup naive / DnC</div>',
                            unsafe_allow_html=True)
                show_chart(speedup_figure, key="speedup")
