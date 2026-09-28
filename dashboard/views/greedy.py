"""Tampilan Greedy: jalur kunjungan 3D dan uji akurasi greedy berulang."""
from __future__ import annotations

import streamlit as st

from dashboard import charts_2d, charts_3d, theme
from dashboard.components import card, detail_rows, note, show_chart
from dashboard.context import AppContext
from utils.benchmark import run_greedy_accuracy, time_function
from utils.divide_conquer import closest_pair_dc
from utils.greedy import closest_pair_greedy


def render(ctx: AppContext) -> None:
    """Gambar tampilan ini untuk dataset aktif."""
    points = ctx.points
    atom_count = ctx.atom_count
    shortest_distance = ctx.shortest_distance
    closest_pair = ctx.closest_pair
    viewer_column, inspector_column = st.columns([3, 1.15], gap="medium")
    with viewer_column:
        with card("Jalur greedy", "Jalur kunjungan (hijau → ungu) dibandingkan pasangan terdekat sebenarnya (merah)"):
            greedy_seed = st.slider("Titik awal acak (seed greedy)", 0, 200, 0)
            visit_order: list[int] = []
            greedy_distance, greedy_pair = closest_pair_greedy(points, seed=greedy_seed, path=visit_order)
            show_chart(charts_3d.greedy_path_3d(points, visit_order, greedy_pair, closest_pair, height=620),
                       key="greedy3d", is_3d=True)
    relative_error = (greedy_distance - shortest_distance) / shortest_distance * 100
    greedy_is_correct = relative_error <= 1e-9
    with inspector_column:
        with card("Hasil greedy 1×", "Satu kali jalan dari titik awal acak"):
            st.markdown(f'<div class="big-value" style="color:{theme.OK if greedy_is_correct else theme.WARN}">'
                        f'{greedy_distance:.3f} Å</div>' + detail_rows([
                            ("Jarak sebenarnya", f"{shortest_distance:.3f} Å"),
                            ("Galat relatif", f"{relative_error:.1f} %"),
                            ("Titik dikunjungi", f"{len(visit_order):,} / {atom_count:,}"),
                            ("Status", "Optimal" if greedy_is_correct else "Salah (optimum lokal)"),
                        ]), unsafe_allow_html=True)
        with card("Mengapa sering salah?"):
            note("Greedy hanya mencari di kubus selebar jarak terbaik saat ini dan berhenti saat kubus kosong. "
                 "Setelah langkah pertama kubus mengecil, sehingga jalur 2–3 titik itu normal dan pasangan "
                 "terdekat yang sebenarnya sering terlewat.")
    with card("Uji akurasi greedy berulang", "Galat relatif terhadap jumlah pengulangan (analog Fig. 6 paper)"):
        accuracy_columns = st.columns([1.2, 1.2, 2, 0.8])
        max_repeats = accuracy_columns[0].select_slider("Pengulangan maks", [10, 50, 100, 200, 500, 1000], value=200)
        trials_per_point = accuracy_columns[1].slider("Percobaan per titik", 3, 20, 8)
        accuracy_columns[3].write("")
        if accuracy_columns[3].button("Jalankan", key="run_greedy", width="stretch"):
            repeat_values = [value for value in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if value <= max_repeats]
            with st.spinner("Menghitung..."):
                accuracy_table = run_greedy_accuracy(points, repeat_values, trials=trials_per_point)
                dnc_time = time_function(closest_pair_dc, points, repeats=3).median
            st.session_state["greedy_accuracy"] = (accuracy_table, dnc_time)
        if "greedy_accuracy" in st.session_state:
            accuracy_table, dnc_time = st.session_state["greedy_accuracy"]
            show_chart(charts_2d.greedy_chart(accuracy_table, dnc_time), key="greedychart")
