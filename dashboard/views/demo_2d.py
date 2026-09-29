"""Mode "Demo 2D (step-by-step)": bandingkan Naive, Greedy, dan Divide & Conquer langkah
demi langkah pada dataset 2D tetap (P1..P7), untuk demo presentasi paper.
"""
from __future__ import annotations

import streamlit as st

from dashboard import charts_2d, theme
from dashboard.components import card, detail_rows, show_chart
from utils.demo_2d import POINTS, run_dnc, run_greedy, run_naive, state_at

ALGO_TABS = ["Naive", "Greedy", "Divide & Conquer"]


def _step_key(tab: str) -> str:
    return f"demo2d_step_{tab}"


def render(_ctx=None) -> None:
    naive_result = run_naive()
    dnc_result = run_dnc()

    with card("Dataset demo", "7 titik tetap (P1..P7), sama untuk ketiga algoritma"):
        control_columns = st.columns([2.4, 1], vertical_alignment="bottom")
        tab = control_columns[0].segmented_control("Algoritma", ALGO_TABS, default=ALGO_TABS[0],
                                                    key="demo2d_tab") or ALGO_TABS[0]
        if tab == "Greedy":
            start_id = control_columns[1].selectbox("Titik awal", list(POINTS), key="demo2d_start",
                                                     label_visibility="collapsed")
        else:
            start_id = "P1"

    greedy_result = run_greedy(start_id)
    result = {"Naive": naive_result, "Greedy": greedy_result, "Divide & Conquer": dnc_result}[tab]
    algo_kind = {"Naive": "naive", "Greedy": "greedy", "Divide & Conquer": "dnc"}[tab]
    log = result["log"]

    step_key = _step_key(tab)
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    step = min(st.session_state[step_key], len(log) - 1)

    viewer_column, inspector_column = st.columns([3, 1.15], gap="medium")
    with viewer_column:
        with card(f"Visualisasi · {tab}"):
            state = state_at(log, step)
            show_chart(charts_2d.demo_step_2d(POINTS, algo_kind, state), key=f"demo2d_chart_{tab}")
    with inspector_column:
        entry = log[step]
        with card(f"Langkah {step + 1} / {len(log)}"):
            best = state["best_dist"]
            st.markdown(f'<div class="big-value">{best if best == float("inf") else f"{best:.3f}"} Å</div>' +
                        detail_rows([
                            ("Tipe", entry["type"]),
                            ("Titik terlibat", ", ".join(entry.get("points") or []) or "-"),
                            ("Jarak dihitung", f"{entry['distance']:.3f}" if entry.get("distance") is not None else "-"),
                        ]), unsafe_allow_html=True)
        button_columns = st.columns(2)
        if button_columns[0].button("← Sebelumnya", width="stretch", disabled=step == 0, key=f"demo2d_prev_{tab}"):
            st.session_state[step_key] = max(0, step - 1)
            st.rerun()
        if button_columns[1].button("Berikutnya →", width="stretch", disabled=step >= len(log) - 1,
                                    key=f"demo2d_next_{tab}"):
            st.session_state[step_key] = min(len(log) - 1, step + 1)
            st.rerun()

    greedy_optimal = abs(greedy_result["finalDistance"] - naive_result["finalDistance"]) < 1e-9
    summary_rows = [
        ("Naive", naive_result["comparisonCount"], naive_result["finalDistance"], naive_result["finalPair"], True),
        (f"Greedy (start {start_id})", greedy_result["comparisonCount"], greedy_result["finalDistance"],
         greedy_result["finalPair"], greedy_optimal),
        ("Divide & Conquer", dnc_result["comparisonCount"], dnc_result["finalDistance"], dnc_result["finalPair"], True),
    ]
    with card("Ringkasan (final)", "Ketiga algoritma dihitung penuh di setiap perubahan; bandingkan tanpa pindah tab"):
        rows_html = "".join(
            f'<tr><td>{algo_name}</td><td>{count}</td><td>{dist:.3f} Å</td>'
            f'<td>{pair[0]}-{pair[1]}</td>'
            f'<td>{"" if is_optimal else f"<span style=color:{theme.ALERT}>⚠ tidak optimal</span>"}</td></tr>'
            for algo_name, count, dist, pair, is_optimal in summary_rows)
        st.markdown(f'''<table style="width:100%;font-size:13px;border-collapse:collapse">
            <thead><tr style="color:{theme.MUTED_TEXT};text-align:left">
                <th style="padding:6px 0">Algoritma</th><th>Jumlah perbandingan</th><th>Hasil</th>
                <th>Pasangan</th><th>Status</th></tr></thead>
            <tbody>{rows_html}</tbody></table>''', unsafe_allow_html=True)
