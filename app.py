"""Dashboard Streamlit: Closest-Pair Analyzer untuk validasi struktur MOF.

Jalankan: streamlit run app.py
Semua logika algoritma ada di `utils/`, komponen tampilan di `dashboard/`; file ini hanya
merangkai halaman: konfigurasi, sidebar, header, lalu memilih tampilan.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.cache import analyze_structure, load_dataset
from dashboard.components import note, sidebar_group
from dashboard.config import DISTRIBUTIONS, LOGO_SVG, SOURCE_DEMO_2D, SOURCE_MOF, SOURCE_SYNTHETIC, VIEWS
from dashboard.context import AppContext
from dashboard.views import benchmark, demo_2d, divide_conquer, greedy, strips, structure

st.set_page_config(page_title="Closest-Pair Analyzer", page_icon="⚛", layout="wide",
                   initial_sidebar_state="expanded")

# Gaya halaman (shadcn/ui) ada di .streamlit/style.css
STYLE_PATH = Path(__file__).parent / ".streamlit" / "style.css"
st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar: dataset & kriteria
with st.sidebar:
    st.markdown(f'''<div class="app-header side-brand"><div class="brand"><div class="logo">{LOGO_SVG}</div>
        <div><div class="app-title" style="font-size:15px">Closest-Pair</div>
        <div class="app-subtitle" style="font-size:12px">Analyzer</div></div></div></div>''',
                unsafe_allow_html=True)
    sidebar_group("Dataset")
    source = st.selectbox("Sumber data", [SOURCE_SYNTHETIC, SOURCE_MOF, SOURCE_DEMO_2D])
    if source == SOURCE_SYNTHETIC:
        num_points = st.slider("Jumlah atom N", 50, 5000, 600, 50)
        distribution = st.selectbox("Distribusi", DISTRIBUTIONS,
                                    help="uniform: acak seragam · clustered: gugus Gaussian · "
                                         "lattice_jitter: grid kristal + noise (mirip MOF simetris)")
        seed = st.number_input("Seed", 0, 9999, 0)
        plant_pair = st.toggle("Sisipkan pasangan dekat", value=True)
        planted_distance = st.slider("Jarak pasangan sisipan (Å)", 0.3, 2.0, 0.96, 0.02, disabled=not plant_pair)
    else:
        num_points, distribution, seed, plant_pair, planted_distance = 0, "uniform", 0, False, 0.0

    if source != SOURCE_DEMO_2D:
        sidebar_group("Kriteria validitas")
        threshold = st.slider("Threshold (Å)", 0.5, 2.5, 1.2, 0.05,
                              help="Struktur dibuang bila jarak atom terdekat < threshold (ikatan C≡C = 1.20 Å).")
    st.divider()
    note("Reproduksi <i>A divide-and-conquer solution for the closest-pair problem in computer-aided "
         "MOF assembly</i>, Li et al., Comput. Mater. Sci. 248 (2025) 113606.")

if source == SOURCE_DEMO_2D:
    st.markdown(f"""
    <div class="app-header">
      <div class="brand"><div class="logo">{LOGO_SVG}</div>
        <div><div class="app-title">Demo 2D · Step-by-step</div>
        <div class="app-subtitle">Naive vs Greedy vs Divide & Conquer pada 7 titik tetap</div></div></div>
      <div class="header-meta"><span class="chip">P1..P7</span><span class="chip">Euclidean 2D</span></div>
    </div>""", unsafe_allow_html=True)
    demo_2d.render()
    st.stop()

try:
    points, symbols = load_dataset(source, num_points, distribution, seed, plant_pair, planted_distance)
except RuntimeError as error:
    st.error(f"Gagal menyisipkan pasangan dekat pada pengaturan ini: {error}")
    st.stop()

analysis = analyze_structure(points, threshold)
atom_count = len(points)
shortest_distance = analysis["shortest_distance"]
closest_pair = analysis["closest_pair"]
is_invalid = shortest_distance < threshold
naive_pair_count = atom_count * (atom_count - 1) // 2

# ---------------------------------------------------------------- header + statistik
source_label = "MOF pcl (CIF)" if source == SOURCE_MOF else f"Sintetis · {distribution}"
status_label = "Struktur dibuang" if is_invalid else "Struktur valid"
st.markdown(f"""
<div class="app-header">
  <div class="brand"><div class="logo">{LOGO_SVG}</div>
    <div><div class="app-title">Analisis Closest-Pair</div>
    <div class="app-subtitle">Filter validitas struktur MOF berbasis jarak atom terdekat</div></div></div>
  <div class="header-meta"><span class="chip">{source_label}</span><span class="chip">threshold {threshold:.2f} Å</span>
    <span class="badge {'bad' if is_invalid else 'good'}"><span class="dot"></span>{status_label}</span></div>
</div>""", unsafe_allow_html=True)

computation_ratio = naive_pair_count / max(analysis["dnc_distance_count"], 1)
stat_cells = [
    ("Jumlah atom", f"{atom_count:,}", f"{naive_pair_count:,} pasangan", ""),
    ("Jarak terdekat", f"{shortest_distance:.3f} Å", f"atom #{closest_pair[0]} ↔ #{closest_pair[1]}",
     "bad" if is_invalid else ""),
    ("Pasangan < threshold", f"{analysis['pairs_below_threshold']:,}", f"threshold {threshold:.2f} Å", ""),
    ("Perhitungan jarak DnC", f"{analysis['dnc_distance_count']:,}", f"{computation_ratio:,.0f}× lebih sedikit dari naive", ""),
    ("Waktu DnC", f"{analysis['dnc_seconds'] * 1e3:,.1f} ms", "divide-and-conquer iteratif", ""),
]
st.markdown('<div class="stats">' + "".join(
    f'<div class="stat"><div class="label">{label}</div><div class="value {tone}">{value}</div>'
    f'<div class="hint">{hint}</div></div>' for label, value, hint, tone in stat_cells) + "</div>",
    unsafe_allow_html=True)

view = st.segmented_control("Tampilan", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
st.write("")

context = AppContext(source=source, seed=seed, threshold=threshold, points=points, symbols=symbols,
                     analysis=analysis, atom_count=atom_count, shortest_distance=shortest_distance,
                     closest_pair=closest_pair, is_invalid=is_invalid)
VIEW_RENDERERS = {
    "Struktur 3D": structure.render,
    "Divide & Conquer": divide_conquer.render,
    "Greedy": greedy.render,
    "Benchmark": benchmark.render,
    "Strip & rasio": strips.render,
}
VIEW_RENDERERS[view](context)
