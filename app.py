"""Dashboard Streamlit: Closest-Pair Analyzer untuk validasi struktur MOF.

Jalankan: streamlit run app.py
Semua logika algoritma ada di `utils/`; file ini hanya UI.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

from dashboard import charts_2d, charts_3d, theme
from utils.baselines import k_closest_pairs, nearest_neighbor_distances
from utils.benchmark import (paper_algos, run_greedy_accuracy, run_scaling_experiment,
                             run_split_ratio_experiment, time_function, verify_correctness)
from utils.data import box_for, generate_points, load_cif, plant_close_pair
from utils.divide_conquer import closest_pair_dc
from utils.greedy import closest_pair_greedy

CIF_PATH = Path(__file__).parent / "data" / "pcl_v1-on15-3D_1-oe107.cif"
SOURCE_SYNTHETIC = "Sintetis"
SOURCE_MOF = "MOF pcl (CIF)"
DISTRIBUTIONS = ["uniform", "clustered", "lattice_jitter"]
MOF_BOND_RADIUS = 1.75  # Å: batas gambar ikatan kovalen pada dataset MOF
VIEWS = ["Struktur 3D", "Divide & Conquer", "Greedy", "Benchmark", "Strip & rasio"]
LOGO_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            '<circle cx="6" cy="17" r="3"/><circle cx="18" cy="7" r="3"/><path d="M8.5 15.2 15.5 8.8"/></svg>')

st.set_page_config(page_title="Closest-Pair Analyzer", page_icon="⚛", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- gaya halaman (shadcn/ui)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
.stApp, .stApp p, .stApp div, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li,
.stApp span:not([data-testid="stIconMaterial"]) { font-family: 'Poppins', ui-sans-serif, system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1520px; }

/* kartu = st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][style*="border"] {
  background: #ffffff; border-radius: 12px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.card-head { margin: 2px 0 10px; }
.card-title { font-size: 15px; font-weight: 600; color: #09090b; letter-spacing: -0.01em; }
.card-desc { font-size: 12.5px; color: #71717a; margin-top: 2px; }

/* header aplikasi */
.app-header { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding-bottom:16px; margin-bottom:16px; border-bottom:1px solid #e4e4e7; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:36px; height:36px; border-radius:8px; background:#18181b; color:#fafafa;
  display:flex; align-items:center; justify-content:center; }
.app-title { font-size:20px; font-weight:600; color:#09090b; letter-spacing:-0.02em; line-height:1.3; }
.app-subtitle { font-size:13px; color:#71717a; }
.header-meta { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.chip { font-size:12px; font-weight:500; color:#3f3f46; border:1px solid #e4e4e7; background:#fff;
  padding:3px 10px; border-radius:6px; }
.badge { display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:500; padding:3px 10px; border-radius:6px; }
.badge .dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
.badge.bad { color:#b91c1c; background:#fef2f2; border:1px solid #fecaca; }
.badge.good { color:#15803d; background:#f0fdf4; border:1px solid #bbf7d0; }

/* blok logo sidebar: tinggi & garis bawah disamakan dengan .app-header di halaman utama */
.side-brand { height:64px; box-sizing:border-box; margin:-8px 0 8px; flex-wrap:nowrap; }

/* strip statistik */
.stats { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); background:#fff; border:1px solid #e4e4e7;
  border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,0.05); overflow:hidden; margin-bottom:18px; }
.stat { padding:14px 20px; border-left:1px solid #e4e4e7; }
.stat:first-child { border-left:0; }
.stat .label { font-size:12.5px; color:#71717a; font-weight:500; }
.stat .value { font-size:22px; font-weight:600; color:#09090b; margin-top:2px; letter-spacing:-0.01em; font-variant-numeric:tabular-nums; }
.stat .value.bad { color:#dc2626; }
.stat .hint { font-size:12px; color:#a1a1aa; }
@media (max-width: 1000px) { .stats { grid-template-columns:repeat(2,minmax(0,1fr)); } .stat { border-top:1px solid #e4e4e7; } }

/* panel inspector */
.big-value { font-size:30px; font-weight:600; letter-spacing:-0.02em; line-height:1.2; font-variant-numeric:tabular-nums; }
.detail-row { display:flex; justify-content:space-between; font-size:13px; color:#71717a; padding:7px 0; border-top:1px solid #f4f4f5; }
.detail-row b { color:#09090b; font-weight:500; font-variant-numeric:tabular-nums; }
.note { font-size:12.5px; color:#71717a; line-height:1.65; }
.note code { background:#f4f4f5; color:#09090b; padding:1px 5px; border-radius:4px; font-size:12px; }
.side-group { font-size:12px; font-weight:600; color:#71717a; margin:18px 0 6px; }

/* segmented control = TabsList shadcn (tombolnya role="radio" + aria-checked) */
[data-testid="stButtonGroup"] [role="radiogroup"] { background:#f4f4f5; padding:4px; border-radius:8px;
  width:fit-content; gap:2px; display:inline-flex; }
[data-testid="stButtonGroup"] [role="radio"] { border:0 !important; background:transparent !important; box-shadow:none !important;
  color:#71717a !important; font-weight:500; border-radius:6px !important; padding:4px 12px !important; min-height:30px; }
[data-testid="stButtonGroup"] [role="radio"]:hover { color:#09090b !important; }
[data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] { background:#ffffff !important; color:#09090b !important;
  box-shadow:0 1px 3px rgba(0,0,0,0.10), 0 1px 2px rgba(0,0,0,0.06) !important; }

/* tombol utama */
.stButton button { background:#18181b; color:#fafafa; border:1px solid #18181b; border-radius:6px; font-weight:500; min-height:36px; }
.stButton button:hover { background:#27272a; color:#fafafa; border-color:#27272a; }
.stButton button:focus:not(:active) { color:#fafafa; }
[data-testid="stPlotlyChart"] { background:transparent; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- komponen UI
def show_chart(fig, key: str | None = None, is_3d: bool = False) -> None:
    """Tampilkan Plotly dengan gaya kita sendiri (theme=None: jangan ditimpa tema Streamlit).

    Chart 3D boleh di-zoom dengan scroll/pinch; chart 2D tidak, agar scroll halaman tetap lancar.
    """
    config = theme.PLOTLY_CONFIG_3D if is_3d else theme.PLOTLY_CONFIG
    st.plotly_chart(fig, theme=None, config=config, width="stretch", key=key)


def card(title: str, description: str = ""):
    """Kartu shadcn: kontainer berbingkai dengan judul dan deskripsi. Pakai: `with card(...):`."""
    container = st.container(border=True)
    description_html = f'<div class="card-desc">{description}</div>' if description else ""
    container.markdown(f'<div class="card-head"><div class="card-title">{title}</div>{description_html}</div>',
                       unsafe_allow_html=True)
    return container


def detail_rows(rows: list[tuple[str, str]]) -> str:
    return "".join(f'<div class="detail-row"><span>{name}</span><b>{value}</b></div>' for name, value in rows)


def note(html: str) -> None:
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


def sidebar_group(text: str) -> None:
    st.markdown(f'<div class="side-group">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- data (di-cache)
@st.cache_data(show_spinner=False)
def load_dataset(source: str, num_points: int, distribution: str, seed: int,
                 plant_pair: bool, planted_distance: float) -> tuple[np.ndarray, list[str] | None]:
    """Titik atom dari CIF atau generator sintetis (opsional dengan pasangan dekat disisipkan)."""
    if source == SOURCE_MOF:
        mof_points, mof_symbols = load_cif(CIF_PATH, return_symbols=True)
        return mof_points, list(mof_symbols)
    points_generated = generate_points(num_points, distribution, box_for(num_points), seed)
    if plant_pair:
        points_generated, _ = plant_close_pair(points_generated, planted_distance, seed=seed)
    return points_generated, None


@st.cache_data(show_spinner=False)
def analyze_structure(points: np.ndarray, threshold: float) -> dict:
    """Jalankan DnC (dengan statistik) dan ringkasan jarak untuk panel utama."""
    started = time.perf_counter()
    (shortest_distance, closest_pair), dnc_stats = closest_pair_dc(points, return_stats=True)
    dnc_seconds = time.perf_counter() - started
    return {
        "shortest_distance": shortest_distance,
        "closest_pair": closest_pair,
        "dnc_seconds": dnc_seconds,
        "dnc_distance_count": dnc_stats["n_distance_computations"],
        "nn_distance": nearest_neighbor_distances(points),
        "pairs_below_threshold": len(cKDTree(points).query_pairs(threshold)),
        "top_pairs": k_closest_pairs(points, k=10),
    }


@st.cache_data(show_spinner=False)
def synthetic_points(num_points: int, distribution: str, seed: int) -> np.ndarray:
    return generate_points(num_points, distribution, box_for(num_points), seed)


# ---------------------------------------------------------------- sidebar: dataset & kriteria
with st.sidebar:
    st.markdown(f'''<div class="app-header side-brand"><div class="brand"><div class="logo">{LOGO_SVG}</div>
        <div><div class="app-title" style="font-size:15px">Closest-Pair</div>
        <div class="app-subtitle" style="font-size:12px">Analyzer</div></div></div></div>''',
                unsafe_allow_html=True)
    sidebar_group("Dataset")
    source = st.selectbox("Sumber data", [SOURCE_SYNTHETIC, SOURCE_MOF])
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

    sidebar_group("Kriteria validitas")
    threshold = st.slider("Threshold (Å)", 0.5, 2.5, 1.2, 0.05,
                          help="Struktur dibuang bila jarak atom terdekat < threshold (ikatan C≡C = 1.20 Å).")
    st.divider()
    note("Reproduksi <i>A divide-and-conquer solution for the closest-pair problem in computer-aided "
         "MOF assembly</i>, Li et al., Comput. Mater. Sci. 248 (2025) 113606.")

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

# ---------------------------------------------------------------- 1. struktur 3D
if view == "Struktur 3D":
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

# ---------------------------------------------------------------- 2. divide & conquer
elif view == "Divide & Conquer":
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

# ---------------------------------------------------------------- 3. greedy
elif view == "Greedy":
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

# ---------------------------------------------------------------- 4. benchmark
elif view == "Benchmark":
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

# ---------------------------------------------------------------- 5. strip & rasio
else:
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
