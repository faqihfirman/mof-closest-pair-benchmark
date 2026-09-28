"""Dashboard Streamlit: belajar interaktif closest-pair 3D (naive vs greedy vs DnC).

Jalankan: streamlit run app.py
Semua logika algoritma ada di `utils/`; file ini hanya UI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.baselines import closest_pair_kdtree
from utils.benchmark import (default_algos, run_greedy_accuracy, run_scaling_experiment,
                             run_split_ratio_experiment, time_function, verify_correctness)
from utils.data import box_for, generate_points, load_cif, plant_close_pair
from utils.divide_conquer import closest_pair_dc

CIF_PATH = Path(__file__).parent / "data" / "pcl_v1-on15-3D_1-oe107.cif"
DISTS = ["uniform", "clustered", "lattice_jitter"]
DIST_HELP = {
    "uniform": "titik acak seragam",
    "clustered": "atom mengumpul di gugus (campuran Gaussian)",
    "lattice_jitter": "grid kristal + noise kecil, meniru MOF simetris",
}

st.set_page_config(page_title="Closest-Pair 3D", page_icon="🔬", layout="wide")


@st.cache_data(show_spinner=False)
def make_points(n: int, dist: str, seed: int) -> np.ndarray:
    return generate_points(n, dist, box_for(n), seed)


@st.cache_data(show_spinner=False)
def cif_points() -> tuple[np.ndarray, list[str]]:
    return load_cif(CIF_PATH, return_symbols=True)


def show_mpl(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def fig_3d(points: np.ndarray, pair: tuple[int, int] | None, title: str, size: float = 3) -> go.Figure:
    fig = go.Figure(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                 marker=dict(size=size, color="#7fb3d5", opacity=0.7), name="Atom"))
    if pair is not None:
        a, b = points[pair[0]], points[pair[1]]
        fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines+markers",
                                   line=dict(color="#e74c3c", width=8), marker=dict(size=6, color="#e74c3c"),
                                   name="Pasangan terdekat"))
    fig.update_layout(title=title, height=520, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)"))
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔬 Closest-Pair 3D")
st.sidebar.caption("Reproduksi Li et al. (2025), *Comput. Mater. Sci.* 248, 113606")
seed = st.sidebar.number_input("Seed acak", 0, 9999, 0, help="Mengubah seed = dataset lain.")
st.sidebar.markdown(
    "**Alur belajar**\n1. Motivasi\n2. Langkah DnC\n3. Bandingkan algoritma\n4. Skala\n"
    "5. Greedy\n6. Strip & rasio\n7. Dataset asli\n8. Kuis")

st.title("Mencari pasangan atom terdekat: Naive vs Greedy vs Divide-and-Conquer")

tabs = st.tabs(["1 Motivasi", "2 Langkah DnC", "3 Bandingkan", "4 Skala", "5 Greedy",
                "6 Strip & rasio", "7 Dataset asli", "8 Kuis"])

# ---------------------------------------------------------------- 1 motivasi
with tabs[0]:
    st.subheader("Kenapa perlu mencari pasangan terdekat?")
    st.markdown(
        "Pada perakitan MOF, iterasi penskalaan yang terbatas dapat menghasilkan atom yang **terlalu dekat** "
        "(di paper: C420-C131 hanya 0.962 Å, lebih pendek dari ikatan rangkap tiga C-C, 1.20 Å). "
        "Struktur seperti itu tidak valid. Filternya: cari jarak terdekat semua pasangan atom, "
        "buang struktur jika di bawah threshold.")
    c1, c2, c3, c4 = st.columns(4)
    n1 = c1.slider("Jumlah atom N", 50, 1000, 300, 50, key="n1")
    d1 = c2.selectbox("Distribusi", DISTS, key="dist1", help="; ".join(f"{k}: {v}" for k, v in DIST_HELP.items()))
    planted = c3.slider("Jarak pasangan disisipkan (Å)", 0.3, 2.0, 0.96, 0.02)
    thr = c4.slider("Threshold (Å)", 0.5, 2.5, 1.2, 0.05)
    try:
        pts1, _ = plant_close_pair(make_points(n1, d1, seed), planted, seed=seed)
        dmin, pair = closest_pair_dc(pts1)
        (st.error if dmin < thr else st.success)(
            f"Jarak terdekat = {dmin:.3f} Å, threshold {thr:.2f} Å → "
            + ("**struktur DIBUANG**" if dmin < thr else "**struktur DITERIMA**"))
        st.plotly_chart(fig_3d(pts1, pair, f"N = {len(pts1)}: pasangan terdekat {dmin:.3f} Å"), width="stretch")
    except RuntimeError as e:
        st.warning(f"Tidak bisa menyisipkan pasangan pada pengaturan ini: {e}")

# ---------------------------------------------------------------- 2 langkah DnC
with tabs[1]:
    st.subheader("Menelusuri Divide-and-Conquer langkah demi langkah")
    st.markdown(
        "Titik diurutkan menurut X lalu dibagi dua. Setelah kiri dan kanan selesai, `d = min(d1, d2)`. "
        "Pasangan yang **melintasi** garis pembagi hanya mungkin ada di strip `|x − x_mid| < d`, "
        "jadi hanya strip itu yang diperiksa. Geser slider untuk melihat tiap tahap penggabungan "
        "(urutan post-order: sub-masalah kecil dulu).")
    c1, c2, c3, c4 = st.columns(4)
    n2 = c1.slider("N", 20, 300, 60, 10, key="n2")
    dist2 = c2.selectbox("Distribusi", DISTS, key="dist2")
    ratio2 = c3.slider("Rasio pembagian (kiri)", 0.1, 0.9, 0.5, 0.05)
    mode2 = c4.radio("Mode strip", ["standard", "paper"], horizontal=True)
    p2 = make_points(n2, dist2, seed)
    events: list[dict] = []
    dtot, ptot = closest_pair_dc(p2, split_ratio=ratio2, strip_mode=mode2, trace=events)
    order = np.argsort(p2[:, 0], kind="stable")
    sp = p2[order]
    step = st.slider("Tahap penggabungan", 1, len(events), len(events))
    ev = events[step - 1]
    lo, hi, mid = ev["lo"], ev["hi"], ev["mid"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp[:, 0], y=sp[:, 1], mode="markers", marker=dict(size=6, color="#d5d8dc"), name="Di luar subset"))
    fig.add_trace(go.Scatter(x=sp[lo:mid, 0], y=sp[lo:mid, 1], mode="markers", marker=dict(size=8, color="#2471a3"), name="Kiri"))
    fig.add_trace(go.Scatter(x=sp[mid:hi, 0], y=sp[mid:hi, 1], mode="markers", marker=dict(size=8, color="#1e8449"), name="Kanan"))
    s_lo, s_hi = ev["strip_lo"], ev["strip_hi"]
    if s_hi > s_lo:
        fig.add_trace(go.Scatter(x=sp[s_lo:s_hi, 0], y=sp[s_lo:s_hi, 1], mode="markers",
                                 marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color="#f39c12", width=2)), name="Titik di strip"))
    d = ev["d_children"]
    fig.add_vrect(x0=ev["x_mid"] - d, x1=ev["x_mid"] + d, fillcolor="#f9e79f", opacity=0.5, line_width=0)
    fig.add_vline(x=ev["x_mid"], line_dash="dash", line_color="black")
    a_, b_ = p2[ev["pair"][0]], p2[ev["pair"][1]]
    fig.add_trace(go.Scatter(x=[a_[0], b_[0]], y=[a_[1], b_[1]], mode="lines", line=dict(color="#e74c3c", width=4), name="Terbaik saat ini"))
    fig.update_layout(height=500, xaxis_title="x (Å)", yaxis_title="y (Å)  [proyeksi XY]", margin=dict(t=30),
                      yaxis_scaleanchor="x")
    st.plotly_chart(fig, width="stretch")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Level rekursi", ev["depth"])
    m2.metric("Ukuran subset", hi - lo)
    m3.metric("d dari anak (min d1, d2)", f"{ev['d_children']:.3f}")
    m4.metric("d setelah cek strip", f"{ev['d_after']:.3f}",
              delta=f"{ev['d_after'] - ev['d_children']:.3f}" if ev["d_after"] < ev["d_children"] else None,
              delta_color="inverse")
    st.caption(f"Strip berisi {s_hi - s_lo} dari {hi - lo} titik. "
               + ("Pasangan lintas garis pembagi **lebih dekat** ditemukan." if ev["d_after"] < ev["d_children"]
                  else "Tidak ada pasangan lintas yang lebih dekat."))
    st.info(f"Hasil akhir seluruh {n2} titik: jarak {dtot:.4f} Å, pasangan {ptot}. "
            f"Total {len(events)} tahap penggabungan.")

# ---------------------------------------------------------------- 3 bandingkan
with tabs[2]:
    st.subheader("Bandingkan semua algoritma pada satu dataset")
    c1, c2, c3 = st.columns(3)
    n3 = c1.slider("N", 100, 6000, 1000, 100, key="n3")
    dist3 = c2.selectbox("Distribusi", DISTS, key="dist3")
    greedy_r = c3.slider("Pengulangan greedy berulang", 2, 200, 20)
    all_algos = default_algos(greedy_repeats=greedy_r, seed=seed)
    chosen = st.multiselect("Algoritma", list(all_algos), default=list(all_algos), format_func=viz._l)
    if st.button("▶ Jalankan perbandingan", key="run3"):
        p3 = make_points(n3, dist3, seed)
        algos = {k: all_algos[k] for k in chosen}
        if "naive" in algos and n3 > 4000:
            st.warning("Naive Python murni dilewati untuk N > 4000 (terlalu lama).")
            algos.pop("naive")
        with st.spinner("Menghitung..."):
            ver = verify_correctness(p3, algos).set_index("algorithm")
            rows = []
            for name, fn in algos.items():
                t = time_function(fn, p3, repeats=3, warmup=1, max_total_s=3.0)
                rows.append({"algoritma": viz._l(name), "key": name, "waktu median (s)": t.median,
                             "jarak (Å)": ver.loc[name, "distance"], "benar": bool(ver.loc[name, "correct"])})
        st.session_state["cmp"] = (n3, dist3, pd.DataFrame(rows))
    if "cmp" in st.session_state:
        n_, dist_, df3 = st.session_state["cmp"]
        st.caption(f"Hasil terakhir: N = {n_}, distribusi {dist_}")
        bar = go.Figure(go.Bar(x=df3["algoritma"], y=df3["waktu median (s)"],
                               marker_color=[viz._c(k) for k in df3["key"]],
                               text=["✔" if b else "✘ salah" for b in df3["benar"]], textposition="outside"))
        bar.update_layout(yaxis_type="log", yaxis_title="Waktu median (detik, skala log)", height=420)
        st.plotly_chart(bar, width="stretch")
        st.dataframe(df3.drop(columns="key").style.format({"waktu median (s)": "{:.5f}", "jarak (Å)": "{:.5f}"}),
                     width="stretch", hide_index=True)
        st.caption("Perhatikan: greedy paling cepat tetapi sering salah; naive NumPy dan KD-tree sering "
                   "mengalahkan DnC Python murni pada N kecil.")

# ---------------------------------------------------------------- 4 skala
with tabs[3]:
    st.subheader("Bagaimana waktu tumbuh terhadap N?")
    c1, c2, c3 = st.columns([3, 2, 1])
    sizes4 = c1.multiselect("Ukuran N", [300, 600, 1000, 2000, 3000, 4000, 6000, 10000, 20000],
                            default=[300, 1000, 3000, 6000, 10000])
    dists4 = c2.multiselect("Distribusi", DISTS, default=["uniform"])
    reps4 = c3.slider("Ulangan", 1, 5, 3)
    if st.button("▶ Jalankan eksperimen skala", key="run4") and sizes4 and dists4:
        with st.spinner("Menjalankan (naive Python dibatasi N ≤ 6000)..."):
            df4 = pd.concat([run_scaling_experiment(sorted(sizes4), d, default_algos(20, seed), repeats=reps4, seed=seed)
                             for d in dists4], ignore_index=True)
        st.session_state["scale"] = df4
    if "scale" in st.session_state:
        df4 = st.session_state["scale"]
        show_mpl(viz.plot_scaling(df4))
        if {"naive", "dc_standard"} <= set(df4.algorithm):
            show_mpl(viz.plot_speedup(df4))
        st.caption("Garis putus-putus = acuan teoritis O(n²) dan O(n log n).")

# ---------------------------------------------------------------- 5 greedy
with tabs[4]:
    st.subheader("Seberapa akurat greedy jika diulang?")
    c1, c2, c3 = st.columns(3)
    n5 = c1.slider("N", 500, 4000, 2000, 500, key="n5")
    maxr = c2.select_slider("Pengulangan maksimum", [10, 50, 100, 200, 500, 1000], value=200)
    trials5 = c3.slider("Percobaan per titik", 3, 20, 8)
    if st.button("▶ Jalankan uji greedy", key="run5"):
        p5 = make_points(n5, "uniform", seed)
        reps = [r for r in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if r <= maxr]
        with st.spinner("Menghitung..."):
            acc = run_greedy_accuracy(p5, reps, trials=trials5)
            dc_t = time_function(closest_pair_dc, p5, repeats=3).median
        st.session_state["greedy"] = (acc, dc_t, n5)
    if "greedy" in st.session_state:
        acc, dc_t, n_ = st.session_state["greedy"]
        show_mpl(viz.plot_greedy_convergence(acc, dc_time=dc_t))
        acc2 = acc.assign(lebih_cepat_dari_DnC=acc.mean_time < dc_t)
        st.dataframe(acc2.round(4), width="stretch", hide_index=True)
        st.caption(f"DnC pada N = {n_}: {dc_t:.4f} s. Greedy 1x hampir selalu salah; makin banyak "
                   "pengulangan makin akurat tetapi waktunya menyusul DnC.")

# ---------------------------------------------------------------- 6 strip & rasio
with tabs[5]:
    st.subheader("Distribusi data dan rasio pembagian")
    n6 = st.slider("N untuk statistik strip", 1000, 20000, 5000, 1000)
    st.markdown("**Ukuran strip per level**: strip besar = lebih banyak pekerjaan, penyebab waktu DnC "
                "tidak monoton di paper.")
    stats = {}
    for dname in DISTS:
        _, stats[dname] = closest_pair_dc(make_points(n6, dname, seed), return_stats=True)
    show_mpl(viz.plot_strip_sizes(stats))
    st.dataframe(pd.DataFrame({k: {"kedalaman maks": v["max_depth"], "hitung jarak": v["n_distance_computations"],
                                   "strip rata-rata": round(v["strip_overall"]["mean"], 1),
                                   "strip maks": v["strip_overall"]["max"]} for k, v in stats.items()}),
                 width="stretch")
    st.markdown("**Rasio pembagian** kiri:kanan (analog Fig. 8 paper)")
    sizes6 = st.multiselect("Ukuran N", [1000, 3000, 6000, 10000], default=[1000, 3000, 6000], key="s6")
    if st.button("▶ Jalankan eksperimen rasio", key="run6") and sizes6:
        with st.spinner("Menghitung..."):
            st.session_state["ratio"] = run_split_ratio_experiment(
                {n: make_points(n, "uniform", seed) for n in sizes6}, repeats=3)
    if "ratio" in st.session_state:
        show_mpl(viz.plot_split_ratio(st.session_state["ratio"]))
        st.caption("Rasio 1:1 umumnya tercepat, sesuai temuan paper.")

# ---------------------------------------------------------------- 7 dataset asli
with tabs[6]:
    st.subheader("Dataset MOF asli (pcl, 3856 atom = dataset No. 8 paper)")
    if not CIF_PATH.exists():
        st.warning("File CIF tidak ditemukan di folder data/.")
    elif st.button("▶ Muat dan jalankan", key="run7"):
        cp, sym = cif_points()
        with st.spinner("Menjalankan semua algoritma (naive Python ±1 detik)..."):
            rows = []
            for name, fn in default_algos(20, seed).items():
                t = time_function(fn, cp, repeats=1 if name == "naive" else 3, warmup=0 if name == "naive" else 1)
                rows.append({"algoritma": viz._l(name), "jarak (Å)": t.result[0], "waktu (s)": t.median})
            truth, tpair = closest_pair_kdtree(cp)
        st.session_state["cif"] = (pd.DataFrame(rows), truth, tpair, [sym[i] for i in tpair])
    if "cif" in st.session_state:
        df7, truth, tpair, syms = st.session_state["cif"]
        cp, _ = cif_points()
        st.dataframe(df7.style.format({"jarak (Å)": "{:.5f}", "waktu (s)": "{:.4f}"}), width="stretch", hide_index=True)
        st.caption("Paper (i5-8250U): naive 39.255 s, DnC 0.782 s, greedy 0.02 s. Jarak lintas batas periodik tidak dihitung.")
        st.plotly_chart(fig_3d(cp, tpair, f"Pasangan terdekat {syms[0]}-{syms[1]}: {truth:.3f} Å", size=1.5),
                        width="stretch")

# ---------------------------------------------------------------- 8 kuis
with tabs[7]:
    st.subheader("Uji pemahaman")
    quiz = [
        ("Kompleksitas waktu naive untuk N atom?", ["O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)"], "O(n²)",
         "Semua C(n,2) pasangan diperiksa."),
        ("Mengapa DnC hanya memeriksa strip selebar 2d di sekitar garis pembagi?",
         ["Karena pasangan lintas garis yang lebih dekat dari d pasti punya kedua titik dalam jarak d dari garis",
          "Karena titik di luar strip selalu identik", "Untuk menghemat memori saja",
          "Karena greedy sudah memeriksa sisanya"],
         "Karena pasangan lintas garis yang lebih dekat dari d pasti punya kedua titik dalam jarak d dari garis",
         "Selisih X pasangan lintas < d, dan salah satu titik ada di tiap sisi garis, jadi keduanya dalam jarak d dari x_mid."),
        ("Apakah greedy (Algo. 2) menjamin pasangan terdekat global?", ["Ya, selalu", "Tidak, hanya optimum lokal"],
         "Tidak, hanya optimum lokal", "Hasil bergantung titik awal; pengulangan hanya menurunkan peluang salah."),
        ("Recurrence T(n) = 2T(n/2) + k dengan k konstan menghasilkan...", ["O(log n)", "O(n)", "O(n log n)", "O(n²)"],
         "O(n)", "Klaim O(n log n) paper baru benar jika langkah gabung (strip) O(n). Ini salah satu kelemahan paper."),
        ("Mengapa DnC diimplementasikan iteratif dengan stack buatan?",
         ["Lebih cepat 100x", "Menghindari batas rekursi Python bila pembagian tidak seimbang",
          "Agar hasilnya deterministik", "Python tidak mendukung rekursi"],
         "Menghindari batas rekursi Python bila pembagian tidak seimbang",
         "Rasio pembagian ekstrem membuat kedalaman rekursi besar."),
    ]
    score = 0
    answered = 0
    for i, (q, opts, ans, why) in enumerate(quiz):
        choice = st.radio(f"**{i + 1}. {q}**", opts, index=None, key=f"q{i}")
        if choice is not None:
            answered += 1
            if choice == ans:
                score += 1
                st.success("Benar. " + why)
            else:
                st.error("Belum tepat. " + why)
    if answered == len(quiz):
        st.metric("Skor", f"{score} / {len(quiz)}")
