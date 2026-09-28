# Closest-pair 3D: Naive vs Greedy vs Divide-and-Conquer

Bahan demo review paper *Li et al., "A divide-and-conquer solution for the closest-pair problem in computer-aided MOF assembly"*, Computational Materials Science 248 (2025) 113606.
Masalah: dari N atom (titik 3D) cari pasangan berjarak terkecil; struktur MOF dibuang bila jarak terdekat di bawah threshold.

## Instalasi
```bash
conda activate dbs            # atau venv lain, Python >= 3.10
pip install -r requirements.txt
```

## Menjalankan
```bash
python -m pytest -q           # 45 test, ~1 detik
```

### Dashboard interaktif
```bash
streamlit run app.py
```
5 tab: struktur 3D (dataset sintetis atau MOF CIF, pasangan terdekat disorot), penelusuran Divide & Conquer langkah demi langkah (3D/2D), jalur greedy 3D + uji akurasi, benchmark, serta strip & rasio pembagian. Semua chart interaktif (Plotly). Komputasi berat berjalan setelah menekan tombol Jalankan.

> Catatan: `streamlit` menuntut `protobuf>=5.26`, sedangkan `tensorflow 2.16` menuntut `<5`. Di env yang juga berisi TensorFlow, pakai env terpisah untuk dashboard ini.

## Struktur
```
app.py                    pintu masuk dashboard: konfigurasi, sidebar, header, pilih tampilan
.streamlit/
  config.toml             tema (font Poppins, warna zinc)
  style.css               gaya halaman gaya shadcn/ui
utils/                    logika murni (tanpa plotly / streamlit)
  data.py                 generator titik, plant_close_pair, load_cif
  naive.py                Algo. 1 (Python murni) + versi NumPy
  greedy.py               Algo. 2 + versi berulang + pencatat jalur
  divide_conquer.py       Algo. 3, iteratif dengan stack buatan; strip "standard" / "paper"; trace
  baselines.py            KD-tree, jarak tetangga terdekat, k pasangan terdekat
  benchmark.py            timing, verifikasi, eksperimen skala / akurasi greedy / rasio pembagian
dashboard/                semua yang berbau UI
  theme.py                palet, font, gaya dasar chart
  charts_3d.py            model atom, penelusuran DnC 3D, jalur greedy
  charts_2d.py            proyeksi DnC, histogram, benchmark, skala, greedy, strip, rasio
  components.py           card, detail_rows, note, show_chart
  cache.py                pemuatan dan analisis data (di-cache)
  config.py               konstanta
  context.py              AppContext: dataset aktif yang dibagikan ke semua tampilan
  views/                  satu file per tampilan (structure, divide_conquer, greedy, benchmark, strips)
tests/                    pytest
data/                     CIF dataset pcl (3856 atom, = dataset No. 8 paper)
references/               paper referensi (PDF tidak ikut repo)
```

## Asumsi interpretasi paper
- **Greedy (Algo. 2)**: shortestDis awal = inf sehingga kubus langkah pertama mencakup semua titik; pasangan terbaik hanya diperbarui saat jarak lebih pendek; jalan berhenti bila kubus kosong. Detail di docstring `greedy.py`.
- **DnC strip "paper"**: paper tidak menyebut pusat filter Y dan Z. Dipakai filter isolasi (buang titik tanpa tetangga dengan selisih < d pada sumbu itu), lalu brute force sisa titik. Hanya pemotongan ini yang menjamin hasil benar.
- **DnC strip "standard"**: strip |x − x_mid| < d, urut menurut Y, bandingkan dengan titik berikutnya selama selisih Y < d.
- Semua algoritma membandingkan jarak kuadrat; akar diambil di akhir. Jarak lintas batas periodik CIF tidak dihitung (seperti paper).
- Kerapatan atom dijaga konstan di eksperimen skala (`box_for`).
- `load_cif`: pakai `ase` bila ada, jika tidak parser bawaan (hanya CIF P1).
