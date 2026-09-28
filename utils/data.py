"""Generator titik 3D, penyisip pasangan dekat, dan loader CIF."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Kerapatan acuan (atom per Å^3): 300 atom dalam kotak 30 Å, seperti demo motivasi.
DEFAULT_DENSITY = 300 / 30**3


def box_for(n: int, density: float = DEFAULT_DENSITY) -> tuple[float, float, float]:
    """Sisi kubus agar kerapatan atom tetap konstan saat N membesar."""
    side = (n / density) ** (1 / 3)
    return (side, side, side)


def generate_points(
    n: int,
    dist: str = "uniform",
    box: tuple[float, float, float] = (30, 30, 30),
    seed: int = 0,
) -> np.ndarray:
    """Bangkitkan `n` titik 3D dalam `box`.

    - "uniform": titik acak seragam.
    - "clustered": campuran Gaussian (atom mengumpul di gugus, ruang lain kosong).
    - "lattice_jitter": grid kristal + noise kecil, meniru MOF simetris.
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(box, dtype=float)
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size=(n, 3)) * b
    if dist == "clustered":
        k = max(3, n // 60)
        centers = rng.uniform(0.0, 1.0, size=(k, 3)) * b
        spacing = b.mean() / k ** (1 / 3)
        sigma = 0.15 * spacing
        labels = rng.integers(0, k, size=n)
        pts = centers[labels] + rng.normal(0.0, sigma, size=(n, 3))
        return np.mod(pts, b)  # bungkus ke dalam kotak, tanpa clipping (hindari titik kembar)
    if dist == "lattice_jitter":
        m = math.ceil(n ** (1 / 3))
        axes = [np.arange(m) * (b[a] / m) for a in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        grid = grid[rng.permutation(len(grid))[:n]]
        spacing = (b / m).min()
        return grid + rng.normal(0.0, 0.05 * spacing, size=(n, 3))
    raise ValueError(f"dist tidak dikenal: {dist!r}")


def plant_close_pair(
    points: np.ndarray, distance: float, seed: int = 0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Sisipkan satu atom baru berjarak tepat `distance` dari atom yang ada.

    Agar pasangan itu pasti yang terdekat (ground truth diketahui), atom asli yang
    berjarak <= `distance` dari atom lain dipindahkan acak ke tempat kosong.
    Mengembalikan (titik baru berukuran n+1, (indeks_jangkar, indeks_atom_baru)).
    """
    rng = np.random.default_rng(seed)
    pts = np.array(points, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    for _ in range(10_000):  # pindahkan atom yang terlalu rapat
        pairs = cKDTree(pts).query_pairs(r=distance * 1.0001, output_type="ndarray")
        if len(pairs) == 0:
            break
        move = np.unique(pairs[:, 1])
        pts[move] = rng.uniform(lo, hi, size=(len(move), 3))
    else:
        raise RuntimeError("gagal membersihkan pasangan yang lebih dekat dari `distance`")

    tree = cKDTree(pts)
    for _ in range(100_000):
        anchor = int(rng.integers(len(pts)))
        u = rng.normal(size=3)
        new = pts[anchor] + distance * u / np.linalg.norm(u)
        nearest, _ = tree.query(new, k=1)
        if nearest >= distance * 0.9999:  # jangkar sendiri berjarak ~distance
            others, _ = tree.query(new, k=2)
            if others[1] > distance * 1.0001:
                return np.vstack([pts, new]), (anchor, len(pts))
    raise RuntimeError("gagal menemukan posisi untuk atom baru")


def load_cif(path: str | Path, return_symbols: bool = False):
    """Ambil koordinat kartesian atom dari file CIF.

    Pakai `ase` bila terpasang (menangani simetri). Jika tidak, pakai parser
    sederhana yang HANYA benar untuk CIF berformat P1 (semua atom sudah tertulis).
    Jarak lintas batas periodik tidak dihitung (sama seperti paper).
    """
    path = Path(path)
    try:
        from ase.io import read  # type: ignore

        atoms = read(str(path))
        pts = np.asarray(atoms.get_positions(), dtype=float)
        return (pts, atoms.get_chemical_symbols()) if return_symbols else pts
    except ImportError:
        pass  # ase tidak ada: pakai parser bawaan

    text = path.read_text()
    cell = {}
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        m = re.search(rf"_cell_(?:length_|angle_)?{key}\s+([-\d.eE+]+)", text)
        if not m:
            raise ValueError(f"parameter sel {key} tidak ditemukan di {path.name}")
        cell[key] = float(m.group(1))
    if not re.search(r"'?P\s*1'?", text.split("loop_")[0]):
        print("Peringatan: CIF bukan P1, atom hasil simetri tidak diekspansi (pasang `ase`).")

    a, b, c = cell["a"], cell["b"], cell["c"]
    al, be, ga = (math.radians(cell[k]) for k in ("alpha", "beta", "gamma"))
    # Matriks sel standar (baris = vektor kisi a, b, c)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    lattice = np.array([[a, 0, 0], [b * math.cos(ga), b * math.sin(ga), 0], [cx, cy, cz]])

    lines = text.splitlines()
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop_header = False
    for line in lines:
        s = line.strip()
        if s == "loop_":
            if rows:  # loop atom sudah selesai (loop berikutnya, mis. ikatan, diabaikan)
                break
            cols, in_loop_header = [], True
            continue
        if in_loop_header and s.startswith("_"):
            cols.append(s)
            continue
        in_loop_header = False
        if cols and "_atom_site_fract_x" in cols and s and not s.startswith(("_", "#")):
            rows.append(s.split())
    ix = [cols.index(f"_atom_site_fract_{k}") for k in "xyz"]
    isym = cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in cols else 0
    frac = np.array([[float(r[i]) for i in ix] for r in rows if len(r) >= len(cols)])
    symbols = [r[isym] for r in rows if len(r) >= len(cols)]
    pts = frac @ lattice
    return (pts, symbols) if return_symbols else pts
