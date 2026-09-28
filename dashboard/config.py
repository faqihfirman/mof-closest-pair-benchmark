"""Konstanta dashboard."""
from pathlib import Path

CIF_PATH = Path(__file__).resolve().parent.parent / "data" / "pcl_v1-on15-3D_1-oe107.cif"
SOURCE_SYNTHETIC = "Sintetis"
SOURCE_MOF = "MOF pcl (CIF)"
DISTRIBUTIONS = ["uniform", "clustered", "lattice_jitter"]
MOF_BOND_RADIUS = 1.75  # Å: batas gambar ikatan kovalen pada dataset MOF
VIEWS = ["Struktur 3D", "Divide & Conquer", "Greedy", "Benchmark", "Strip & rasio"]
LOGO_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            '<circle cx="6" cy="17" r="3"/><circle cx="18" cy="7" r="3"/><path d="M8.5 15.2 15.5 8.8"/></svg>')
