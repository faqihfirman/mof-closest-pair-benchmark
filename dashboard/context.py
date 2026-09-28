"""Konteks aplikasi yang dibagikan ke semua tampilan (menggantikan puluhan argumen)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AppContext:
    """Dataset aktif beserta hasil analisisnya."""

    source: str                   # sumber data (Sintetis / MOF pcl)
    seed: int
    threshold: float              # batas validitas (Å)
    points: np.ndarray            # koordinat atom (N, 3)
    symbols: list[str] | None     # simbol elemen (hanya untuk CIF)
    analysis: dict                # keluaran analyze_structure()
    atom_count: int
    shortest_distance: float      # jarak pasangan terdekat (Å)
    closest_pair: tuple[int, int]
    is_invalid: bool              # True bila shortest_distance < threshold
