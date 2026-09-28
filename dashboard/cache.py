"""Pemuatan dan analisis data (di-cache oleh Streamlit)."""
from __future__ import annotations

import time

import numpy as np
import streamlit as st
from scipy.spatial import cKDTree

from utils.baselines import k_closest_pairs, nearest_neighbor_distances
from utils.data import box_for, generate_points, load_cif, plant_close_pair
from utils.divide_conquer import closest_pair_dc

from .config import CIF_PATH, SOURCE_MOF


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
