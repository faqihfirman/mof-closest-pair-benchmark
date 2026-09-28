"""Tema visual dashboard (gaya shadcn/ui): palet, font, dan gaya dasar semua chart Plotly."""
from __future__ import annotations

import plotly.graph_objects as go

# ---------------------------------------------------------------- palet (gaya shadcn/ui: zinc + aksen Tailwind)
TRANSPARENT = "rgba(0,0,0,0)"
PANEL_COLOR = "#ffffff"
GRID_COLOR = "#f4f4f5"
AXIS_COLOR = "#e4e4e7"
MUTED_TEXT = "#a1a1aa"
BODY_TEXT = "#71717a"
BRIGHT_TEXT = "#09090b"  # teks utama
ACCENT = "#2563eb"       # blue-600: warna data utama
ALERT = "#ef4444"        # merah: pasangan terdekat / pelanggaran
WARN = "#f59e0b"         # oranye: strip / peringatan
OK = "#16a34a"           # hijau
VIOLET = "#7c3aed"
FONT_FAMILY = "Poppins, ui-sans-serif, system-ui, sans-serif"
ALGO_COLORS = {
    "naive": ALERT, "naive_vec": WARN, "greedy_1x": "#eab308", "greedy_rep": "#a16207",
    "dc_standard": ACCENT, "dc_paper": VIOLET, "kdtree": OK,
}
ALGO_LABELS = {
    "naive": "Naive", "naive_vec": "Naive (NumPy)", "greedy_1x": "Greedy",
    "greedy_rep": "Greedy berulang", "dc_standard": "Divide & Conquer", "dc_paper": "DnC (strip paper)",
    "kdtree": "KD-tree",
}
DIST_COLORS = {"uniform": ACCENT, "clustered": WARN, "lattice_jitter": OK}
ELEMENT_COLORS = {
    "H": "#38bdf8", "C": "#0d9488", "N": "#4f46e5", "O": "#ef4444", "F": "#22c55e", "B": "#f97316",
    "Si": "#eab308", "Al": "#a855f7", "V": "#ec4899", "Cu": "#d97706", "In": "#e11d48",
}
ELEMENT_SIZES = {"H": 2.0, "C": 3.0, "N": 3.2, "O": 3.2, "F": 3.0, "B": 3.2}
METAL_SIZE = 6.0  # elemen yang tidak ada di ELEMENT_SIZES (logam) digambar lebih besar
# Warna jarak tetangga: dekat = merah/oranye (bahaya), jauh = biru tua yang tenang.
# Transisi oranye -> biru dibuat sangat pendek: campuran RGB keduanya menghasilkan cokelat keabuan.
NEAREST_NEIGHBOR_SCALE = [[0.0, ALERT], [0.2, WARN], [0.23, "#93c5fd"], [0.55, "#3b82f6"], [1.0, "#1e3a8a"]]
BACKGROUND_ATOM = "rgba(59,130,246,0.75)"   # atom latar (greedy): biru muda, bukan abu-abu
INACTIVE_ATOM = "rgba(165,180,252,0.85)"    # atom di luar subset (DnC): lavender
# scrollZoom mati: scroll halaman tidak boleh "dibajak" chart (zoom lewat toolbar / pinch).
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
# 3D: scroll / pinch untuk zoom masuk ke struktur (juga saat fullscreen).
PLOTLY_CONFIG_3D = {**PLOTLY_CONFIG, "scrollZoom": True}
MODEBAR_STYLE = dict(bgcolor="rgba(255,255,255,0.9)", color="#a1a1aa", activecolor=BRIGHT_TEXT)


def label(algo_key: str) -> str:
    return ALGO_LABELS.get(algo_key, algo_key)


def color(algo_key: str) -> str:
    return ALGO_COLORS.get(algo_key, BODY_TEXT)


def rgba(hex_color: str, alpha: float) -> str:
    """'#22d3ee' + alpha -> 'rgba(34,211,238,alpha)'."""
    hex_digits = hex_color.lstrip("#")
    red, green, blue = (int(hex_digits[offset:offset + 2], 16) for offset in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def hover_style() -> dict:
    return dict(bgcolor=PANEL_COLOR, bordercolor=AXIS_COLOR,
                font=dict(family=FONT_FAMILY, color=BRIGHT_TEXT, size=11))


def style(fig: go.Figure, height: int = 360, title: str | None = None,
          show_legend: bool = True, legend_bottom: bool = False) -> go.Figure:
    """Tema gelap transparan untuk chart 2D."""
    title_settings = {}
    if title:
        title_settings["title"] = dict(text=title, font=dict(size=14, color=BRIGHT_TEXT),
                                       x=0.005, xanchor="left", y=0.97)
    if legend_bottom:
        legend_position = dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    else:
        legend_position = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
    fig.update_layout(
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT, height=height, showlegend=show_legend,
        font=dict(family=FONT_FAMILY, size=12, color=BODY_TEXT),
        margin=dict(l=56, r=20, t=50 if title else 20, b=48),
        legend=dict(bgcolor=TRANSPARENT, font=dict(size=12, color=BRIGHT_TEXT), **legend_position),
        hoverlabel=hover_style(), modebar=MODEBAR_STYLE, **title_settings,
    )
    axis_style = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=AXIS_COLOR, automargin=True,
                      tickfont=dict(size=11), title_font=dict(size=12, color=BODY_TEXT), title_standoff=8)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def style_subplot_titles(fig: go.Figure, title_count: int) -> None:
    """Judul subplot dibuat make_subplots sebagai anotasi pertama; samakan gayanya."""
    for annotation in fig.layout.annotations[:title_count]:
        annotation.font = dict(size=12, color=BODY_TEXT, family=FONT_FAMILY)


# ---------------------------------------------------------------- 3D
def _axis_3d(title: str) -> dict:
    return dict(title=dict(text=title, font=dict(size=10, color=MUTED_TEXT)), showbackground=False,
                gridcolor="#e4e4e7", zeroline=False, showspikes=False,
                color=MUTED_TEXT, tickfont=dict(size=9), linecolor=AXIS_COLOR)


def style_3d(fig: go.Figure, height: int = 640) -> go.Figure:
    """Scene 3D tanpa latar: sumbu tipis, kamera orbit, sudut kamera dipertahankan antar-rerun."""
    fig.update_layout(
        paper_bgcolor=TRANSPARENT, height=height, margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",  # jangan reset kamera saat slider digeser
        modebar=MODEBAR_STYLE,
        font=dict(family=FONT_FAMILY, size=12, color=BODY_TEXT), hoverlabel=hover_style(),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7", borderwidth=1, x=0.015, y=0.975,
                    font=dict(size=12, color=BRIGHT_TEXT), itemsizing="constant"),
        scene=dict(bgcolor=TRANSPARENT, aspectmode="data", dragmode="orbit",
                   xaxis=_axis_3d("x (Å)"), yaxis=_axis_3d("y (Å)"), zaxis=_axis_3d("z (Å)"),
                   camera=dict(eye=dict(x=1.45, y=1.45, z=0.85))),
    )
    return fig
