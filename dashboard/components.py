"""Komponen UI kecil yang dipakai berulang (gaya shadcn/ui)."""
from __future__ import annotations

import streamlit as st

from . import theme


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
