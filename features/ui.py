"""features/ui.py — ParselyFi shared design system.

Version-agnostic visual layer (works on any Streamlit): a single ``inject_css``
call plus reusable presentational helpers (hero, KPI metrics, tier badges,
section headers). Import-safe; touches no secrets. Every tab calls
``ui.inject_css()`` once near the top so the whole app shares one look.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

# --- design tokens ----------------------------------------------------------
GREEN = "#10b981"
GREEN_DK = "#059669"
BLUE = "#0a7cff"
VIOLET = "#8e75b2"
INK = "#0f172a"
MUTE = "#64748b"
LINE = "#e2e8f0"
BG_SOFT = "#f1f5f9"

# Tier -> (text color, soft background) for evidence-cited scoring rubrics.
TIER_STYLE = {
    "very high": ("#065f46", "#d1fae5"),
    "high": ("#166534", "#dcfce7"),
    "medium": ("#92400e", "#fef3c7"),
    "low": ("#9f1239", "#ffe4e6"),
    "no match": ("#475569", "#e2e8f0"),
}

_CSS = """
<style>
:root { --pf-green:#10b981; --pf-ink:#0f172a; --pf-mute:#64748b; --pf-line:#e2e8f0; }
/* Tighten the top padding and widen content a touch. */
.block-container { padding-top: 2.2rem !important; max-width: 1280px; }
/* Headings */
h1, h2, h3 { letter-spacing: -0.01em; }
/* Hero band */
.pf-hero {
  background: radial-gradient(1200px 360px at 85% -40%, rgba(52,211,153,.18), transparent 60%),
              linear-gradient(135deg, #0b1220 0%, #0f1b2e 60%, #14253f 100%);
  border-radius: 18px; padding: 26px 30px; margin: 2px 0 18px;
  box-shadow: 0 18px 40px rgba(2,8,20,.18); color: #eaf2ff;
}
.pf-hero h1 { color:#eaf2ff; font-weight:900; font-size: 2.0rem; margin:0; letter-spacing:-.02em; }
.pf-hero .pf-sub { color:#9fb3c8; font-size:1.02rem; margin-top:6px; }
.pf-hero .pf-chips { margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
.pf-chip { font-size:.78rem; font-weight:600; padding:5px 11px; border-radius:999px;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); color:#cfe3f5; }
.pf-chip b { color:#34d399; }
/* Cards */
.pf-card { background:#fff; border:1px solid var(--pf-line); border-radius:14px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
/* Section header */
.pf-sec { display:flex; align-items:center; gap:10px; margin:6px 0 2px; }
.pf-sec .pf-bar { width:4px; height:26px; border-radius:6px; background:var(--pf-green); }
.pf-sec .pf-t { font-weight:800; font-size:1.18rem; color:var(--pf-ink); }
.pf-sec .pf-d { color:var(--pf-mute); font-size:.92rem; margin-left:2px; }
/* Tier badges */
.pf-badge { display:inline-block; font-weight:700; font-size:.74rem; padding:3px 10px;
  border-radius:999px; letter-spacing:.01em; }
/* st.metric polish */
[data-testid="stMetric"] { background:#fff; border:1px solid var(--pf-line);
  border-radius:14px; padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
[data-testid="stMetricLabel"] p { color:var(--pf-mute); font-weight:600; font-size:.82rem; }
[data-testid="stMetricValue"] { font-weight:800; letter-spacing:-.02em; }
/* Buttons */
.stButton button[kind="primary"] { border-radius:10px; font-weight:700;
  box-shadow:0 6px 16px rgba(16,185,129,.28); }
.stButton button { border-radius:10px; }
/* Dataframes / data_editor */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border:1px solid var(--pf-line); border-radius:12px; overflow:hidden; }
/* Tabs */
[data-baseweb="tab-list"] { gap:4px; }
[data-baseweb="tab"] { font-weight:600; }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet once per session (idempotent)."""
    if st.session_state.get("_pf_css_done"):
        return
    st.session_state["_pf_css_done"] = True
    st.html(_CSS)


def hero(title: str, subtitle: str = "", chips: Optional[Sequence[str]] = None) -> None:
    """Branded gradient hero band. ``chips`` may contain inline <b> for accents."""
    chip_html = ""
    if chips:
        chip_html = '<div class="pf-chips">' + "".join(
            f'<span class="pf-chip">{c}</span>' for c in chips
        ) + "</div>"
    sub = f'<div class="pf-sub">{subtitle}</div>' if subtitle else ""
    st.html(
        f'<div class="pf-hero"><h1>{title}</h1>{sub}{chip_html}</div>'
    )


def section(title: str, desc: str = "") -> None:
    """A green-accent section header."""
    d = f'<span class="pf-d">{desc}</span>' if desc else ""
    st.html(f'<div class="pf-sec"><div class="pf-bar"></div>'
            f'<span class="pf-t">{title}</span>{d}</div>')


def tier_badge(tier: Optional[str]) -> str:
    """Return an HTML badge for a rubric tier (Very High / High / Medium / Low)."""
    t = (tier or "").strip().lower()
    fg, bg = TIER_STYLE.get(t, ("#475569", "#e2e8f0"))
    label = (tier or "—").strip() or "—"
    return f'<span class="pf-badge" style="color:{fg};background:{bg}">{label}</span>'


def kpi_row(items: Iterable[Tuple]) -> None:
    """Render a row of st.metric cards.

    ``items`` is an iterable of (label, value) or (label, value, delta) or
    (label, value, delta, help) tuples.
    """
    items = list(items)
    if not items:
        return
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        label, value = it[0], it[1]
        delta = it[2] if len(it) > 2 else None
        helptxt = it[3] if len(it) > 3 else None
        col.metric(label, value, delta=delta, help=helptxt)


def score_color(pct: float) -> str:
    """Green→amber→red ramp for a 0-1 score (used for ProgressColumn-ish hints)."""
    if pct >= 0.75:
        return GREEN
    if pct >= 0.5:
        return "#f59e0b"
    return "#ef4444"
