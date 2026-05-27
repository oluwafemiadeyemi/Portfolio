"""
shared/design_system.py
=======================
Award-winning UI design system for the MIT AI & Data Science Portfolio.

Import in any dashboard:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from shared.design_system import apply_theme, kpi_card, section_header, chart_layout, risk_badge
"""

from __future__ import annotations
import streamlit as st
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour Tokens
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":           "#0E1117",
    "surface":      "#161B27",
    "surface2":     "#1E2538",
    "border":       "#2A3350",
    "primary":      "#4F8EF7",
    "primary_dark": "#2D6FDB",
    "success":      "#00C896",
    "warning":      "#FFB020",
    "danger":       "#FF5A65",
    "purple":       "#9B7FEA",
    "teal":         "#00D4C8",
    "text":         "#F0F4FF",
    "text_muted":   "#7A8BAD",
    "text_dim":     "#4A5568",
}

# Plotly colour sequence for consistent charts
PLOTLY_COLORS = [
    "#4F8EF7", "#00C896", "#FFB020", "#FF5A65",
    "#9B7FEA", "#00D4C8", "#F97316", "#A3E635",
]


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root overrides ──────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.stApp {{
    background-color: {bg};
}}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {surface} 0%, {surface2} 100%) !important;
    border-right: 1px solid {border} !important;
}}

section[data-testid="stSidebar"] .stRadio label {{
    color: {text_muted} !important;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 6px 0;
    transition: color 0.2s;
}}

section[data-testid="stSidebar"] .stRadio label:hover {{
    color: {primary} !important;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {{
    color: {text} !important;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

/* ── Main content ────────────────────────────────────────────── */
.main .block-container {{
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}}

/* ── Headings ────────────────────────────────────────────────── */
h1 {{
    color: {text} !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem !important;
}}

h2 {{
    color: {text} !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}

h3 {{
    color: {text_muted} !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s, transform 0.15s;
}}

[data-testid="stMetric"]:hover {{
    border-color: {primary};
    transform: translateY(-1px);
}}

[data-testid="stMetricLabel"] {{
    color: {text_muted} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

[data-testid="stMetricValue"] {{
    color: {text} !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
}}

[data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {primary} 0%, {primary_dark} 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.5rem 1.25rem;
    transition: opacity 0.2s, transform 0.15s;
    letter-spacing: 0.01em;
}}

.stButton > button:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
}}

/* ── Selectbox / Multiselect ─────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {surface2} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
    color: {text} !important;
}}

/* ── Dataframe ───────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {border};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {surface};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {border};
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    color: {text_muted} !important;
    font-weight: 500;
    font-size: 0.88rem;
}}

.stTabs [aria-selected="true"] {{
    background: {primary} !important;
    color: white !important;
}}

/* ── Expander ────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
    color: {text} !important;
    font-weight: 500;
}}

/* ── Divider ─────────────────────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {border};
    margin: 1.5rem 0;
}}

/* ── Info / Warning / Success boxes ─────────────────────────── */
.stAlert {{
    border-radius: 10px !important;
    border-left-width: 4px !important;
}}

/* ── KPI Card ────────────────────────────────────────────────── */
.kpi-card {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}}

.kpi-card:hover {{
    border-color: var(--kpi-accent, {primary});
    transform: translateY(-2px);
}}

.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--kpi-accent, {primary});
    border-radius: 14px 14px 0 0;
}}

.kpi-icon {{
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
    display: block;
}}

.kpi-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {text};
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.2rem;
}}

.kpi-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.kpi-delta {{
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.35rem;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}

.kpi-delta.up   {{ color: {success}; }}
.kpi-delta.down {{ color: {danger}; }}
.kpi-delta.flat {{ color: {text_muted}; }}

/* ── Section Header ──────────────────────────────────────────── */
.section-header {{
    margin: 2rem 0 1rem 0;
}}

.section-header h2 {{
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: {text} !important;
    margin: 0 !important;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}}

.section-header p {{
    font-size: 0.85rem;
    color: {text_muted};
    margin: 0.2rem 0 0 0;
}}

.section-divider {{
    height: 2px;
    background: linear-gradient(90deg, {primary} 0%, transparent 100%);
    border-radius: 2px;
    margin-top: 0.5rem;
}}

/* ── Risk / Status Badges ────────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.badge-critical {{ background: rgba(255,90,101,0.15); color: {danger}; border: 1px solid rgba(255,90,101,0.3); }}
.badge-high     {{ background: rgba(255,176,32,0.15); color: {warning}; border: 1px solid rgba(255,176,32,0.3); }}
.badge-medium   {{ background: rgba(79,142,247,0.15); color: {primary}; border: 1px solid rgba(79,142,247,0.3); }}
.badge-low      {{ background: rgba(0,200,150,0.15); color: {success}; border: 1px solid rgba(0,200,150,0.3); }}
.badge-info     {{ background: rgba(155,127,234,0.15); color: {purple}; border: 1px solid rgba(155,127,234,0.3); }}

/* ── Page hero banner ────────────────────────────────────────── */
.page-hero {{
    background: linear-gradient(135deg, {surface} 0%, {surface2} 100%);
    border: 1px solid {border};
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}

.page-hero::after {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,142,247,0.08) 0%, transparent 70%);
}}

.page-hero h1 {{
    font-size: 1.6rem !important;
    margin-bottom: 0.3rem !important;
}}

.page-hero p {{
    color: {text_muted};
    font-size: 0.9rem;
    max-width: 600px;
    margin: 0;
}}
""".format(**COLORS)


def apply_theme() -> None:
    """Inject the global CSS theme into the Streamlit app."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Card Component
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(
    label: str,
    value: str,
    icon: str = "📊",
    delta: Optional[str] = None,
    delta_direction: str = "flat",   # "up", "down", "flat"
    accent: str = None,
) -> None:
    """Render a styled KPI card.

    Parameters
    ----------
    label : str
        Short metric label (e.g. "AUC-ROC")
    value : str
        Pre-formatted value string (e.g. "0.947")
    icon : str
        Emoji icon
    delta : str, optional
        Change indicator (e.g. "+2.3%")
    delta_direction : "up" | "down" | "flat"
        Determines colour: up=green, down=red, flat=muted
    accent : str, optional
        Hex colour for the top accent bar (defaults to primary blue)
    """
    accent_color = accent or COLORS["primary"]
    delta_html = ""
    if delta:
        arrow = "▲" if delta_direction == "up" else ("▼" if delta_direction == "down" else "—")
        delta_html = f'<div class="kpi-delta {delta_direction}">{arrow} {delta}</div>'

    st.markdown(
        f"""
        <div class="kpi-card" style="--kpi-accent:{accent_color}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section Header
# ─────────────────────────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a styled section header with gradient underline."""
    icon_html = f"{icon} " if icon else ""
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{icon_html}{title}</h2>
            {subtitle_html}
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
def page_hero(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a full-width hero banner for the top of each page."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div class="page-hero">
            <h1>{icon_html}{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Risk Badge
# ─────────────────────────────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    """Return HTML for a coloured risk badge. Embed in st.markdown(..., unsafe_allow_html=True)."""
    level_lower = level.lower()
    cls = {
        "critical": "badge-critical",
        "high":     "badge-high",
        "elevated": "badge-high",
        "medium":   "badge-medium",
        "moderate": "badge-medium",
        "low":      "badge-low",
        "safe":     "badge-low",
        "info":     "badge-info",
    }.get(level_lower, "badge-info")
    return f'<span class="badge {cls}">{level}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Chart Theme
# ─────────────────────────────────────────────────────────────────────────────
def chart_layout(
    title: str = "",
    height: int = 380,
    show_legend: bool = True,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> dict:
    """Return a Plotly layout dict for consistent dark chart styling.

    Usage:
        fig.update_layout(**chart_layout("My Chart", height=400))
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter", size=14, color=COLORS["text"]),
            x=0,
            xanchor="left",
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter", color=COLORS["text_muted"], size=12),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11)),
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"], size=10),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11)),
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"], size=10),
            zeroline=False,
        ),
        legend=dict(
            visible=show_legend,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=12, r=12, t=40, b=12),
        colorway=PLOTLY_COLORS,
        hoverlabel=dict(
            bgcolor=COLORS["surface2"],
            bordercolor=COLORS["border"],
            font=dict(family="Inter", size=12, color=COLORS["text"]),
        ),
    )


def gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    suffix: str = "%",
    thresholds: Optional[list] = None,
    height: int = 260,
):
    """Create a styled gauge chart figure.

    Parameters
    ----------
    thresholds : list of (max_val, color) tuples
        e.g. [(25, '#00C896'), (50, '#FFB020'), (75, '#FF5A65'), (100, '#FF5A65')]
    """
    import plotly.graph_objects as go

    if thresholds is None:
        thresholds = [
            (max_val * 0.25, COLORS["success"]),
            (max_val * 0.50, COLORS["warning"]),
            (max_val * 0.75, COLORS["danger"]),
            (max_val,        COLORS["danger"]),
        ]

    steps = []
    prev = min_val
    for thresh, color in thresholds:
        steps.append({"range": [prev, thresh], "color": color + "22"})
        prev = thresh

    # Bar colour based on value position
    ratio = (value - min_val) / (max_val - min_val + 1e-9)
    bar_color = COLORS["success"] if ratio < 0.35 else (COLORS["warning"] if ratio < 0.65 else COLORS["danger"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": COLORS["text"], "family": "Inter"}},
        title={"text": title, "font": {"size": 13, "color": COLORS["text_muted"], "family": "Inter"}},
        gauge={
            "axis": {
                "range": [min_val, max_val],
                "tickwidth": 1,
                "tickcolor": COLORS["border"],
                "tickfont": {"size": 9, "color": COLORS["text_muted"]},
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
        },
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=10),
        font=dict(family="Inter"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar branding block
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_brand(project_name: str, project_icon: str = "🤖", tagline: str = "") -> None:
    """Render a consistent branded sidebar header."""
    st.sidebar.markdown(
        f"""
        <div style="padding:0.75rem 0 1rem 0; border-bottom:1px solid {COLORS['border']}; margin-bottom:1rem;">
            <div style="font-size:2rem; margin-bottom:0.3rem;">{project_icon}</div>
            <div style="font-size:1rem; font-weight:700; color:{COLORS['text']}; letter-spacing:-0.02em;">
                {project_name}
            </div>
            <div style="font-size:0.75rem; color:{COLORS['text_muted']}; margin-top:0.15rem;">
                {tagline}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div style="margin-top:auto; padding-top:1rem; font-size:0.72rem; color:{COLORS['text_dim']};">
            MIT Applied AI & Data Science<br>
            Oluwafemi Adeyemi
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
def empty_state(message: str = "No data to display", icon: str = "📭") -> None:
    """Render a centred empty state message."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:3rem 1rem; color:{COLORS['text_muted']};">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">{icon}</div>
            <div style="font-size:0.95rem; font-weight:500;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
