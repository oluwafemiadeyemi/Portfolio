"""
Shared UI Design System — applied across all 7 portfolio dashboards.
Inspired by: Nexora (gradients), Lukana (dark sidebar, metric cards), Glintly (imagery).
"""

import base64
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"

# ── Colour Palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#6366F1",   # Indigo
    "secondary":  "#8B5CF6",   # Violet
    "success":    "#10B981",   # Emerald
    "warning":    "#F59E0B",   # Amber
    "danger":     "#EF4444",   # Red
    "info":       "#3B82F6",   # Blue
    "bg":         "#F0F4FF",
    "card":       "#FFFFFF",
    "sidebar_bg": "#0F172A",
    "text":       "#1E293B",
    "muted":      "#64748B",
    "border":     "#E2E8F0",
}

# Per-project accent colours and gradients
PROJECT_THEMES = {
    "marketing": {
        "accent": "#6366F1",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #6366F1 100%)",
        "icon": "📣",
    },
    "automotive": {
        "accent": "#F59E0B",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #1C1400 55%, #D97706 100%)",
        "icon": "🚗",
    },
    "loans": {
        "accent": "#EF4444",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #1C0A0A 55%, #DC2626 100%)",
        "icon": "💳",
    },
    "malaria": {
        "accent": "#10B981",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #042F2E 55%, #059669 100%)",
        "icon": "🔬",
    },
    "emotion": {
        "accent": "#8B5CF6",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #1E0A3C 55%, #7C3AED 100%)",
        "icon": "😊",
    },
    "music": {
        "accent": "#06B6D4",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #041B2A 55%, #0891B2 100%)",
        "icon": "🎵",
    },
    "reviews": {
        "accent": "#F97316",
        "gradient": "linear-gradient(135deg, #0F172A 0%, #1C0A00 55%, #EA580C 100%)",
        "icon": "⭐",
    },
}

# ── Global CSS ────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App Background ── */
.stApp {
    background: linear-gradient(160deg, #F0F4FF 0%, #F8FAFC 50%, #EEF2FF 100%) !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

/* ── Main Content Padding ── */
.main .block-container {
    padding: 1.5rem 2.5rem 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: white !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06) !important;
    border: 1px solid #E2E8F0 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.15) !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #64748B !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important;
    padding: 4px !important;
    border-radius: 12px !important;
    gap: 4px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #64748B !important;
    transition: all 0.2s !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #6366F1 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
    font-weight: 600 !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    border: 1px solid #E2E8F0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
}

/* ── Select boxes & sliders ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border-color: #E2E8F0 !important;
    background: white !important;
}
.stSlider [data-testid="stSlider"] > div > div > div {
    background: #6366F1 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: white !important;
    border-radius: 12px !important;
    border: 2px dashed #6366F1 !important;
}

/* ── Divider ── */
hr {
    border-color: #E2E8F0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Plotly charts ── */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ── Card wrapper utility ── */
.portfolio-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid #E2E8F0;
    margin-bottom: 16px;
}

.portfolio-card:hover {
    box-shadow: 0 4px 20px rgba(99,102,241,0.12);
    transform: translateY(-1px);
    transition: all 0.2s;
}

/* ── KPI badge ── */
.kpi-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.kpi-badge.up   { background: #D1FAE5; color: #059669; }
.kpi-badge.down { background: #FEE2E2; color: #DC2626; }
</style>
"""


# ── Helper Functions ──────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


def hero_banner(project_key: str, title: str, subtitle: str,
                stats: list = None) -> str:
    """Full-width hero banner (Nexora/Lukana inspired) with optional stat pills."""
    theme = PROJECT_THEMES.get(project_key, PROJECT_THEMES["marketing"])
    img_path = ASSETS_DIR / f"{project_key}_hero.jpg"
    b64 = img_to_b64(img_path)

    if b64:
        bg_layer = f'url("data:image/jpeg;base64,{b64}") center/cover'
        overlay  = f'background:{theme["gradient"]}; opacity:0.88;'
    else:
        bg_layer = theme["gradient"]
        overlay  = ""

    stat_html = ""
    if stats:
        pills = "".join([
            f'<div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);'
            f'border-radius:12px;padding:10px 18px;text-align:center;">'
            f'<div style="font-size:1.4rem;font-weight:800;color:white;">{s[0]}</div>'
            f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.7);font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">{s[1]}</div>'
            f'</div>'
            for s in stats
        ])
        stat_html = f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:24px;">{pills}</div>'

    overlay_div = f'<div style="position:absolute;inset:0;border-radius:20px;{overlay}"></div>' if overlay else ""

    return f"""
    <div style="
        background: {bg_layer};
        border-radius: 20px;
        padding: 44px 48px 40px 48px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
        {overlay_div}
        <!-- Decorative circles (Nexora style) -->
        <div style="position:absolute;right:-20px;top:-20px;width:180px;height:180px;
            background:rgba(255,255,255,0.06);border-radius:50%;"></div>
        <div style="position:absolute;right:80px;bottom:-40px;width:120px;height:120px;
            background:rgba(255,255,255,0.04);border-radius:50%;"></div>
        <div style="position:absolute;right:30px;top:50%;width:60px;height:60px;
            background:rgba(255,255,255,0.08);border-radius:50%;transform:translateY(-50%);"></div>
        <!-- Content -->
        <div style="position:relative;z-index:1;max-width:700px;">
            <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
                border-radius:20px;padding:4px 14px;margin-bottom:16px;">
                <span style="font-size:1.2rem;">{theme['icon']}</span>
                <span style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.9);
                    text-transform:uppercase;letter-spacing:0.08em;">Enterprise AI Platform</span>
            </div>
            <h1 style="color:white;font-size:clamp(1.6rem,3vw,2.4rem);font-weight:800;
                margin:0 0 10px 0;line-height:1.15;
                text-shadow:0 2px 12px rgba(0,0,0,0.25);">{title}</h1>
            <p style="color:rgba(255,255,255,0.82);font-size:1rem;
                margin:0;font-weight:400;line-height:1.55;">{subtitle}</p>
            {stat_html}
        </div>
    </div>
    """


def kpi_card(title: str, value: str, delta: str = None,
             delta_up: bool = True, icon: str = "📈",
             accent: str = "#6366F1") -> str:
    """Styled KPI card matching Lukana dashboard style."""
    badge = ""
    if delta:
        cls = "up" if delta_up else "down"
        arrow = "↑" if delta_up else "↓"
        badge = f'<span class="kpi-badge {cls}">{arrow} {delta}</span>'
    return f"""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        border-top: 4px solid {accent};
        transition: transform 0.2s, box-shadow 0.2s;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <span style="
                font-size: 0.75rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 0.08em;
                color: #64748B;
            ">{title}</span>
            <span style="font-size: 1.6rem; opacity: 0.8;">{icon}</span>
        </div>
        <div style="
            font-size: 2rem; font-weight: 800;
            color: #1E293B; line-height: 1; margin-bottom: 10px;
        ">{value}</div>
        {badge}
    </div>
    """


def section_header(title: str, subtitle: str = "") -> str:
    """Section divider with title and optional subtitle."""
    sub = f'<p style="color:#64748B;font-size:0.9rem;margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin: 8px 0 20px 0;">
        <h3 style="
            font-size: 1.2rem; font-weight: 700;
            color: #1E293B; margin: 0;
        ">{title}</h3>
        {sub}
    </div>
    """


def sidebar_branding(project_name: str, project_key: str) -> str:
    """Sidebar logo + project name."""
    theme = PROJECT_THEMES.get(project_key, PROJECT_THEMES["marketing"])
    return f"""
    <div style="
        text-align: center; padding: 8px 0 24px 0;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 3rem; margin-bottom: 8px;
        ">{theme['icon']}</div>
        <div style="
            font-size: 0.95rem; font-weight: 700;
            color: white; line-height: 1.3;
        ">{project_name}</div>
        <div style="
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 2px 12px;
            font-size: 0.7rem;
            font-weight: 600;
            color: rgba(255,255,255,0.8);
            margin-top: 6px;
            letter-spacing: 0.05em;
        ">ENTERPRISE</div>
    </div>
    """


def status_pill(label: str, color: str = "#10B981") -> str:
    bg = color + "22"
    return f"""
    <span style="
        background: {bg}; color: {color};
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
        border: 1px solid {color}44;
    ">{label}</span>
    """


def apply_theme() -> None:
    """Call at the top of every dashboard app."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "color": "#1E293B"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "colorway": ["#6366F1","#10B981","#F59E0B","#EF4444","#3B82F6","#8B5CF6","#EC4899"],
        "xaxis": {
            "gridcolor": "#F1F5F9", "linecolor": "#E2E8F0",
            "tickfont": {"size": 11, "color": "#64748B"},
            "title_font": {"size": 12, "color": "#94A3B8"},
        },
        "yaxis": {
            "gridcolor": "#F1F5F9", "linecolor": "#E2E8F0",
            "tickfont": {"size": 11, "color": "#64748B"},
            "title_font": {"size": 12, "color": "#94A3B8"},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#E2E8F0", "borderwidth": 1,
            "font": {"size": 12},
        },
        "margin": {"l": 16, "r": 16, "t": 40, "b": 16},
        "hoverlabel": {
            "bgcolor": "white",
            "bordercolor": "#E2E8F0",
            "font": {"size": 13, "family": "Inter"},
        },
    }
}


def style_plotly_fig(fig, height: int = 420, title: str = None):
    """Apply consistent Plotly styling to any figure."""
    import plotly.graph_objects as go
    fig.update_layout(
        font_family="Inter, sans-serif",
        font_color="#1E293B",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=16, r=16, t=48 if title else 24, b=16),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E2E8F0", borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor="#E2E8F0",
            font_size=13, font_family="Inter",
        ),
        title=dict(text=title, font=dict(size=15, weight=700, color="#1E293B")) if title else None,
    )
    fig.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                     tickfont=dict(size=11, color="#64748B"))
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                     tickfont=dict(size=11, color="#64748B"))
    return fig
