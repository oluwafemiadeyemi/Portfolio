"""Shared UI helpers — light theme portfolio dashboard."""
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path


# ── CSS ───────────────────────────────────────────────────────────────────────
# Light theme: white background, slate text, blue accents

GLOBAL_CSS = """
<style>
  /* ── Base / App ── */
  html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #f8fafc !important;
    color: #0f172a !important;
  }
  .block-container { padding-top: 1.5rem !important; max-width: 1100px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stButton > button {
    background: #334155 !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
    text-align: left !important;
    width: 100% !important;
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #fff !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
  }
  .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 4px 0;
    line-height: 1.2;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
  }
  .metric-delta { font-size: 0.8rem; margin-top: 4px; }

  /* ── Project header ── */
  .proj-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid var(--accent, #2563eb);
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .proj-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
    color: #0f172a;
  }
  .proj-tagline { color: #64748b; margin: 5px 0 0 0; font-size: 0.92rem; }

  /* ── Badges ── */
  .buyer-badge {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.73rem;
    margin: 2px 3px;
    color: #334155;
    font-weight: 500;
  }

  /* ── Section header ── */
  .section-head {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin: 20px 0 6px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    font-weight: 600;
  }

  /* ── Home project cards ── */
  .home-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
    transition: box-shadow 0.15s, border-color 0.15s;
  }
  .home-card:hover {
    border-color: #93c5fd;
    box-shadow: 0 4px 12px rgba(37,99,235,0.08);
  }
  .home-card-num  { font-size: 0.64rem; color: #94a3b8; font-weight: 600; }
  .home-card-name { font-size: 0.92rem; font-weight: 600; margin: 4px 0; color: #1e293b; }
  .home-card-tag  { font-size: 0.73rem; color: #64748b; }

  /* ── Status pills ── */
  .pill-green { background: #dcfce7; color: #15803d; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 500; }
  .pill-blue  { background: #dbeafe; color: #1d4ed8; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 500; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background: #f1f5f9 !important; border-radius: 8px; padding: 3px; }
  .stTabs [data-baseweb="tab"] {
    font-size: 0.84rem !important;
    color: #64748b !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
  }
  .stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    border: 1px solid #cbd5e1 !important;
    color: #374151 !important;
    background: #fff !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #eff6ff !important;
  }
  .stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #fff !important;
    border-color: #2563eb !important;
  }

  /* ── Inputs / Selects ── */
  [data-baseweb="select"] > div,
  [data-baseweb="input"] > div {
    background: #fff !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
  }
  .stSlider > label { color: #374151 !important; }

  div[data-testid="column"] { padding: 4px 6px; }
  h1, h2, h3 { color: #0f172a !important; }
  p, label, span { color: #374151; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Components ────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", color: str = "#2563eb"):
    delta_html = f'<div class="metric-delta" style="color:{color};font-weight:600">{delta}</div>' if delta else ""
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color}">{value}</div>
              {delta_html}
            </div>""",
        unsafe_allow_html=True,
    )


def project_header(proj: dict):
    buyers_html = " ".join(f'<span class="buyer-badge">{b}</span>' for b in proj["buyers"])
    stack_html  = " ".join(
        f'<span class="buyer-badge" style="border-color:{proj["color"]}60;color:{proj["color"]};background:{proj["color"]}0D">{s}</span>'
        for s in proj["stack"]
    )
    st.markdown(
        f"""<div class="proj-header" style="--accent:{proj['color']}">
              <div style="display:flex;align-items:center;gap:14px">
                <span style="font-size:2rem;line-height:1">{proj['icon']}</span>
                <div>
                  <h1 class="proj-title">{proj['name']}</h1>
                  <p class="proj-tagline">{proj['tagline']}</p>
                </div>
              </div>
              <div style="margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:4px">
                <span style="font-size:0.7rem;color:#94a3b8;margin-right:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Buyers</span>
                {buyers_html}
              </div>
              <div style="margin-top:8px;display:flex;flex-wrap:wrap;align-items:center;gap:4px">
                <span style="font-size:0.7rem;color:#94a3b8;margin-right:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Stack</span>
                {stack_html}
              </div>
              <div style="margin-top:10px;font-size:0.75rem;color:#94a3b8">
                📊 {proj['dataset']} &nbsp;·&nbsp; 🏷️ {proj['category']}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def section_header(title: str):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


def kpi_row(metrics: list):
    """Render a row of KPI metric cards. metrics = [(label, value, color?), ...]"""
    cols = st.columns(len(metrics))
    for col, (label, value, *rest) in zip(cols, metrics):
        color = rest[0] if rest else "#2563eb"
        with col:
            metric_card(label, value, color=color)


def load_metrics(proj: dict) -> dict:
    for candidate in [
        proj["path"] / "models" / "metrics.json",
        proj["path"] / "data" / "models" / "metrics.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def load_parquet_sample(proj: dict, filename: str, n: int = 500) -> pd.DataFrame:
    for candidate in [
        proj["path"] / "data" / "processed" / filename,
        proj["path"] / "models" / filename,
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            return df.sample(min(n, len(df)), random_state=42) if len(df) > n else df
    return pd.DataFrame()


def placeholder_chart(title: str = "No data available", color: str = "#2563eb") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=title, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color="#94a3b8"),
    )
    fig.update_layout(**_light_layout(250))
    return fig


# ── Light chart theme ─────────────────────────────────────────────────────────

def _light_layout(height: int = 300) -> dict:
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#374151", size=11, family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=35, b=35),
        height=height,
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", tickcolor="#94a3b8"),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", tickcolor="#94a3b8"),
        legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", font=dict(color="#374151")),
    )

# Keep old name for backward compat
CHART_LAYOUT = _light_layout(300)


def light_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(**_light_layout(height))
    return fig

# Alias
dark_fig = light_fig
