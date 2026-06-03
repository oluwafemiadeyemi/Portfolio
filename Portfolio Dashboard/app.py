"""
Unified Portfolio Dashboard
17 AI/ML projects · One Streamlit app · Fortune 500 grade
"""

import sys
from pathlib import Path

# Ensure utils/ and views/ are importable
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st
from utils.registry import PROJECTS, CATEGORIES, PROJECT_BY_ID
from utils.ui import inject_css, section_header

st.set_page_config(
    page_title="AI/ML Portfolio — Oluwafemi Adeyemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── State ─────────────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"


def navigate(page_id: str):
    st.session_state.page = page_id
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Header
    st.markdown("""
    <div style="padding:16px 8px 12px">
      <div style="font-size:1.15rem;font-weight:700;color:#E6EDF3">🧠 AI/ML Portfolio</div>
      <div style="font-size:0.72rem;color:#8B949E;margin-top:2px">Oluwafemi Adeyemi · MIT Applied AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Home button
    if st.button("🏠  Portfolio Overview", use_container_width=True,
                  type="primary" if st.session_state.page == "home" else "secondary"):
        navigate("home")

    st.markdown("<div style='margin:8px 0;border-top:1px solid #21262D'></div>", unsafe_allow_html=True)

    # Projects by category
    for cat in CATEGORIES:
        cat_projs = [p for p in PROJECTS if p["category"] == cat]
        if not cat_projs:
            continue
        st.markdown(
            f"<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:#8B949E;padding:8px 4px 4px;margin-top:4px'>{cat}</div>",
            unsafe_allow_html=True,
        )
        for proj in cat_projs:
            is_active = st.session_state.page == proj["id"]
            label = f"{proj['icon']}  P{proj['num']:02d} · {proj['short']}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"sb_{proj['id']}", use_container_width=True, type=btn_type):
                navigate(proj["id"])

    # Footer
    st.markdown("""
    <div style="padding:16px 8px 8px;margin-top:16px;border-top:1px solid #21262D">
      <div style="font-size:0.68rem;color:#8B949E">
        <div>🦙 Llama 3.2 · Ollama</div>
        <div style="margin-top:3px">17 projects · 24 GB+ real data</div>
        <div style="margin-top:3px">FastAPI · Streamlit · ONNX</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────

page = st.session_state.page

if page == "home":
    from views.home import render
    render(navigate)

elif page == "brand":
    from views.p01_brand import render
    render()

elif page == "fraud":
    from views.p02_fraud import render
    render()

elif page == "mortgage":
    from views.p03_mortgage import render
    render()

elif page == "people":
    from views.p04_people import render
    render()

elif page == "parkinsons":
    from views.p05_parkinsons import render
    render()

elif page == "supply_chain":
    from views.p06_supply_chain import render
    render()

elif page == "retail":
    from views.p07_retail import render
    render()

elif page == "ergonomics":
    from views.p08_ergonomics import render
    render()

elif page == "clv":
    from views.p09_clv import render
    render()

elif page == "ppe":
    from views.p10_ppe import render
    render()

elif page == "marketing":
    from views.p11_marketing import render
    render()

elif page == "automotive":
    from views.p12_automotive import render
    render()

elif page == "loan":
    from views.p13_loan import render
    render()

elif page == "malaria":
    from views.p14_malaria import render
    render()

elif page == "emotion":
    from views.p15_emotion import render
    render()

elif page == "music":
    from views.p16_music import render
    render()

elif page == "reviews":
    from views.p17_reviews import render
    render()
