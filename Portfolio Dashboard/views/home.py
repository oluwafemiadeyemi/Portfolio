"""Home page — portfolio overview with all 17 project cards."""
import streamlit as st
import plotly.graph_objects as go
from utils.registry import PROJECTS, CATEGORIES
from utils.ui import section_header, dark_fig


def render(navigate):
    st.markdown("""
    <div style="text-align:center;padding:32px 0 16px">
      <h1 style="font-size:2.6rem;font-weight:800;margin:0">
        Enterprise AI/ML Portfolio
      </h1>
      <p style="color:#8B949E;font-size:1.05rem;margin:10px 0 0">
        17 production-grade data science projects · Real datasets · Fortune 500 targets
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    cols = st.columns(5)
    stats = [
        ("Projects", "17", "Across 7 domains"),
        ("Real Data", "24 GB+", "Actual production datasets"),
        ("Models Trained", "60+", "ML, DL, GenAI, Vision"),
        ("APIs Live", "17", "FastAPI + Streamlit"),
        ("LLM", "Llama 3.2", "Local · Zero API cost"),
    ]
    for col, (label, val, sub) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
                        padding:14px;text-align:center">
              <div style="font-size:1.6rem;font-weight:700;color:#58A6FF">{val}</div>
              <div style="font-size:0.75rem;font-weight:600;margin:2px 0">{label}</div>
              <div style="font-size:0.68rem;color:#8B949E">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project cards by category ─────────────────────────────────────────────
    for cat in CATEGORIES:
        cat_projects = [p for p in PROJECTS if p["category"] == cat]
        if not cat_projects:
            continue
        section_header(cat)
        cols = st.columns(min(len(cat_projects), 4))
        for col, proj in zip(cols, cat_projects):
            with col:
                clicked = st.button(
                    f"{proj['icon']} {proj['short']}",
                    key=f"home_btn_{proj['id']}",
                    use_container_width=True,
                    help=proj["tagline"],
                )
                if clicked:
                    navigate(proj["id"])
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #30363D;border-left:3px solid {proj['color']};
                            border-radius:6px;padding:8px 10px;margin-top:4px">
                  <div style="font-size:0.68rem;color:#8B949E">P{proj['num']:02d} · {proj['category']}</div>
                  <div style="font-size:0.72rem;color:#C9D1D9;margin-top:3px">{proj['tagline'][:60]}...</div>
                  <div style="font-size:0.65rem;color:{proj['color']};margin-top:5px">
                    {'  ·  '.join(proj['buyers'][:2])}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Tech stack radar ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Technology Coverage")
    col1, col2 = st.columns(2)

    with col1:
        cats   = ["ML/Tabular", "Deep Learning", "NLP/GenAI", "Computer Vision",
                  "Recommendation", "MLOps", "Explainability"]
        values = [10, 8, 7, 8, 5, 9, 9]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="#58A6FF", width=2),
            name="Coverage",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#0D1117",
                radialaxis=dict(visible=True, range=[0, 10],
                                gridcolor="#21262D", linecolor="#30363D",
                                tickfont=dict(color="#8B949E", size=9)),
                angularaxis=dict(gridcolor="#21262D", linecolor="#30363D",
                                 tickfont=dict(color="#E6EDF3", size=10)),
            ),
            paper_bgcolor="#161B22",
            showlegend=False,
            margin=dict(l=30, r=30, t=40, b=30),
            height=300,
            title=dict(text="AI/ML Stack Breadth", font=dict(color="#E6EDF3", size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        domains  = [p["category"] for p in PROJECTS]
        from collections import Counter
        cnt = Counter(domains)
        labels = list(cnt.keys())
        values = list(cnt.values())
        colors = ["#58A6FF", "#F39C12", "#E74C3C", "#1DB954", "#9B59B6", "#27AE60", "#E67E22"]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker=dict(colors=colors[:len(labels)], line=dict(color="#0D1117", width=2)),
            textfont=dict(color="#E6EDF3", size=11),
        ))
        fig2.update_layout(
            paper_bgcolor="#161B22",
            showlegend=True,
            legend=dict(bgcolor="#161B22", font=dict(color="#E6EDF3", size=10)),
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
            title=dict(text="Projects by Domain", font=dict(color="#E6EDF3", size=12)),
        )
        st.plotly_chart(fig2, use_container_width=True)
