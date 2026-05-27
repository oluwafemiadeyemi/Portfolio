"""
Enterprise Brand Intelligence Dashboard — Streamlit multi-page application.
Provides overview, aspect analysis, crisis detection, competitive intel,
topic explorer, and review search pages.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brand Intelligence Platform",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
_portfolio_root = Path(__file__).resolve().parents[2]
if str(_portfolio_root) not in sys.path:
    sys.path.insert(0, str(_portfolio_root))
try:
    from shared.design_system import (
        apply_theme, kpi_card, section_header, chart_layout,
        page_hero, sidebar_brand, risk_badge, gauge_chart, COLORS,
    )
    apply_theme()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project-specific CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .alert-critical { background: rgba(255,90,101,0.08); border-left: 3px solid #FF5A65; padding: 10px 14px; border-radius:0 8px 8px 0; }
    .alert-warning  { background: rgba(255,176,32,0.08); border-left: 3px solid #FFB020; padding: 10px 14px; border-radius:0 8px 8px 0; }
    .badge-positive { background: rgba(0,200,150,0.15); color: #00C896; padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-weight:700; }
    .badge-negative { background: rgba(255,90,101,0.15); color: #FF5A65; padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-weight:700; }
    .badge-neutral  { background: rgba(122,139,173,0.15); color: #7A8BAD; padding: 2px 10px; border-radius: 20px; font-size:0.75rem; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load and enrich data from all available sources."""
    try:
        from src.data_loader import load_all_data
        df = load_all_data(str(_BASE_DIR))
    except Exception as exc:
        st.warning(f"Data loading error: {exc}. Using synthetic data.")
        from src.data_loader import generate_synthetic_reviews
        df = generate_synthetic_reviews(n=2000)

    # Compute sentiment if not already present
    if "compound_sentiment" not in df.columns:
        try:
            from src.features import extract_sentiment_lexicon
            scores = df["text"].fillna("").apply(extract_sentiment_lexicon)
            df["compound_sentiment"] = scores.apply(lambda s: s["compound"])
            df["pos_sentiment"] = scores.apply(lambda s: s["pos"])
            df["neg_sentiment"] = scores.apply(lambda s: s["neg"])
        except Exception as exc:
            st.warning(f"VADER scoring failed: {exc}")
            df["compound_sentiment"] = 0.0
            df["pos_sentiment"] = 0.0
            df["neg_sentiment"] = 0.0

    # Sentiment label
    def _label(c: float) -> str:
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["compound_sentiment"].apply(_label)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp("2022-01-01"))
    return df


@st.cache_data(ttl=3600)
def load_topics(df: pd.DataFrame):
    """Fit topic model on a sample of the data."""
    try:
        from src.topic_modeling import run_topic_pipeline
        sample = df.sample(min(500, len(df)), random_state=42)
        topics, assignments, trending = run_topic_pipeline(sample, n_topics=10)
        return topics, assignments, trending, sample
    except Exception as exc:
        logger.warning("Topic modeling failed: %s", exc)
        return {}, [], pd.DataFrame(), df.head(0)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered DataFrame."""
    st.sidebar.title("🎛 Filters")
    st.sidebar.markdown("---")

    # Date range
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max_date - timedelta(days=90)
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, max_date

    # Source filter
    sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox("Source", sources)

    # Brand filter
    brands = ["All"] + sorted(df["business_name"].dropna().unique().tolist()[:50])
    selected_brand = st.sidebar.selectbox("Brand", brands)

    # Apply filters
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    if selected_source != "All":
        mask &= df["source"] == selected_source
    if selected_brand != "All":
        mask &= df["business_name"] == selected_brand

    filtered = df[mask].copy()
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Reviews", f"{len(filtered):,}")

    return filtered, start_date, end_date, selected_source, selected_brand


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame) -> None:
    """Overview page with KPIs and 30-day sentiment trend."""
    st.title("📊 Brand Intelligence Overview")
    st.markdown("Real-time brand reputation monitoring — powered by NLP")
    st.markdown("---")

    # KPI cards
    avg_sentiment = float(df["compound_sentiment"].mean()) if "compound_sentiment" in df.columns else 0.0
    total_reviews = len(df)

    try:
        from src.competitive import compute_nps_proxy
        nps_data = compute_nps_proxy(df)
        nps_score = nps_data.get("nps", 0.0)
    except Exception:
        nps_score = 0.0

    # Count anomalies as proxy for crisis alerts
    try:
        from src.models import detect_anomaly_series
        anomaly_df = detect_anomaly_series(df, score_col="compound_sentiment", date_col="date")
        crisis_alerts = int(anomaly_df["is_anomaly"].sum())
    except Exception:
        crisis_alerts = 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        color = "#38a169" if avg_sentiment > 0.05 else ("#e53e3e" if avg_sentiment < -0.05 else "#718096")
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{avg_sentiment:+.3f}</div>'
            f'<div class="kpi-label">Avg Sentiment Score</div></div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{total_reviews:,}</div>'
            f'<div class="kpi-label">Total Reviews</div></div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        nps_color = "#38a169" if nps_score > 0 else "#e53e3e"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{nps_color}">{nps_score:+.0f}</div>'
            f'<div class="kpi-label">NPS Proxy Score</div></div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        alert_color = "#e53e3e" if crisis_alerts > 0 else "#38a169"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{alert_color}">{crisis_alerts}</div>'
            f'<div class="kpi-label">Crisis Alerts</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 30-day sentiment trend
    st.subheader("30-Day Sentiment Trend")
    df_trend = df.copy()
    df_trend["date"] = pd.to_datetime(df_trend["date"])
    cutoff = df_trend["date"].max() - pd.Timedelta(days=30)
    df_trend = df_trend[df_trend["date"] >= cutoff]

    if len(df_trend) > 0 and "compound_sentiment" in df_trend.columns:
        daily = (
            df_trend.groupby(df_trend["date"].dt.date)
            .agg(avg_sentiment=("compound_sentiment", "mean"), count=("text", "count"))
            .reset_index()
        )
        daily.columns = ["date", "avg_sentiment", "count"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["avg_sentiment"],
            mode="lines+markers", name="Avg Sentiment",
            line=dict(color="#4299e1", width=2),
            fill="tozeroy", fillcolor="rgba(66,153,225,0.1)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.05, line_dash="dot", line_color="green", opacity=0.4, annotation_text="Positive threshold")
        fig.add_hline(y=-0.05, line_dash="dot", line_color="red", opacity=0.4, annotation_text="Negative threshold")
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Sentiment Score",
            height=350, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment distribution pie + rating distribution
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Sentiment Distribution")
        if "sentiment_label" in df.columns:
            dist = df["sentiment_label"].value_counts().reset_index()
            dist.columns = ["sentiment", "count"]
            color_map = {"positive": "#48bb78", "neutral": "#a0aec0", "negative": "#fc8181"}
            fig_pie = px.pie(
                dist, values="count", names="sentiment",
                color="sentiment", color_discrete_map=color_map,
                hole=0.45,
            )
            fig_pie.update_layout(height=300, template="plotly_dark", margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Rating Distribution")
        if "rating" in df.columns:
            rating_counts = df["rating"].round(0).astype(int).value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            fig_bar = px.bar(
                rating_counts, x="rating", y="count",
                color="rating", color_continuous_scale="RdYlGn",
                labels={"rating": "Star Rating", "count": "Count"},
            )
            fig_bar.update_layout(height=300, template="plotly_dark",
                                   margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Aspect Analysis
# ---------------------------------------------------------------------------

def page_aspect_analysis(df: pd.DataFrame) -> None:
    """Aspect analysis page with radar chart and top reviews per aspect."""
    st.title("🎯 Aspect-Based Sentiment Analysis")
    st.markdown("---")

    aspects = ["cleanliness", "staff", "value", "location", "amenities"]
    aspect_cols = [f"aspect_{a}_compound" for a in aspects]
    available_aspects = [a for a, c in zip(aspects, aspect_cols) if c in df.columns]
    available_cols = [f"aspect_{a}_compound" for a in available_aspects]

    if not available_aspects:
        st.info("Aspect scores not yet computed. Run the feature pipeline first.")
        return

    col_radar, col_bar = st.columns([1, 1])

    with col_radar:
        st.subheader("Aspect Radar Chart")
        means = [float(df[c].mean()) for c in available_cols]
        # Normalize to 0-1 for radar
        normalized = [(m + 1) / 2 for m in means]  # compound is -1 to 1

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=available_aspects + [available_aspects[0]],
            fill="toself",
            fillcolor="rgba(66,153,225,0.2)",
            line=dict(color="#4299e1", width=2),
            name="Aspect Scores",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400, template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        st.subheader("Aspect Score Comparison")
        aspect_df = pd.DataFrame({
            "aspect": available_aspects,
            "avg_score": [float(df[c].mean()) for c in available_cols],
        })
        color_fn = lambda s: "#48bb78" if s > 0.05 else ("#fc8181" if s < -0.05 else "#a0aec0")
        aspect_df["color"] = aspect_df["avg_score"].apply(color_fn)
        fig_h = px.bar(
            aspect_df, x="avg_score", y="aspect", orientation="h",
            color="avg_score", color_continuous_scale="RdYlGn",
            labels={"avg_score": "Avg Compound Score", "aspect": "Aspect"},
            range_color=[-1, 1],
        )
        fig_h.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_h.update_layout(height=400, template="plotly_dark",
                             margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)

    # Aspect trend over time
    st.subheader("Aspect Sentiment Trend Over Time")
    selected_aspect = st.selectbox("Select Aspect", available_aspects)
    aspect_col = f"aspect_{selected_aspect}_compound"

    df_trend = df.copy()
    df_trend["date"] = pd.to_datetime(df_trend["date"])
    daily_aspect = (
        df_trend.groupby(df_trend["date"].dt.to_period("W").apply(lambda p: p.start_time))
        [aspect_col].mean().reset_index()
    )
    daily_aspect.columns = ["date", "avg_score"]

    fig_trend = px.line(daily_aspect, x="date", y="avg_score",
                        title=f"Weekly {selected_aspect.capitalize()} Sentiment")
    fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_trend.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Top positive and negative reviews
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.subheader(f"Top Positive — {selected_aspect.capitalize()}")
        pos_reviews = (
            df[df[aspect_col] > 0.3][["text", aspect_col]]
            .sort_values(aspect_col, ascending=False)
            .head(5)
        )
        for _, row in pos_reviews.iterrows():
            st.markdown(
                f'<div class="badge-positive">+{row[aspect_col]:.3f}</div> '
                f'{str(row["text"])[:200]}...',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

    with col_neg:
        st.subheader(f"Top Negative — {selected_aspect.capitalize()}")
        neg_reviews = (
            df[df[aspect_col] < -0.3][["text", aspect_col]]
            .sort_values(aspect_col, ascending=True)
            .head(5)
        )
        for _, row in neg_reviews.iterrows():
            st.markdown(
                f'<div class="badge-negative">{row[aspect_col]:.3f}</div> '
                f'{str(row["text"])[:200]}...',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Crisis Detection
# ---------------------------------------------------------------------------

def page_crisis_detection(df: pd.DataFrame) -> None:
    """Crisis detection page with anomaly timeline and alert log."""
    st.title("🚨 Crisis Detection & Early Warning System")
    st.markdown("Detects sentiment drops 72+ hours before they escalate")
    st.markdown("---")

    try:
        from src.models import detect_anomaly_series
        anomaly_df = detect_anomaly_series(
            df, score_col="compound_sentiment", date_col="date",
            window_hours=7, threshold=0.1, freq="D",
        )
    except Exception as exc:
        st.error(f"Anomaly detection error: {exc}")
        return

    if anomaly_df.empty:
        st.info("No time-series data available for anomaly detection.")
        return

    # Timeline
    st.subheader("Sentiment Timeline with Anomalies")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomaly_df["date"], y=anomaly_df["avg_sentiment"],
        mode="lines", name="Avg Sentiment",
        line=dict(color="#4299e1", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=anomaly_df["date"], y=anomaly_df["rolling_baseline"],
        mode="lines", name="Rolling Baseline",
        line=dict(color="#a0aec0", dash="dash", width=1),
    ))

    anomalies = anomaly_df[anomaly_df["is_anomaly"] == True]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["avg_sentiment"],
            mode="markers", name="Anomaly Alert",
            marker=dict(color="red", size=12, symbol="x"),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Sentiment Score",
        height=400, template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alert log
    col_log, col_breakdown = st.columns([1, 1])
    with col_log:
        st.subheader("Alert Log")
        if not anomalies.empty:
            alert_df = anomalies[["date", "avg_sentiment", "rolling_baseline", "drop"]].copy()
            alert_df["severity"] = alert_df["drop"].apply(
                lambda d: "CRITICAL" if d > 0.3 else "WARNING"
            )
            alert_df = alert_df.sort_values("date", ascending=False)
            st.dataframe(
                alert_df.style.applymap(
                    lambda v: "background-color: #fed7d7" if v == "CRITICAL" else "background-color: #fefcbf",
                    subset=["severity"],
                ),
                use_container_width=True,
            )
        else:
            st.success("No anomalies detected in the current filter window.")

    with col_breakdown:
        st.subheader("Sentiment Drop Distribution")
        fig_hist = px.histogram(
            anomaly_df, x="drop", nbins=30,
            color_discrete_sequence=["#4299e1"],
            labels={"drop": "Sentiment Drop", "count": "Frequency"},
        )
        fig_hist.add_vline(x=0.1, line_dash="dash", line_color="orange",
                           annotation_text="Warning threshold")
        fig_hist.add_vline(x=0.3, line_dash="dash", line_color="red",
                           annotation_text="Critical threshold")
        fig_hist.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Summary stats
    total_days = len(anomaly_df)
    total_alerts = int(anomaly_df["is_anomaly"].sum())
    alert_rate = total_alerts / max(total_days, 1) * 100

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Days Monitored", total_days)
    m2.metric("Total Alerts", total_alerts)
    m3.metric("Alert Rate", f"{alert_rate:.1f}%")
    m4.metric("Avg Sentiment Drop", f"{float(anomaly_df['drop'].mean()):.4f}")


# ---------------------------------------------------------------------------
# Page: Competitive Intel
# ---------------------------------------------------------------------------

def page_competitive_intel(df: pd.DataFrame) -> None:
    """Competitive intelligence page with benchmarking and gap analysis."""
    st.title("🏆 Competitive Intelligence")
    st.markdown("---")

    try:
        from src.competitive import benchmark_brands, market_share_of_voice, compute_nps_proxy
    except ImportError as exc:
        st.error(f"Import error: {exc}")
        return

    if "compound_sentiment" not in df.columns:
        st.info("Sentiment scores not computed yet.")
        return

    bench = benchmark_brands(df)
    sov = market_share_of_voice(df)
    nps_data = compute_nps_proxy(df)

    # Top brands bar chart
    st.subheader("Brand Sentiment Ranking")
    top_brands = bench.head(20)
    fig_bench = px.bar(
        top_brands, x="avg_sentiment", y="brand",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        labels={"avg_sentiment": "Avg Sentiment", "brand": "Brand"},
        text="avg_sentiment",
    )
    fig_bench.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bench.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_bench.update_layout(height=500, template="plotly_dark",
                            margin=dict(l=20, r=80, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_bench, use_container_width=True)

    # Market share of voice + NPS
    col_sov, col_nps = st.columns([2, 1])

    with col_sov:
        st.subheader("Market Share of Voice")
        top_sov = sov.head(15)
        fig_pie = px.pie(
            top_sov, values="share_pct", names="brand",
            hole=0.3,
            title="Review Volume Share",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=400, template="plotly_dark",
                              margin=dict(t=30, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_nps:
        st.subheader("NPS Breakdown")
        nps_score = nps_data.get("nps", 0)
        promoters = nps_data.get("promoters_pct", 0)
        passives = nps_data.get("passives_pct", 0)
        detractors = nps_data.get("detractors_pct", 0)

        st.metric("NPS Score", f"{nps_score:+.0f}")
        nps_breakdown = pd.DataFrame({
            "Category": ["Promoters", "Passives", "Detractors"],
            "Percentage": [promoters, passives, detractors],
        })
        fig_nps = px.bar(
            nps_breakdown, x="Category", y="Percentage",
            color="Category",
            color_discrete_map={"Promoters": "#48bb78", "Passives": "#ecc94b", "Detractors": "#fc8181"},
        )
        fig_nps.update_layout(height=300, template="plotly_dark",
                              showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_nps, use_container_width=True)

    # Gap analysis table
    st.subheader("Competitive Gap Analysis")
    if len(bench) >= 2:
        target = bench.iloc[0]["brand"]
        competitors = bench.iloc[1:6]["brand"].tolist()
        try:
            from src.competitive import compute_gap_analysis
            gap_df = compute_gap_analysis(bench, target, competitors)
            if not gap_df.empty:
                st.dataframe(gap_df, use_container_width=True)
        except Exception as exc:
            st.info(f"Gap analysis unavailable: {exc}")

    # Full benchmark table
    st.subheader("Full Brand Benchmark Table")
    display_cols = [c for c in ["brand", "avg_sentiment", "review_count", "avg_rating", "rank"]
                    if c in bench.columns]
    st.dataframe(bench[display_cols].head(30), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Topic Explorer
# ---------------------------------------------------------------------------

def page_topic_explorer(df: pd.DataFrame) -> None:
    """Topic explorer with topic cards, trend chart, and sample reviews."""
    st.title("🔬 Topic Explorer")
    st.markdown("---")

    with st.spinner("Fitting topic model (cached after first run)..."):
        topics, assignments, trending, sample_df = load_topics(df)

    if not topics:
        st.info("Topic model fitting failed or returned no topics.")
        return

    # Topic cards
    st.subheader("Discovered Topics")
    topic_items = list(topics.items())
    cols_per_row = 3
    for i in range(0, len(topic_items), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(row_cols):
            if i + j < len(topic_items):
                topic_id, words = topic_items[i + j]
                with col:
                    st.markdown(
                        f"**{topic_id}**\n\n"
                        + " • ".join(words[:8]),
                    )
                    count = assignments.count(list(topics.keys()).index(topic_id)) if assignments else 0
                    st.caption(f"{count} documents")

    # Topic trend
    if not trending.empty and "trend_score" in trending.columns:
        st.subheader("Topic Trend Scores")
        trend_plot = trending.head(10).copy()
        trend_plot["topic_label"] = trend_plot["topic_id"].astype(str).apply(
            lambda t: list(topics.keys())[int(t)] if int(t) < len(topics) else f"Topic {t}"
        )
        fig_trend = px.bar(
            trend_plot, x="trend_score", y="topic_label",
            orientation="h",
            color="trend_score",
            color_continuous_scale="RdYlGn",
            labels={"trend_score": "Trend Score (recent vs prior)", "topic_label": "Topic"},
        )
        fig_trend.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_trend.update_layout(height=400, template="plotly_dark",
                                margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Sample reviews per topic
    st.subheader("Sample Reviews by Topic")
    topic_labels = list(topics.keys())
    selected_topic = st.selectbox("Select Topic", topic_labels)
    topic_idx = topic_labels.index(selected_topic)

    if assignments and len(assignments) == len(sample_df):
        topic_reviews = sample_df[[a == topic_idx for a in assignments]].head(5)
        if not topic_reviews.empty:
            for _, row in topic_reviews.iterrows():
                sentiment = row.get("sentiment_label", "neutral")
                badge_class = f"badge-{sentiment}"
                st.markdown(
                    f'<span class="{badge_class}">{sentiment}</span> {str(row["text"])[:300]}',
                    unsafe_allow_html=True,
                )
                st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Review Search
# ---------------------------------------------------------------------------

def page_review_search(df: pd.DataFrame) -> None:
    """Full-text review search with sentiment filtering."""
    st.title("🔍 Review Search")
    st.markdown("---")

    col_search, col_filter = st.columns([3, 1])
    with col_search:
        query = st.text_input("Search reviews", placeholder="e.g. WiFi, parking, cleanliness...")
    with col_filter:
        sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])

    if not query and sentiment_filter == "All":
        results = df.copy()
    elif query:
        mask = df["text"].str.lower().str.contains(query.lower(), na=False)
        results = df[mask].copy()
    else:
        results = df.copy()

    if sentiment_filter != "All" and "sentiment_label" in results.columns:
        results = results[results["sentiment_label"] == sentiment_filter]

    st.markdown(f"**{len(results):,} results found**")

    if len(results) == 0:
        st.info("No reviews match the search criteria.")
        return

    # Sort options
    sort_options = {
        "Most Recent": ("date", False),
        "Most Positive": ("compound_sentiment", False),
        "Most Negative": ("compound_sentiment", True),
        "Highest Rating": ("rating", False),
    }
    sort_label = st.selectbox("Sort by", list(sort_options.keys()))
    sort_col, ascending = sort_options[sort_label]
    if sort_col in results.columns:
        results = results.sort_values(sort_col, ascending=ascending)

    # Display results
    display_results = results.head(50)
    for _, row in display_results.iterrows():
        sentiment = row.get("sentiment_label", "neutral")
        badge_class = f"badge-{sentiment}"
        rating = row.get("rating", "N/A")
        brand = row.get("business_name", "Unknown")
        date_str = str(row.get("date", ""))[:10]
        compound = row.get("compound_sentiment", 0.0)

        st.markdown(
            f'<span class="{badge_class}">{sentiment} ({compound:+.3f})</span> '
            f'⭐ {rating} | {brand} | {date_str}',
            unsafe_allow_html=True,
        )
        st.markdown(str(row["text"])[:400])
        st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    # Sidebar filters
    filtered_df, start_date, end_date, selected_source, selected_brand = build_sidebar(df)

    # Page navigation
    pages = {
        "Overview": page_overview,
        "Aspect Analysis": page_aspect_analysis,
        "Crisis Detection": page_crisis_detection,
        "Competitive Intel": page_competitive_intel,
        "Topic Explorer": page_topic_explorer,
        "Review Search": page_review_search,
    }

    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

    # Render selected page
    pages[selected_page](filtered_df)

    # Footer
    st.markdown("---")
    st.caption(
        "Enterprise Brand Intelligence Platform | Powered by VADER NLP + NMF Topic Modeling | "
        f"Data: {len(filtered_df):,} reviews | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


if __name__ == "__main__":
    main()
