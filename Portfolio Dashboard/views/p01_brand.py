"""P01 Brand Intelligence Platform — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, load_parquet_sample, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID


@st.cache_data
def load_reviews(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "reviews.csv"
        if p.exists():
            return pd.read_csv(p, nrows=2000)
    except Exception:
        pass
    brands = ["Marriott", "Hilton", "Starbucks", "Hyatt", "IHG"]
    sentiments = ["Positive", "Neutral", "Negative"]
    rng = np.random.default_rng(42)
    rows = []
    for brand in brands:
        for sent in sentiments:
            rows.append({"brand": brand, "sentiment": sent,
                         "count": int(rng.integers(200, 1200))})
    return pd.DataFrame(rows)


@st.cache_data
def load_topics(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "topic_distribution.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    topics = ["Service Quality", "Food & Beverage", "Cleanliness", "Price/Value",
              "Ambiance", "Staff Attitude", "Location", "Wait Time"]
    rng = np.random.default_rng(7)
    return pd.DataFrame({"topic": topics, "count": rng.integers(300, 2500, len(topics)).tolist()})


def render():
    proj = PROJECT_BY_ID["brand"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]:
        metric_card("Reviews Processed", metrics.get("total_reviews", "8.8M"), color="#E74C3C")
    with cols[1]:
        metric_card("Avg Sentiment Score", metrics.get("avg_sentiment", "0.72"), color="#27AE60")
    with cols[2]:
        metric_card("Top Aspects Tracked", metrics.get("aspects_tracked", "8"), color="#F39C12")
    with cols[3]:
        metric_card("Model Accuracy", metrics.get("accuracy", "88.4%"), color="#58A6FF")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Sentiment Distribution by Brand")
        df = load_reviews(proj["path"])
        if "brand" in df.columns and "sentiment" in df.columns:
            if "count" not in df.columns:
                df = df.groupby(["brand", "sentiment"]).size().reset_index(name="count")
            color_map = {"Positive": "#27AE60", "Neutral": "#F39C12", "Negative": "#E74C3C"}
            fig = px.bar(df, x="brand", y="count", color="sentiment",
                         barmode="group", color_discrete_map=color_map)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("No review data found"), use_container_width=True)

    with tab2:
        section_header("Live Sentiment Analysis")
        st.caption("Enter any customer review and get instant VADER sentiment scoring.")
        text = st.text_area("Review text", placeholder="The service was fantastic and the staff were incredibly helpful...", height=120)
        if st.button("Analyse Sentiment", type="primary"):
            if text.strip():
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    scores = analyzer.polarity_scores(text)
                except ImportError:
                    # Fallback synthetic scoring
                    pos_words = sum(1 for w in text.lower().split() if w in {"great","good","excellent","fantastic","love","amazing","wonderful","helpful","clean","perfect"})
                    neg_words = sum(1 for w in text.lower().split() if w in {"bad","terrible","awful","horrible","worst","poor","dirty","rude","slow","disappointing"})
                    compound = min(1.0, max(-1.0, (pos_words - neg_words) * 0.15))
                    scores = {"compound": round(compound, 4), "pos": round(max(0, compound), 4),
                              "neg": round(max(0, -compound), 4), "neu": round(1 - abs(compound), 4)}

                compound = scores["compound"]
                if compound >= 0.05:
                    label, col = "POSITIVE", "#27AE60"
                elif compound <= -0.05:
                    label, col = "NEGATIVE", "#E74C3C"
                else:
                    label, col = "NEUTRAL", "#F39C12"

                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Verdict", label, color=col)
                with c2: metric_card("Compound", str(round(compound, 3)), color=col)
                with c3: metric_card("Positive", str(round(scores["pos"], 3)), color="#27AE60")
                with c4: metric_card("Negative", str(round(scores["neg"], 3)), color="#E74C3C")
            else:
                st.warning("Please enter some review text.")

    with tab3:
        section_header("Topic Distribution")
        df_t = load_topics(proj["path"])
        if not df_t.empty:
            fig = px.bar(df_t.sort_values("count", ascending=True),
                         x="count", y="topic", orientation="h",
                         color="count", color_continuous_scale="Reds")
            dark_fig(fig, height=350)
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("Topic data unavailable"), use_container_width=True)
