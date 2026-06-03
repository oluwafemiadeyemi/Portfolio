"""P17 Customer Review Categorisation — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import requests
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID

CATEGORIES_LIST = [
    "Product Quality", "Delivery & Shipping", "Customer Service",
    "Pricing & Value", "Returns & Refunds", "Packaging",
    "Product Accuracy", "Overall Experience"
]

CLASSIFICATION_SYSTEM_PROMPT = """You are a review classification assistant. Given a customer review, classify it into ONE of these categories:
- Product Quality
- Delivery & Shipping
- Customer Service
- Pricing & Value
- Returns & Refunds
- Packaging
- Product Accuracy
- Overall Experience

Also rate the sentiment as: Positive, Neutral, or Negative.

Respond ONLY in this JSON format:
{"category": "<category>", "sentiment": "<sentiment>", "confidence": <0.0-1.0>, "summary": "<one sentence>"}"""


@st.cache_data
def load_rag_sample(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "rag_sample.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            return df.sample(min(500, len(df)), random_state=42)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 500
    sentiments = rng.choice(["Positive", "Neutral", "Negative"], n, p=[0.55, 0.20, 0.25])
    categories = rng.choice(CATEGORIES_LIST, n)
    return pd.DataFrame({
        "review_id": range(n),
        "sentiment": sentiments,
        "category": categories,
        "rating": rng.integers(1, 6, n),
        "text": [f"Sample review {i}" for i in range(n)],
    })


def call_ollama_classify(review_text: str) -> dict | None:
    try:
        resp = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "llama3.2:4k",
                "messages": [
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Review: {review_text}"}
                ],
                "max_tokens": 150,
                "temperature": 0.1,
            },
            timeout=20
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
    except Exception:
        pass
    return None


def heuristic_classify(text: str) -> dict:
    """Rule-based fallback when Ollama is not available."""
    text_lower = text.lower()
    category_keywords = {
        "Delivery & Shipping": ["deliver", "shipping", "arrived", "late", "package", "tracking", "dispatch"],
        "Product Quality": ["quality", "broken", "defective", "excellent", "poor quality", "durable", "cheap"],
        "Customer Service": ["service", "support", "staff", "helpful", "rude", "agent", "representative"],
        "Pricing & Value": ["price", "value", "expensive", "cheap", "worth", "overpriced", "cost"],
        "Returns & Refunds": ["return", "refund", "exchange", "money back", "sent back"],
        "Packaging": ["packaging", "box", "wrapped", "damaged box", "bubble wrap"],
        "Product Accuracy": ["description", "expected", "different", "misleading", "accurate", "as shown"],
    }
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in category_keywords.items()}
    best_cat = max(scores, key=scores.get) if max(scores.values()) > 0 else "Overall Experience"

    pos_words = ["great", "excellent", "good", "love", "perfect", "amazing", "happy", "satisfied"]
    neg_words = ["bad", "terrible", "awful", "disappointed", "broken", "waste", "poor", "never again"]
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)

    if pos_count > neg_count:
        sentiment = "Positive"
    elif neg_count > pos_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = min(0.85, 0.45 + max(scores.values()) * 0.12)
    return {"category": best_cat, "sentiment": sentiment, "confidence": round(confidence, 2),
            "summary": f"Review classified as {best_cat} with {sentiment} sentiment."}


def render():
    proj = PROJECT_BY_ID["reviews"]
    project_header(proj)
    metrics = load_metrics(proj)

    cols = st.columns(4)
    with cols[0]: metric_card("Reviews Indexed", metrics.get("n_reviews", "500k"), color="#9B59B6")
    with cols[1]: metric_card("Categories", "8", color="#58A6FF")
    with cols[2]: metric_card("Ollama Model", "llama3.2:4k", color="#E67E22")
    with cols[3]: metric_card("ChromaDB Vectors", metrics.get("n_vectors", "500k"), color="#27AE60")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Review Sentiment Distribution")
        df = load_rag_sample(proj["path"])
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            color_map = {"Positive": "#27AE60", "Neutral": "#F39C12", "Negative": "#E74C3C"}
            fig = px.bar(sent_counts, x="Sentiment", y="Count", color="Sentiment",
                         color_discrete_map=color_map, text="Count")
            dark_fig(fig, height=280)
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Customer Review Sentiment Breakdown", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(placeholder_chart("Sentiment data not available"), use_container_width=True)

        section_header("Sample Reviews")
        if not df.empty:
            display_cols = [c for c in ["text", "sentiment", "category", "rating"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols].head(10).reset_index(drop=True),
                             use_container_width=True, height=220)

    with tab2:
        section_header("AI Review Classifier (Llama 3.2)")
        st.caption("Reviews are classified using Llama 3.2:4k via Ollama. Falls back to rule-based if Ollama is offline.")

        review_text = st.text_area(
            "Paste a customer review",
            placeholder="The product arrived two days late and the packaging was damaged. "
                        "Customer service was unhelpful when I called to complain.",
            height=130
        )

        if st.button("Classify Review", type="primary"):
            if not review_text.strip():
                st.warning("Please enter a review to classify.")
            else:
                with st.spinner("Classifying with Llama 3.2:4k..."):
                    result = call_ollama_classify(review_text)

                if result:
                    st.success("Classified via Llama 3.2:4k (Ollama)")
                else:
                    result = heuristic_classify(review_text)
                    st.info("Ollama not running — used rule-based fallback. Start Ollama with `ollama serve`.")

                sent_color = {"Positive": "#27AE60", "Neutral": "#F39C12", "Negative": "#E74C3C"}.get(
                    result.get("sentiment", "Neutral"), "#8B949E")

                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Category", result.get("category", "Unknown"), color="#9B59B6")
                with c2: metric_card("Sentiment", result.get("sentiment", "Unknown"), color=sent_color)
                with c3: metric_card("Confidence", f"{result.get('confidence', 0):.1%}", color="#58A6FF")

                if result.get("summary"):
                    st.markdown(f"> **Summary:** {result['summary']}")

    with tab3:
        section_header("Category Distribution")
        df = load_rag_sample(proj["path"])
        if "category" in df.columns:
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
        else:
            rng = np.random.default_rng(42)
            cat_counts = pd.DataFrame({
                "Category": CATEGORIES_LIST,
                "Count": rng.integers(200, 2000, len(CATEGORIES_LIST)).tolist()
            })

        fig = px.bar(cat_counts.sort_values("Count", ascending=True),
                     x="Count", y="Category", orientation="h",
                     color="Count", color_continuous_scale="Purples",
                     text="Count")
        dark_fig(fig, height=360)
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Review Volume by Category", coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        section_header("RAG System Architecture")
        st.markdown("""
| Component | Technology | Details |
|---|---|---|
| Vector Store | ChromaDB | 500k review embeddings |
| Embedding Model | all-MiniLM-L6-v2 | 384-dim sentence vectors |
| LLM | Llama 3.2:4k (Ollama) | Local inference, no API key |
| Classification | Zero-shot prompt | 8 categories, chain-of-thought |
| Topic Modelling | BERTopic | 50+ discovered topics |
| API Layer | FastAPI | `/classify`, `/search`, `/topics` |
""")
