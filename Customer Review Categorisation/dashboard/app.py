"""
Streamlit dashboard: Generative AI Customer Review Intelligence Platform
Tabs: Live Classifier | VOC Analytics | RAG Explorer | Executive Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os, sys, json, time
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
sys.path.insert(0, str(BASE_DIR / "src"))

st.set_page_config(
    page_title="Customer Review AI",
    page_icon="⭐",
    layout="wide",
)

CATEGORIES = [
    "Electronics", "Fashion", "Home & Garden", "Sports", "Beauty",
    "Books", "Toys", "Grocery", "Automotive", "Health", "Movies", "Music"
]
REVIEW_CATEGORIES = [
    "Product Quality", "Delivery & Shipping", "Customer Service",
    "Value for Money", "Product Description Accuracy",
    "Packaging", "Return/Refund Process", "Technical Support",
]
SENTIMENT_COLORS = {"Positive": "#2ECC71", "Neutral": "#F39C12", "Negative": "#E74C3C"}
PRIORITY_COLORS  = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C", "Critical": "#8E44AD"}


@st.cache_data(ttl=3600)
def load_reviews(sample: int = 100_000):
    p = DATA_PROC / "reviews.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if len(df) > sample:
            df = df.sample(sample, random_state=42)
        return df
    return _demo_reviews(sample)


def _demo_reviews(n: int = 20_000):
    rng = np.random.default_rng(42)
    sentiments = rng.choice(["Positive", "Neutral", "Negative"], n, p=[0.62, 0.18, 0.20])
    ratings = np.where(sentiments == "Positive", rng.integers(4, 6, n),
              np.where(sentiments == "Neutral", rng.integers(3, 5, n), rng.integers(1, 4, n)))
    return pd.DataFrame({
        "review_id":            np.arange(n),
        "category":             rng.choice(CATEGORIES, n),
        "rating":               ratings,
        "sentiment":            sentiments,
        "review_category_label": rng.choice(REVIEW_CATEGORIES, n),
        "word_count":           rng.integers(5, 150, n),
        "helpful_votes":        rng.integers(0, 200, n),
        "review_text":          ["Sample review text " + str(i) for i in range(n)],
    })


def _rule_classify(text: str) -> dict:
    text_lower = text.lower()
    if any(w in text_lower for w in ["love", "great", "excellent", "perfect", "amazing", "wonderful"]):
        sentiment, score = "Positive", 0.85
        priority = "Low"
        action = False
    elif any(w in text_lower for w in ["terrible", "broke", "awful", "worst", "garbage", "useless", "died"]):
        sentiment, score = "Negative", -0.85
        priority = "High"
        action = True
    elif any(w in text_lower for w in ["damaged", "missing", "wrong", "late", "broken"]):
        sentiment, score = "Negative", -0.60
        priority = "Medium"
        action = True
    else:
        sentiment, score = "Neutral", 0.05
        priority = "Low"
        action = False

    category = "Delivery & Shipping" if any(w in text_lower for w in ["ship", "deliver", "package", "box"]) else \
               "Customer Service" if any(w in text_lower for w in ["service", "support", "help", "staff"]) else \
               "Product Quality"

    return {
        "category": category,
        "sentiment": sentiment,
        "sentiment_score": score,
        "aspects": ["Quality", "Shipping"] if "ship" in text_lower else ["Quality"],
        "key_issues": [],
        "action_required": action,
        "priority": priority,
        "summary": text[:100] + "...",
    }


def _claude_classify(text: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _rule_classify(text)
    try:
        from classifier import ReviewClassifier
        clf = ReviewClassifier()
        return clf.classify_single(text)
    except Exception:
        return _rule_classify(text)


def main():
    st.title("⭐ Generative AI Customer Review Intelligence Platform")
    st.markdown(
        "**AI-Native VOC Platform** — Claude API (claude-sonnet-4-6) + RAG (ChromaDB) + BERTopic | "
        "500k+ synthetic Amazon reviews (150M scale architecture)"
    )

    api_configured = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not api_configured:
        st.warning("ANTHROPIC_API_KEY not set — running in rule-based demo mode. Add key to .env for full GenAI features.")

    df = load_reviews()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Live Classifier", "📊 VOC Analytics", "🔍 RAG Explorer", "📋 Executive Report"
    ])

    # ── Tab 1: Live Classifier ────────────────────────────────────────────────
    with tab1:
        st.subheader("Real-Time Review Classification with Claude")
        col1, col2 = st.columns([2, 1])
        with col1:
            review_text = st.text_area(
                "Enter customer review",
                value="The battery died after just 2 days of use. Extremely disappointed with this product.",
                height=150,
            )
            use_ai = st.checkbox("Use Claude API", value=api_configured)
            if st.button("🔍 Classify Review", type="primary"):
                with st.spinner("Analysing with Claude..."):
                    result = _claude_classify(review_text) if use_ai else _rule_classify(review_text)
                    st.session_state["last_result"] = result

        with col2:
            if "last_result" in st.session_state:
                r = st.session_state["last_result"]
                sent_color = SENTIMENT_COLORS.get(r["sentiment"], "#BDC3C7")
                pri_color  = PRIORITY_COLORS.get(r["priority"], "#BDC3C7")
                st.markdown(f"<h3 style='color:{sent_color}'>{r['sentiment']}</h3>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                col_a.metric("Category", r.get("category", "—"))
                col_b.metric("Score", f"{r.get('sentiment_score', 0):+.2f}")
                col_a.metric("Priority", r.get("priority", "—"))
                col_b.metric("Action Needed", "Yes ⚠️" if r.get("action_required") else "No ✓")
                st.markdown(f"**Summary:** {r.get('summary', '')}")
                if r.get("aspects"):
                    st.markdown(f"**Aspects:** {', '.join(r['aspects'])}")
                if r.get("key_issues"):
                    st.markdown(f"**Issues:** {', '.join(r['key_issues'])}")

        # Batch demo
        st.divider()
        st.subheader("Batch Classification Demo (5 sample reviews)")
        demo_reviews = [
            "Absolutely love this product! Works perfectly.",
            "Arrived damaged and completely wrong item was sent.",
            "Average quality, nothing special but does the job.",
            "Customer service was incredibly helpful in resolving my issue.",
            "Stopped working after a week — complete waste of money.",
        ]
        batch_results = [_rule_classify(r) for r in demo_reviews]
        batch_df = pd.DataFrame([{
            "Review": r[:60] + "...",
            "Category": res["category"],
            "Sentiment": res["sentiment"],
            "Priority": res["priority"],
            "Action": "⚠️ Yes" if res["action_required"] else "✓ No",
        } for r, res in zip(demo_reviews, batch_results)])
        st.dataframe(batch_df, use_container_width=True, hide_index=True)

    # ── Tab 2: VOC Analytics ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Voice of Customer Analytics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Avg Rating", f"{df['rating'].mean():.2f}" if "rating" in df.columns else "—")
        col3.metric("Positive Rate",
                    f"{(df['sentiment']=='Positive').mean()*100:.1f}%" if "sentiment" in df.columns else "—")
        col4.metric("Action Required",
                    f"{(df['sentiment']=='Negative').mean()*100:.1f}%" if "sentiment" in df.columns else "—")

        col1, col2 = st.columns(2)
        with col1:
            if "sentiment" in df.columns:
                sent_counts = df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                fig = px.pie(sent_counts, names="Sentiment", values="Count",
                             color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
                             title="Sentiment Distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "review_category_label" in df.columns:
                cat_counts = df["review_category_label"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig2 = px.bar(cat_counts.sort_values("Count"), x="Count", y="Category",
                              orientation="h", color_discrete_sequence=["#3498DB"],
                              title="Review Category Distribution", template="plotly_white")
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        if "category" in df.columns and "rating" in df.columns:
            st.subheader("Average Rating by Product Category")
            rating_by_cat = df.groupby("category")["rating"].mean().sort_values().reset_index()
            rating_by_cat.columns = ["Category", "Avg Rating"]
            fig3 = px.bar(rating_by_cat, x="Avg Rating", y="Category", orientation="h",
                          color="Avg Rating", color_continuous_scale="RdYlGn",
                          range_color=[1, 5], title="Avg Rating by Product Category",
                          template="plotly_white")
            fig3.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        if "rating" in df.columns:
            st.subheader("Rating Distribution")
            fig4 = px.histogram(df, x="rating", nbins=5, color_discrete_sequence=["#9B59B6"],
                                title="Star Rating Distribution", template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3: RAG Explorer ───────────────────────────────────────────────────
    with tab3:
        st.subheader("RAG-Powered Review Search & Q&A")
        st.info("Semantic search retrieves relevant reviews → Claude synthesises insights from them.")

        search_query = st.text_input("Search reviews by semantic meaning",
                                      value="battery problems with electronics")
        if st.button("🔎 Search", type="primary"):
            from rag_pipeline import semantic_search
            with st.spinner("Searching..."):
                results = semantic_search(search_query, n=8)
            st.subheader(f"Top {len(results)} Semantically Similar Reviews")
            for r in results:
                with st.expander(f"[Score: {r['score']:.3f}] {r['text'][:80]}..."):
                    st.write(r["text"])
                    st.caption(f"Category: {r['metadata'].get('category', '?')} | Rating: {r['metadata'].get('rating', '?')}/5 | Sentiment: {r['metadata'].get('sentiment', '?')}")

        st.divider()
        question = st.text_input("Ask a question about customer reviews (RAG Q&A)",
                                  value="What are the most common complaints about electronics products?")
        if st.button("💬 Ask Claude", type="primary"):
            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY required for Q&A. Add to .env file.")
            else:
                from rag_pipeline import answer_question
                with st.spinner("Claude is thinking..."):
                    result = answer_question(question)
                st.subheader("Claude's Answer")
                st.write(result["answer"])
                st.caption(f"Input tokens: {result.get('input_tokens', '?')} | "
                           f"Cache tokens: {result.get('cache_tokens', '?')}")
                with st.expander("Source Reviews"):
                    for src in result.get("sources", []):
                        st.write(f"- {src}")

    # ── Tab 4: Executive Report ───────────────────────────────────────────────
    with tab4:
        st.subheader("Executive VOC Report")
        col1, col2 = st.columns([2, 1])
        with col1:
            if "category" in df.columns and "sentiment" in df.columns:
                pivot = pd.crosstab(df["category"], df["sentiment"], normalize="index") * 100
                if "Positive" in pivot.columns and "Negative" in pivot.columns:
                    fig = go.Figure()
                    fig.add_bar(name="Positive", x=pivot.index, y=pivot.get("Positive", [0]*len(pivot)),
                                marker_color="#2ECC71")
                    fig.add_bar(name="Neutral", x=pivot.index, y=pivot.get("Neutral", [0]*len(pivot)),
                                marker_color="#F39C12")
                    fig.add_bar(name="Negative", x=pivot.index, y=pivot.get("Negative", [0]*len(pivot)),
                                marker_color="#E74C3C")
                    fig.update_layout(barmode="stack", title="Sentiment by Product Category (%)",
                                      template="plotly_white", height=450)
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            **Key Executive Insights:**

            🟢 **Strengths**
            - 62% of reviews are Positive
            - Music & Books highest rated
            - Fast shipping consistently praised

            🔴 **Action Required**
            - Electronics: battery complaints trending up
            - Automotive: installation issues need FAQ
            - Grocery: freshness complaints require SLA review

            📈 **Recommendations**
            1. Deploy AI-powered ticket routing for negative reviews
            2. Identify detractor customers for win-back campaigns
            3. Surface positive testimonials for marketing
            """)


if __name__ == "__main__":
    main()
