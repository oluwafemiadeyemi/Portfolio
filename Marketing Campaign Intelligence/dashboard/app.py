"""
Streamlit dashboard: Marketing Campaign Intelligence Platform
Tabs: Segment Explorer | RFM Analysis | Campaign Attribution | Market Basket
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
sys.path.insert(0, str(BASE_DIR / "src"))

st.set_page_config(
    page_title="Marketing Campaign Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Colour palette ──────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":          "#2ECC71",
    "Loyal Customers":    "#3498DB",
    "Potential Loyalist": "#9B59B6",
    "At Risk":            "#E74C3C",
    "Lost":               "#95A5A6",
    "New Customer":       "#F39C12",
    "Promising":          "#1ABC9C",
    "Needs Attention":    "#E67E22",
    "Can't Lose Them":    "#C0392B",
    "Noise / Unclustered": "#BDC3C7",
}

# ─── Data loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_rfm():
    p = DATA_PROC / "rfm_scored.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # Normalise column names from real training pipeline
        if "recency" in df.columns and "recency_days" not in df.columns:
            df = df.rename(columns={"recency": "recency_days"})
        if "clv_estimate" not in df.columns:
            df["clv_estimate"] = (
                df["monetary"] * df["frequency"] * 12 / np.log1p(df["recency_days"].clip(lower=1))
            ).round(2)
        return df
    return _generate_demo_rfm()


@st.cache_data(ttl=3600)
def load_segments():
    p = DATA_PROC / "segment_embeddings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return _generate_demo_segments()


@st.cache_data(ttl=3600)
def load_attribution():
    p = DATA_PROC / "attribution_pct.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return _generate_demo_attribution()


@st.cache_data(ttl=3600)
def load_rules():
    p = DATA_PROC / "association_rules.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # Cast numeric columns that may have been saved as object dtype
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[
            ~df["antecedents_str"].str.contains("ch:", na=False) &
            ~df["consequents_str"].str.contains("ch:", na=False)
        ]
    return _generate_demo_rules()


# ─── Demo data generators ────────────────────────────────────────────────────

def _generate_demo_rfm(n: int = 50_000):
    rng = np.random.default_rng(42)
    segments = ["Champions", "Loyal Customers", "Potential Loyalist", "At Risk",
                "Lost", "New Customer", "Needs Attention", "Promising"]
    weights = [0.10, 0.18, 0.20, 0.15, 0.12, 0.13, 0.07, 0.05]
    segs = rng.choice(segments, size=n, p=weights)
    seg_params = {
        "Champions":          (10, 15, 400),
        "Loyal Customers":    (20, 10, 220),
        "Potential Loyalist": (35, 5, 120),
        "At Risk":            (80, 3, 90),
        "Lost":               (180, 1, 40),
        "New Customer":       (10, 2, 75),
        "Needs Attention":    (60, 4, 100),
        "Promising":          (15, 3, 130),
    }
    rows = []
    for i, s in enumerate(segs):
        r_mu, f_mu, m_mu = seg_params[s]
        r = max(1, int(rng.exponential(r_mu)))
        f = max(1, int(rng.normal(f_mu, 2)))
        m = max(0.0, rng.normal(m_mu, m_mu * 0.35))
        rows.append((i, s, r, f, m))
    df = pd.DataFrame(rows, columns=["customer_id", "segment", "recency_days", "frequency", "monetary"])
    df["clv_estimate"] = (df["monetary"] * df["frequency"] * 12 / np.log1p(df["recency_days"])).round(2)
    return df


def _generate_demo_segments(n: int = 20_000):
    rng = np.random.default_rng(42)
    centers = {"Champions": (2, 3), "Loyal Customers": (-1, 2), "At Risk": (-3, -2),
                "Lost": (-4, -4), "New Customer": (3, -1), "Needs Attention": (0, -2)}
    rows = []
    for seg, (cx, cy) in centers.items():
        sz = int(n * {"Champions": 0.10, "Loyal Customers": 0.20, "At Risk": 0.18,
                      "Lost": 0.15, "New Customer": 0.20, "Needs Attention": 0.17}[seg])
        x = rng.normal(cx, 0.8, sz)
        y = rng.normal(cy, 0.8, sz)
        rows.append(pd.DataFrame({"umap_x": x, "umap_y": y, "segment_name": seg}))
    return pd.concat(rows, ignore_index=True)


def _generate_demo_attribution():
    channels = ["email", "social", "paid_search", "display", "organic", "affiliate", "sms"]
    data = {
        "last_touch":  [8, 22, 35, 12, 15, 5, 3],
        "first_touch": [18, 20, 25, 14, 14, 6, 3],
        "linear":      [14, 21, 29, 13, 14, 6, 3],
        "time_decay":  [11, 21, 32, 12, 14, 6, 4],
        "shapley":     [15, 20, 28, 13, 15, 6, 3],
    }
    return pd.DataFrame(data, index=channels)


def _generate_demo_rules():
    products = ["electronics", "fashion", "home_garden", "sports", "beauty", "automotive"]
    rows = []
    for a in products:
        for b in products:
            if a != b:
                rows.append({
                    "antecedents_str": a,
                    "consequents_str": b,
                    "support": round(np.random.uniform(0.02, 0.15), 3),
                    "confidence": round(np.random.uniform(0.3, 0.8), 3),
                    "lift": round(np.random.uniform(1.1, 3.5), 3),
                })
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


# ─── Main app ────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Marketing Campaign Intelligence Platform")
    st.markdown(
        "**Enterprise customer segmentation, RFM scoring, multi-touch attribution & basket analysis** "
        "— powered by 5M+ synthetic marketing events + HDBSCAN / UMAP / FP-Growth"
    )

    rfm = load_rfm()
    segments = load_segments()
    attribution = load_attribution()
    rules = load_rules()

    # ── Sidebar KPIs ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Portfolio KPIs")
        col1, col2 = st.columns(2)
        col1.metric("Total Customers", f"{len(rfm):,}")
        col2.metric("Total Revenue", f"${rfm['monetary'].sum():,.0f}")
        col1.metric("Avg CLV", f"${rfm['clv_estimate'].mean():,.0f}")
        col2.metric("Champions", f"{(rfm['segment']=='Champions').sum():,}")
        st.divider()
        st.caption("Data: 5M synthetic marketing events + UCI Bank Marketing")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Segment Explorer", "📈 RFM Analysis", "🎯 Attribution", "🛒 Market Basket"
    ])

    # ── Tab 1: Segment Explorer ───────────────────────────────────────────────
    with tab1:
        st.subheader("Customer Segment Landscape (UMAP + HDBSCAN)")
        col1, col2 = st.columns([3, 1])
        with col1:
            sample = segments.sample(min(10_000, len(segments)), random_state=42)
            fig = px.scatter(
                sample, x="umap_x", y="umap_y", color="segment_name",
                color_discrete_map=SEGMENT_COLORS,
                opacity=0.6, template="plotly_white",
                title="UMAP 2D Embedding — Coloured by Segment",
                labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
            )
            fig.update_traces(marker_size=3)
            fig.update_layout(height=550, legend_title="Segment")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            seg_counts = segments["segment_name"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig2 = px.pie(
                seg_counts, names="segment", values="count",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                title="Segment Distribution",
            )
            fig2.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

        # Segment profiles table
        st.subheader("Segment Profiles")
        seg_profile = rfm.groupby("segment").agg(
            Count=("customer_id", "count"),
            Avg_Recency=("recency_days", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Monetary=("monetary", "mean"),
            Avg_CLV=("clv_estimate", "mean"),
        ).round(1).reset_index()
        seg_profile["% of Base"] = (seg_profile["Count"] / len(rfm) * 100).round(1)
        seg_profile = seg_profile.sort_values("Avg_CLV", ascending=False)
        st.dataframe(seg_profile, use_container_width=True, hide_index=True)

    # ── Tab 2: RFM Analysis ───────────────────────────────────────────────────
    with tab2:
        st.subheader("RFM Scoring — 5M Customer Base")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(
                rfm.sample(min(50_000, len(rfm)), random_state=1),
                x="recency_days", nbins=50, color_discrete_sequence=["#3498DB"],
                title="Recency Distribution", template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                rfm.sample(min(50_000, len(rfm)), random_state=1),
                x="frequency", nbins=30, color_discrete_sequence=["#2ECC71"],
                title="Frequency Distribution", template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.histogram(
                rfm.sample(min(50_000, len(rfm)), random_state=1),
                x="monetary", nbins=50, color_discrete_sequence=["#E74C3C"],
                title="Monetary Distribution", template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue by Segment")
        rev_seg = rfm.groupby("segment")["monetary"].sum().reset_index()
        rev_seg.columns = ["segment", "total_revenue"]
        rev_seg = rev_seg.sort_values("total_revenue", ascending=True)
        fig = px.bar(
            rev_seg, x="total_revenue", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Total Revenue by Segment", template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("CLV vs Recency vs Frequency (Bubble Chart)")
        bubble = rfm.sample(min(5_000, len(rfm)), random_state=2)
        fig = px.scatter(
            bubble, x="recency_days", y="frequency",
            size="clv_estimate", color="segment",
            color_discrete_map=SEGMENT_COLORS,
            size_max=25, opacity=0.7, template="plotly_white",
            title="CLV Bubble Chart — Size = CLV",
            labels={"recency_days": "Recency (days)", "frequency": "Purchase Frequency"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Attribution ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Multi-Touch Attribution — Channel Credit Allocation")
        st.markdown(
            "Five attribution models compared: Last Touch, First Touch, Linear, Time Decay, and **Shapley Value** (data-driven)."
        )
        model = st.selectbox(
            "Select Attribution Model",
            ["shapley", "linear", "time_decay", "last_touch", "first_touch"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        attr_values = attribution[model].sort_values(ascending=False)
        fig = px.bar(
            x=attr_values.index, y=attr_values.values,
            color=attr_values.index,
            labels={"x": "Channel", "y": "Attribution %"},
            title=f"{model.replace('_', ' ').title()} Attribution by Channel",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison (All Channels)")
        fig2 = px.bar(
            attribution.reset_index().melt(id_vars="index", var_name="Model", value_name="Attribution %"),
            x="index", y="Attribution %", color="Model", barmode="group",
            labels={"index": "Channel"},
            title="Attribution Comparison Across All Models",
            template="plotly_white",
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Shapley vs Last-Touch Difference")
        diff = (attribution["shapley"] - attribution["last_touch"]).reset_index()
        diff.columns = ["channel", "delta"]
        diff["direction"] = diff["delta"].apply(lambda x: "Under-credited" if x > 0 else "Over-credited")
        fig3 = px.bar(
            diff, x="channel", y="delta", color="direction",
            color_discrete_map={"Under-credited": "#2ECC71", "Over-credited": "#E74C3C"},
            title="Shapley vs Last-Touch: Channel Credit Delta",
            template="plotly_white",
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: Market Basket ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Market Basket Analysis — FP-Growth Association Rules")
        min_lift = st.slider("Minimum Lift", 1.0, 4.0, 1.2, 0.1)
        top_n = st.slider("Top N Rules", 5, 50, 15)

        filtered = rules[rules["lift"] >= min_lift].nlargest(top_n, "lift")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(
                filtered,
                x="support", y="confidence", size="lift",
                color="lift", color_continuous_scale="Viridis",
                hover_data=["antecedents_str", "consequents_str"],
                title="Support vs Confidence (Bubble = Lift)",
                template="plotly_white", size_max=30,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(
                filtered[["antecedents_str", "consequents_str", "lift", "confidence", "support"]]
                .rename(columns={"antecedents_str": "If Buys", "consequents_str": "Also Buys"}),
                use_container_width=True,
                hide_index=True,
                height=430,
            )

        # Product affinity heatmap
        products = sorted(set(
            rules["antecedents_str"].tolist() + rules["consequents_str"].tolist()
        ))[:12]
        heatmap_data = pd.DataFrame(0.0, index=products, columns=products)
        for _, row in rules.iterrows():
            a, b = row["antecedents_str"], row["consequents_str"]
            if a in products and b in products:
                heatmap_data.loc[a, b] = row["lift"]

        fig4 = px.imshow(
            heatmap_data, color_continuous_scale="RdYlGn",
            title="Product Affinity Heatmap (Lift Values)",
            template="plotly_white",
        )
        fig4.update_layout(height=450)
        st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()
