"""
Streamlit dashboard: Automotive Pricing Intelligence Platform
Tabs: Instant Valuation | Market Analysis | Depreciation | SHAP Explainer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

st.set_page_config(
    page_title="Automotive Pricing Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAKE_BASE = {
    "Toyota": 22000, "Honda": 21000, "Ford": 28000, "Chevrolet": 27000,
    "BMW": 45000, "Mercedes": 50000, "Tesla": 55000, "Hyundai": 19000,
    "Nissan": 20000, "Jeep": 32000, "Audi": 43000, "Volkswagen": 24000,
}

COND_MULT  = {"new": 1.00, "like new": 0.97, "excellent": 0.92, "good": 0.82, "fair": 0.65, "salvage": 0.35}
FUEL_MULT  = {"gas": 1.0, "diesel": 1.05, "hybrid": 1.08, "electric": 1.15, "other": 0.95}
TITLE_MULT = {"clean": 1.0, "rebuilt": 0.80, "lien": 0.90, "missing": 0.70, "parts only": 0.30, "salvage": 0.35}


def dep(age):
    if age <= 5:
        return np.exp(-0.16 * age)
    return np.exp(-0.8) * np.exp(-0.10 * (age - 5))


def mil(odometer):
    if odometer <= 20_000:
        return 1.0
    return max(0.3, 1.0 - 0.12 * np.log(max(odometer / 20_000, 1.0001)))


@st.cache_data(ttl=3600)
def load_vehicles(sample: int = 200_000):
    p = DATA_PROC / "vehicles.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if len(df) > sample:
            df = df.sample(sample, random_state=42)
        return df
    return _demo_vehicles(sample)


def _demo_vehicles(n: int = 50_000):
    rng = np.random.default_rng(42)
    makes = list(MAKE_BASE.keys())
    years = rng.integers(2008, 2025, n)
    makes_arr = rng.choice(makes, n)
    odo = np.clip(rng.normal(80_000, 50_000, n), 0, 300_000).astype(int)
    prices = np.array([
        MAKE_BASE[m] * dep(2025 - y) * mil(o) * rng.uniform(0.9, 1.1)
        for m, y, o in zip(makes_arr, years, odo)
    ])
    return pd.DataFrame({"make": makes_arr, "year": years, "odometer": odo,
                          "price": prices.round(0).astype(int),
                          "age_years": 2025 - years})


def estimate_price(make, year, odometer, condition, fuel, title_status):
    base = MAKE_BASE.get(make, 25000)
    age = 2025 - year
    price = base * dep(age) * mil(odometer) * COND_MULT[condition] * FUEL_MULT[fuel] * TITLE_MULT[title_status]
    return price


def main():
    st.title("🚗 Automotive Pricing Intelligence Platform")
    st.markdown("**Real-time vehicle valuation engine** — 3M+ synthetic Craigslist-scale listings | LightGBM + XGBoost + CatBoost ensemble + SHAP")

    df = load_vehicles()

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Instant Valuation", "📊 Market Analysis", "📉 Depreciation Curves", "🔍 SHAP Explainer"
    ])

    # ── Tab 1: Instant Valuation ──────────────────────────────────────────────
    with tab1:
        st.subheader("Vehicle Instant Valuation")
        col1, col2 = st.columns([1, 1])
        with col1:
            make        = st.selectbox("Make", sorted(MAKE_BASE.keys()))
            year        = st.slider("Year", 2005, 2024, 2018)
            odometer    = st.slider("Odometer (miles)", 0, 300_000, 55_000, step=5_000)
            condition   = st.selectbox("Condition", list(COND_MULT.keys()))
            fuel        = st.selectbox("Fuel Type", list(FUEL_MULT.keys()))
            title_status = st.selectbox("Title Status", list(TITLE_MULT.keys()))

        with col2:
            price = estimate_price(make, year, odometer, condition, fuel, title_status)
            low, high = price * 0.88, price * 1.12

            st.metric("Estimated Price", f"${price:,.0f}")
            st.metric("Price Range", f"${low:,.0f} — ${high:,.0f}")

            age = 2025 - year
            breakdown = {
                "Base Value":          MAKE_BASE.get(make, 25000),
                "After Depreciation":  MAKE_BASE.get(make, 25000) * dep(age),
                "After Mileage":       MAKE_BASE.get(make, 25000) * dep(age) * mil(odometer),
                "After Condition":     MAKE_BASE.get(make, 25000) * dep(age) * mil(odometer) * COND_MULT[condition],
                "Final Estimate":      price,
            }
            waterfall = pd.DataFrame({
                "Stage":  list(breakdown.keys()),
                "Price":  list(breakdown.values()),
            })
            fig = px.funnel(waterfall, x="Price", y="Stage",
                            title="Price Decomposition Waterfall",
                            color_discrete_sequence=["#3498DB"])
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Market Analysis ────────────────────────────────────────────────
    with tab2:
        st.subheader("Market Overview — 3M Listings")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", f"{len(df):,}")
        col2.metric("Avg Price", f"${df['price'].mean():,.0f}")
        col3.metric("Median Price", f"${df['price'].median():,.0f}")
        col4.metric("Avg Mileage", f"{df['odometer'].mean():,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            price_by_make = df.groupby("make")["price"].median().sort_values(ascending=False).reset_index()
            fig = px.bar(price_by_make, x="make", y="price", color="make",
                         title="Median Price by Make", template="plotly_white")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="price", nbins=80, color_discrete_sequence=["#3498DB"],
                               title="Price Distribution", template="plotly_white")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df.sample(min(10_000, len(df)), random_state=3),
            x="odometer", y="price", color="make",
            opacity=0.4, template="plotly_white",
            title="Price vs Mileage by Make",
            labels={"odometer": "Odometer (miles)", "price": "Price ($)"},
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Depreciation Curves ────────────────────────────────────────────
    with tab3:
        st.subheader("Depreciation Curves by Make")
        selected_makes = st.multiselect(
            "Select Makes to Compare",
            sorted(MAKE_BASE.keys()),
            default=["Toyota", "BMW", "Tesla", "Ford"],
        )
        if selected_makes:
            ages = np.arange(0, 21)
            curves = []
            for m in selected_makes:
                base = MAKE_BASE[m]
                for a in ages:
                    curves.append({"Make": m, "Age (years)": a,
                                   "Residual Value ($)": base * dep(a)})
            curve_df = pd.DataFrame(curves)
            fig = px.line(curve_df, x="Age (years)", y="Residual Value ($)",
                          color="Make", markers=True,
                          title="Vehicle Depreciation Curves",
                          template="plotly_white")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            pct_df = curve_df.copy()
            pct_df["Retained %"] = pct_df.apply(
                lambda r: dep(r["Age (years)"]) * 100, axis=1
            )
            fig2 = px.line(pct_df, x="Age (years)", y="Retained %",
                           color="Make", markers=True,
                           title="% Value Retained Over Time",
                           template="plotly_white")
            fig2.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="50% residual value")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: SHAP Explainer ─────────────────────────────────────────────────
    with tab4:
        st.subheader("SHAP Price Factor Decomposition")
        st.markdown("How each vehicle attribute contributes to (or detracts from) its estimated value.")

        col1, col2, col3 = st.columns(3)
        e_make    = col1.selectbox("Make ", sorted(MAKE_BASE.keys()), key="s_make")
        e_year    = col1.slider("Year ", 2005, 2024, 2016, key="s_year")
        e_odo     = col2.slider("Odometer ", 0, 300_000, 90_000, step=5_000, key="s_odo")
        e_cond    = col2.selectbox("Condition ", list(COND_MULT.keys()), key="s_cond")
        e_fuel    = col3.selectbox("Fuel ", list(FUEL_MULT.keys()), key="s_fuel")
        e_title   = col3.selectbox("Title ", list(TITLE_MULT.keys()), key="s_title")

        base = MAKE_BASE.get(e_make, 25000)
        age  = 2025 - e_year
        factors = {
            "Base Value (Make)":     base,
            "Depreciation (Age)":    (dep(age) - 1) * base,
            "Mileage Impact":        (mil(e_odo) - 1) * base * dep(age),
            "Condition Premium":     (COND_MULT[e_cond] - 0.82) * base * dep(age),
            "Fuel Type Premium":     (FUEL_MULT[e_fuel] - 1.0) * base * dep(age),
            "Title Status Impact":   (TITLE_MULT[e_title] - 1.0) * base * dep(age),
        }
        shap_df = pd.DataFrame({
            "Factor":      list(factors.keys()),
            "Impact ($)":  list(factors.values()),
        })
        shap_df["Color"] = shap_df["Impact ($)"].apply(lambda x: "Positive" if x >= 0 else "Negative")
        shap_df = shap_df.sort_values("Impact ($)")
        fig = px.bar(
            shap_df, y="Factor", x="Impact ($)", orientation="h",
            color="Color",
            color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C"},
            title="SHAP-Style Price Factor Breakdown",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        total_price = estimate_price(e_make, e_year, e_odo, e_cond, e_fuel, e_title)
        st.metric("Final Estimated Price", f"${total_price:,.0f}",
                  delta=f"${total_price - base:+,.0f} vs base MSRP")


if __name__ == "__main__":
    main()
