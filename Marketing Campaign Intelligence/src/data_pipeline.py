"""
Data pipeline: loads UCI Bank Marketing + Customer Personality datasets,
generates 5M-row synthetic marketing events for big-data processing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Synthetic Big-Data Generator (5M marketing events)
# ──────────────────────────────────────────────

RNG = np.random.default_rng(42)

CHANNELS = ["email", "social", "paid_search", "display", "organic", "affiliate", "sms"]
CAMPAIGN_TYPES = ["awareness", "consideration", "conversion", "retention", "reactivation"]
PRODUCTS = [
    "electronics", "fashion", "home_garden", "sports", "beauty",
    "automotive", "books", "toys", "grocery", "travel",
]
SEGMENTS_TRUE = ["champion", "loyal", "potential_loyalist", "at_risk", "lost", "new_customer"]


def generate_synthetic_events(n: int = 5_000_000) -> pd.DataFrame:
    """Generate n synthetic marketing interaction events with realistic patterns."""
    print(f"Generating {n:,} synthetic marketing events ...")
    chunk_size = 500_000
    frames = []

    segment_labels = RNG.choice(SEGMENTS_TRUE, size=n, p=[0.10, 0.20, 0.25, 0.20, 0.10, 0.15])

    # Per-segment behaviour parameters
    seg_params = {
        "champion":           dict(freq_mu=18, freq_sig=4,  recency_mu=7,   spend_mu=450),
        "loyal":              dict(freq_mu=10, freq_sig=3,  recency_mu=20,  spend_mu=220),
        "potential_loyalist": dict(freq_mu=5,  freq_sig=2,  recency_mu=40,  spend_mu=120),
        "at_risk":            dict(freq_mu=3,  freq_sig=2,  recency_mu=80,  spend_mu=90),
        "lost":               dict(freq_mu=1,  freq_sig=1,  recency_mu=180, spend_mu=40),
        "new_customer":       dict(freq_mu=2,  freq_sig=1,  recency_mu=10,  spend_mu=75),
    }

    for start in tqdm(range(0, n, chunk_size), desc="Generating chunks"):
        end = min(start + chunk_size, n)
        segs = segment_labels[start:end]
        sz = end - start

        freq = np.array([
            max(1, int(RNG.normal(seg_params[s]["freq_mu"], seg_params[s]["freq_sig"])))
            for s in segs
        ])
        recency = np.array([
            max(1, int(RNG.exponential(seg_params[s]["recency_mu"])))
            for s in segs
        ])
        spend = np.array([
            max(0.0, RNG.normal(seg_params[s]["spend_mu"], seg_params[s]["spend_mu"] * 0.4))
            for s in segs
        ])

        channel_weights = {"champion": [0.3, 0.2, 0.15, 0.1, 0.15, 0.05, 0.05],
                           "loyal":    [0.35, 0.15, 0.1, 0.1, 0.15, 0.1, 0.05]}
        channels = RNG.choice(CHANNELS, size=sz)
        campaign = RNG.choice(CAMPAIGN_TYPES, size=sz)
        product = RNG.choice(PRODUCTS, size=sz)

        age = np.clip(RNG.normal(38, 12, sz).astype(int), 18, 75)
        income = np.clip(RNG.normal(65_000, 28_000, sz), 15_000, 300_000)
        num_children = RNG.choice([0, 1, 2, 3], size=sz, p=[0.45, 0.30, 0.18, 0.07])
        education = RNG.choice(
            ["high_school", "some_college", "bachelors", "masters", "phd"],
            size=sz, p=[0.15, 0.20, 0.35, 0.22, 0.08],
        )
        marital = RNG.choice(["single", "married", "divorced", "widowed"], size=sz,
                             p=[0.30, 0.50, 0.15, 0.05])

        clicks = np.clip((freq * RNG.uniform(0.2, 2.5, sz)).astype(int), 0, 50)
        impressions = np.clip(clicks * RNG.integers(5, 30, sz), 1, 2000)
        conversions = (RNG.uniform(size=sz) < np.where(segs == "champion", 0.15, 0.05)).astype(int)
        open_rate = np.clip(
            np.where(segs == "champion", RNG.uniform(0.35, 0.65, sz),
                     RNG.uniform(0.05, 0.35, sz)), 0, 1,
        )

        chunk = pd.DataFrame({
            "customer_id":    np.arange(start, end),
            "segment_true":   segs,
            "age":            age,
            "income":         income.round(2),
            "num_children":   num_children,
            "education":      education,
            "marital_status": marital,
            "recency_days":   recency,
            "frequency":      freq,
            "monetary":       spend.round(2),
            "channel":        channels,
            "campaign_type":  campaign,
            "product_category": product,
            "clicks":         clicks,
            "impressions":    impressions,
            "conversions":    conversions,
            "email_open_rate": open_rate.round(4),
            "customer_since_years": np.clip(RNG.exponential(3, sz), 0, 15).round(1),
            "web_visits_30d":  np.clip(RNG.poisson(5, sz), 0, 60),
            "app_sessions_30d": np.clip(RNG.poisson(3, sz), 0, 40),
            "support_tickets": np.clip(RNG.poisson(0.5, sz), 0, 10),
            "nps_score":       np.clip(RNG.normal(7, 2.5, sz).round(1), 0, 10),
        })
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"Generated {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def load_bank_marketing() -> pd.DataFrame:
    """Load UCI Bank Marketing dataset (45k rows, direct download)."""
    path = DATA_RAW / "bank_marketing.csv"
    if path.exists():
        return pd.read_csv(path, sep=";")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    import requests, zipfile, io
    print("Downloading UCI Bank Marketing dataset ...")
    r = requests.get(url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if "bank-additional-full.csv" in name:
            with z.open(name) as f:
                df = pd.read_csv(f, sep=";")
            df.to_csv(path, index=False)
            print(f"Saved {len(df):,} rows to {path}")
            return df
    raise FileNotFoundError("bank-additional-full.csv not found in zip")


def load_customer_personality() -> pd.DataFrame:
    """Load Customer Personality Analysis dataset (Kaggle, 2.2k rows)."""
    path = DATA_RAW / "customer_personality.csv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_path.exists():
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "imakash3011/customer-personality-analysis",
             "-p", str(DATA_RAW), "--unzip"],
            check=True,
        )
        for f in DATA_RAW.glob("*.csv"):
            if "marketing" in f.name.lower() or "personality" in f.name.lower():
                df = pd.read_csv(f, sep="\t")
                df.to_csv(path, sep="\t", index=False)
                return df
    print("Kaggle credentials not found — skipping Customer Personality dataset.")
    return pd.DataFrame()


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalised RFM features from the synthetic events dataframe."""
    rfm = df[["customer_id", "recency_days", "frequency", "monetary"]].copy()
    rfm = rfm.rename(columns={"recency_days": "R", "frequency": "F", "monetary": "M"})
    # Invert recency so higher = better
    rfm["R_inv"] = rfm["R"].max() - rfm["R"]
    scaler = StandardScaler()
    rfm[["R_norm", "F_norm", "M_norm"]] = scaler.fit_transform(rfm[["R_inv", "F", "M"]])
    rfm["rfm_score"] = rfm[["R_norm", "F_norm", "M_norm"]].mean(axis=1)
    return rfm


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric feature matrix used for clustering."""
    cat_cols = ["channel", "campaign_type", "product_category", "education", "marital_status"]
    le = LabelEncoder()
    encoded = {}
    for col in cat_cols:
        encoded[col + "_enc"] = le.fit_transform(df[col].astype(str))

    numeric = df[[
        "age", "income", "num_children",
        "recency_days", "frequency", "monetary",
        "clicks", "impressions", "conversions",
        "email_open_rate", "customer_since_years",
        "web_visits_30d", "app_sessions_30d",
        "support_tickets", "nps_score",
    ]].copy()

    for k, v in encoded.items():
        numeric[k] = v

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(numeric),
        columns=numeric.columns,
        index=numeric.index,
    )
    return scaled, scaler


def prepare_all(n_synthetic: int = 5_000_000, sample_for_clustering: int = 200_000) -> dict:
    """
    Full pipeline: generate big dataset, load real datasets, engineer features.
    Returns dict with keys: synthetic_df, rfm, feature_matrix, sample_features.
    """
    synthetic_df = generate_synthetic_events(n_synthetic)
    synthetic_df.to_parquet(DATA_PROC / "synthetic_events.parquet", index=False)
    print(f"Saved synthetic events to {DATA_PROC / 'synthetic_events.parquet'}")

    rfm = build_rfm_features(synthetic_df)
    rfm.to_parquet(DATA_PROC / "rfm_features.parquet", index=False)

    feature_matrix, scaler = build_feature_matrix(synthetic_df)
    feature_matrix.to_parquet(DATA_PROC / "feature_matrix.parquet", index=False)

    # Sub-sample for clustering (UMAP+HDBSCAN don't need all 5M rows)
    sample_idx = RNG.choice(len(feature_matrix), size=min(sample_for_clustering, len(feature_matrix)), replace=False)
    sample_features = feature_matrix.iloc[sample_idx].reset_index(drop=True)
    sample_features.to_parquet(DATA_PROC / "sample_features.parquet", index=False)
    print(f"Sample of {len(sample_features):,} rows saved for clustering.")

    # Load supplementary real datasets
    try:
        bank_df = load_bank_marketing()
        bank_df.to_csv(DATA_PROC / "bank_marketing_clean.csv", index=False)
    except Exception as e:
        print(f"Bank marketing load skipped: {e}")
        bank_df = pd.DataFrame()

    return {
        "synthetic_df": synthetic_df,
        "rfm": rfm,
        "feature_matrix": feature_matrix,
        "sample_features": sample_features,
        "bank_df": bank_df,
    }


if __name__ == "__main__":
    result = prepare_all()
    print("\nData pipeline complete.")
    print(f"  Synthetic events : {len(result['synthetic_df']):,} rows")
    print(f"  RFM features     : {len(result['rfm']):,} rows")
    print(f"  Cluster sample   : {len(result['sample_features']):,} rows")
