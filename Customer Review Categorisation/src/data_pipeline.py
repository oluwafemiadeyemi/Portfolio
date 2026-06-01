"""
Data pipeline: generates 500k realistic synthetic Amazon-style reviews
across 12 product categories. Designed to simulate the 150M Amazon dataset.
Supports loading real Amazon Reviews CSV if available in data/raw/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

CATEGORIES = [
    "Electronics", "Fashion", "Home & Garden", "Sports", "Beauty",
    "Books", "Toys", "Grocery", "Automotive", "Health", "Movies", "Music"
]

SENTIMENT_PHRASES = {
    "positive": [
        "Absolutely love this product!", "Exceeded my expectations.",
        "Great quality for the price.", "Fast shipping and well packaged.",
        "Would definitely recommend to friends.", "Works perfectly.",
        "Best purchase I've made this year.", "Exactly as described.",
    ],
    "negative": [
        "Stopped working after a week.", "Terrible quality.",
        "Not worth the money.", "Arrived broken.",
        "Completely different from the description.", "Very disappointed.",
        "Customer service was unhelpful.", "Do not buy this.",
    ],
    "neutral": [
        "It's okay for the price.", "Does what it says.",
        "Nothing special but works fine.", "Average quality.",
        "Shipping took longer than expected.", "Product is as described.",
    ],
}

CATEGORIES_ISSUES = {
    "Electronics":   ["battery life", "screen quality", "connectivity", "software bugs", "build quality"],
    "Fashion":       ["sizing", "material quality", "color accuracy", "stitching", "fit"],
    "Home & Garden": ["assembly", "durability", "design", "functionality", "materials"],
    "Sports":        ["performance", "comfort", "durability", "size", "value"],
    "Beauty":        ["skin reaction", "scent", "effectiveness", "packaging", "texture"],
    "Books":         ["content quality", "binding", "printing", "delivery", "price"],
    "Toys":          ["safety", "durability", "age appropriateness", "parts missing", "instructions"],
    "Grocery":       ["freshness", "taste", "packaging", "quantity", "price"],
    "Automotive":    ["fit", "quality", "ease of installation", "performance", "durability"],
    "Health":        ["effectiveness", "side effects", "dosage instructions", "packaging", "quality"],
    "Movies":        ["video quality", "audio", "subtitles", "extra features", "packaging"],
    "Music":         ["sound quality", "album completeness", "packaging", "value", "format"],
}

REVIEW_CATEGORIES = [
    "Product Quality", "Delivery & Shipping", "Customer Service",
    "Value for Money", "Product Description Accuracy",
    "Packaging", "Return/Refund Process", "Technical Support",
]

ASPECT_LABELS = [
    "Quality", "Price", "Shipping", "Service", "Durability",
    "Functionality", "Design", "Packaging",
]


def generate_review_text(category: str, rating: int, rng_local) -> str:
    """Generate realistic review text based on category and rating."""
    if rating >= 4:
        sentiment_key = "positive"
    elif rating == 3:
        sentiment_key = "neutral"
    else:
        sentiment_key = "negative"

    phrases = SENTIMENT_PHRASES[sentiment_key]
    issues = CATEGORIES_ISSUES.get(category, ["quality", "value", "delivery"])

    # Build multi-sentence review
    n_sentences = rng_local.integers(1, 5)
    sentences = [rng_local.choice(phrases)]
    for _ in range(n_sentences - 1):
        issue = rng_local.choice(issues)
        if sentiment_key == "positive":
            sentences.append(f"The {issue} is excellent.")
        elif sentiment_key == "negative":
            sentences.append(f"The {issue} is disappointing.")
        else:
            sentences.append(f"The {issue} is acceptable.")

    return " ".join(sentences)


def generate_synthetic_reviews(n: int = 500_000) -> pd.DataFrame:
    """Generate n synthetic Amazon-style customer reviews."""
    print(f"Generating {n:,} synthetic reviews ...")
    chunk_size = 50_000
    frames = []

    for start in tqdm(range(0, n, chunk_size), desc="Generating"):
        end = min(start + chunk_size, n)
        sz = end - start
        rng_local = np.random.default_rng(start)

        categories = rng_local.choice(CATEGORIES, sz)
        ratings    = rng_local.choice([1, 2, 3, 4, 5], sz,
                                       p=[0.08, 0.07, 0.12, 0.28, 0.45])
        verified   = rng_local.choice([True, False], sz, p=[0.85, 0.15])
        helpful    = rng_local.integers(0, 200, sz)

        texts = [generate_review_text(cat, rat, rng_local)
                 for cat, rat in zip(categories, ratings)]

        # Ground truth labels for evaluation
        review_cats = rng_local.choice(REVIEW_CATEGORIES, sz)
        sentiment = np.where(ratings >= 4, "Positive",
                    np.where(ratings == 3, "Neutral", "Negative"))

        chunk = pd.DataFrame({
            "review_id":      np.arange(start, end),
            "category":       categories,
            "rating":         ratings,
            "review_text":    texts,
            "verified_purchase": verified,
            "helpful_votes":  helpful,
            "sentiment":      sentiment,
            "review_category_label": review_cats,
            "word_count":     [len(t.split()) for t in texts],
        })
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"Generated {len(df):,} reviews | Avg rating: {df['rating'].mean():.2f}")
    return df


def load_amazon_reviews() -> pd.DataFrame:
    """Load real Amazon Reviews CSV if available in data/raw/."""
    for name in ["amazon_reviews.csv", "reviews.csv", "all_amazon_review.csv"]:
        p = DATA_RAW / name
        if p.exists():
            print(f"Loading Amazon Reviews: {p}")
            df = pd.read_csv(p, low_memory=False)
            return df
    return pd.DataFrame()


def prepare_all(n: int = 500_000, sample_for_rag: int = 50_000) -> dict:
    """Full pipeline: load or generate, split into RAG sample and full set."""
    real = load_amazon_reviews()
    if len(real) > 0:
        df = real
    else:
        df = generate_synthetic_reviews(n)

    df.to_parquet(DATA_PROC / "reviews.parquet", index=False)
    print(f"Saved {len(df):,} reviews")

    # Sub-sample for ChromaDB RAG pipeline
    rag_sample = df.sample(min(sample_for_rag, len(df)), random_state=42)
    rag_sample.to_parquet(DATA_PROC / "rag_sample.parquet", index=False)
    print(f"RAG sample: {len(rag_sample):,} reviews saved")

    return {"df": df, "rag_sample": rag_sample}


if __name__ == "__main__":
    prepare_all()
