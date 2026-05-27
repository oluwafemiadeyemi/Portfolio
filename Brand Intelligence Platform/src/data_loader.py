"""
Data loading utilities for the Enterprise Brand Intelligence Platform.
Handles Yelp JSON, Airbnb CSV, and synthetic data generation.
"""

import logging
import random
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic review templates
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "The {aspect} was absolutely wonderful. I was impressed by how {adj} everything was.",
    "Amazing {aspect}! The {staff_adj} staff made our stay unforgettable.",
    "Exceptional {aspect} — truly {adj}. Would highly recommend this place.",
    "We loved the {aspect}. Everything was {adj} and the service was top-notch.",
    "Fantastic experience overall. The {aspect} exceeded all expectations — very {adj}.",
    "One of the best places I have stayed. {aspect} was {adj} and staff were incredibly helpful.",
    "Highly recommend! The {aspect} was spotless and the location was perfect.",
    "Great value for money. The rooms were {adj} and the {aspect} was well-maintained.",
    "The {aspect} was impeccable. Staff were {staff_adj} and always available to help.",
    "Loved every moment. The {aspect} was {adj}, food was delicious, and pricing was fair.",
]

_NEGATIVE_TEMPLATES = [
    "The {aspect} was a complete disappointment. Nothing was {adj} as advertised.",
    "Terrible experience. The {aspect} was filthy and the staff were {staff_adj}.",
    "Would not recommend. The {aspect} left much to be desired — very {adj}.",
    "Awful stay. The {aspect} was broken, the rooms were dirty, and management was {staff_adj}.",
    "Overpriced for what you get. The {aspect} was {adj} and nothing worked properly.",
    "The worst hotel I have stayed in. The {aspect} was unacceptable and staff were {staff_adj}.",
    "Very disappointed. The {aspect} was {adj} and there was no hot water.",
    "Do not stay here. The {aspect} was dirty and the location was unsafe.",
    "Horrible service. The {aspect} was {adj} and the staff were completely unhelpful.",
    "Avoid at all costs. The {aspect} was broken, WiFi was non-existent, parking was {adj}.",
]

_NEUTRAL_TEMPLATES = [
    "The {aspect} was average. Nothing special but nothing terrible either.",
    "An okay experience overall. The {aspect} was standard for the price.",
    "Mixed feelings about this place. The {aspect} was decent but location was inconvenient.",
    "The {aspect} met basic expectations. Service was neither great nor poor.",
    "Fairly standard {aspect}. Some things worked well, others could be improved.",
    "Neither impressed nor disappointed. The {aspect} was exactly what you would expect.",
    "It was fine. The {aspect} was clean enough and the staff were polite.",
    "Average stay. The {aspect} was acceptable though the price felt slightly high.",
    "Middle of the road experience. The {aspect} was okay, location was manageable.",
    "Decent for a budget option. The {aspect} worked and service was adequate.",
]

_ASPECTS = ["cleanliness", "staff service", "value for money", "location", "amenities", "food quality", "room comfort"]
_POS_ADJ = ["spotless", "excellent", "outstanding", "superb", "immaculate", "perfect", "exceptional", "brilliant"]
_NEG_ADJ = ["dirty", "terrible", "awful", "disgusting", "broken", "unacceptable", "horrible", "dreadful"]
_NEU_ADJ = ["average", "standard", "acceptable", "moderate", "typical", "basic", "ordinary", "adequate"]
_POS_STAFF = ["friendly", "attentive", "professional", "courteous", "helpful", "warm", "responsive"]
_NEG_STAFF = ["rude", "unhelpful", "dismissive", "incompetent", "unfriendly", "negligent", "indifferent"]

_BUSINESS_NAMES = [
    "Grand Hyatt Downtown", "Marriott City Center", "Hilton Garden Inn", "Holiday Inn Express",
    "The Ritz-Carlton", "Four Seasons Hotel", "Westin Resort & Spa", "Sheraton Grand",
    "InterContinental Hotel", "Best Western Plus", "Comfort Inn & Suites", "Hampton Inn",
    "La Quinta Inn", "Courtyard by Marriott", "Embassy Suites", "DoubleTree by Hilton",
    "The Waldorf Astoria", "Sofitel Luxury Hotel", "Radisson Blu", "Crowne Plaza",
]

_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "Indianapolis", "San Francisco", "Seattle",
    "Denver", "Nashville", "Oklahoma City", "El Paso", "Washington DC", "Boston",
]


def generate_synthetic_reviews(n: int = 2000) -> pd.DataFrame:
    """Generate realistic hotel/restaurant reviews with ratings."""
    logger.info("Generating %d synthetic reviews", n)
    random.seed(42)
    np.random.seed(42)

    records = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days

    for i in range(n):
        sentiment_roll = random.random()
        if sentiment_roll < 0.55:
            # positive review
            rating = random.choices([4, 5], weights=[0.35, 0.65])[0]
            template = random.choice(_POSITIVE_TEMPLATES)
            adj = random.choice(_POS_ADJ)
            staff_adj = random.choice(_POS_STAFF)
        elif sentiment_roll < 0.80:
            # negative review
            rating = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
            template = random.choice(_NEGATIVE_TEMPLATES)
            adj = random.choice(_NEG_ADJ)
            staff_adj = random.choice(_NEG_STAFF)
        else:
            # neutral review
            rating = random.choices([3, 4], weights=[0.6, 0.4])[0]
            template = random.choice(_NEUTRAL_TEMPLATES)
            adj = random.choice(_NEU_ADJ)
            staff_adj = random.choice(_POS_STAFF if random.random() > 0.5 else _NEG_STAFF)

        aspect = random.choice(_ASPECTS)
        text = template.format(aspect=aspect, adj=adj, staff_adj=staff_adj)

        # Add extra sentences for realism
        extra_sentences = [
            f"The location in {random.choice(_CITIES)} was convenient.",
            "WiFi was fast and reliable throughout the stay.",
            "Parking was easy to find and reasonably priced.",
            "The breakfast buffet offered a good variety of options.",
            "Check-in and check-out were smooth and efficient.",
            "The pool and gym facilities were well-maintained.",
            "Room service was prompt and the food was tasty.",
            "The concierge was very knowledgeable about local attractions.",
            "Housekeeping was thorough and came at convenient times.",
            "The noise levels were acceptable throughout the night.",
        ]
        text += " " + " ".join(random.sample(extra_sentences, k=random.randint(1, 3)))

        review_date = start_date + timedelta(days=random.randint(0, date_range))
        business = random.choice(_BUSINESS_NAMES)
        source = random.choice(["yelp", "airbnb", "google"])

        records.append({
            "review_id": f"SYN_{i:06d}",
            "text": text,
            "rating": rating,
            "date": review_date.strftime("%Y-%m-%d"),
            "source": source,
            "business_name": business,
            "city": random.choice(_CITIES),
            "useful": random.randint(0, 20),
            "funny": random.randint(0, 10),
            "cool": random.randint(0, 15),
            "is_synthetic": True,
        })

    df = pd.DataFrame(records)
    logger.info("Synthetic data generated: %d rows, rating distribution: %s",
                len(df), df["rating"].value_counts().to_dict())
    return df


def load_yelp_data(filepath: str, max_records: int = 500000) -> pd.DataFrame:
    """Load Yelp JSON dataset; falls back to synthetic data if file is absent."""
    path = Path(filepath)
    if not path.exists():
        logger.warning("Yelp data not found at %s — generating synthetic reviews", filepath)
        return generate_synthetic_reviews(n=2000)

    logger.info("Loading Yelp data from %s (max %d records)", filepath, max_records)
    records = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append({
                        "review_id": record.get("review_id", f"YELP_{i}"),
                        "text": record.get("text", ""),
                        "rating": float(record.get("stars", 3)),
                        "date": record.get("date", "2022-01-01"),
                        "source": "yelp",
                        "business_name": record.get("business_id", "Unknown"),
                        "city": "",
                        "useful": record.get("useful", 0),
                        "funny": record.get("funny", 0),
                        "cool": record.get("cool", 0),
                        "is_synthetic": False,
                    })
                except json.JSONDecodeError as exc:
                    logger.debug("Skipping malformed JSON line %d: %s", i, exc)
    except OSError as exc:
        logger.error("Failed to read Yelp file %s: %s", filepath, exc)
        return generate_synthetic_reviews(n=2000)

    df = pd.DataFrame(records)
    logger.info("Loaded %d Yelp records", len(df))
    return df


def load_airbnb_data(filepath: str) -> pd.DataFrame:
    """Load Airbnb reviews CSV."""
    path = Path(filepath)
    if not path.exists():
        logger.warning("Airbnb data not found at %s", filepath)
        return pd.DataFrame()

    logger.info("Loading Airbnb data from %s", filepath)
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    except Exception as exc:
        logger.error("Failed to load Airbnb CSV %s: %s", filepath, exc)
        return pd.DataFrame()

    logger.info("Raw Airbnb columns: %s", df.columns.tolist())

    # Normalize column names to unified schema
    col_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    for candidate in ["comments", "review", "text", "review_text", "body"]:
        if candidate in col_lower:
            col_map[col_lower[candidate]] = "text"
            break

    for candidate in ["overall_satisfaction", "rating", "stars", "score", "review_scores_rating"]:
        if candidate in col_lower:
            col_map[col_lower[candidate]] = "rating"
            break

    for candidate in ["date", "review_date", "created_at", "submitted_at"]:
        if candidate in col_lower:
            col_map[col_lower[candidate]] = "date"
            break

    for candidate in ["listing_name", "name", "property_name", "business_name", "listing_id"]:
        if candidate in col_lower:
            col_map[col_lower[candidate]] = "business_name"
            break

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "text" not in df.columns:
        # Try to combine available text columns
        text_cols = [c for c in df.columns if "comment" in c.lower() or "review" in c.lower()]
        if text_cols:
            df["text"] = df[text_cols[0]].fillna("").astype(str)
        else:
            df["text"] = ""

    if "rating" not in df.columns:
        df["rating"] = 3.0
    if "date" not in df.columns:
        df["date"] = "2022-01-01"
    if "business_name" not in df.columns:
        df["business_name"] = "Airbnb Property"

    df["source"] = "airbnb"
    df["is_synthetic"] = False

    # Drop rows with empty text
    df = df[df["text"].astype(str).str.strip().str.len() > 10].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)

    # Normalize rating to 1-5 scale (Airbnb uses 20-100)
    if df["rating"].max() > 10:
        df["rating"] = (df["rating"] / 20.0).clip(1, 5).round(1)
    elif df["rating"].max() > 5:
        df["rating"] = (df["rating"] / 10.0 * 5).clip(1, 5).round(1)

    logger.info("Loaded %d Airbnb records", len(df))
    return df


def merge_sources(yelp_df: pd.DataFrame, airbnb_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge Yelp and Airbnb DataFrames into a unified schema."""
    required_cols = ["review_id", "text", "rating", "date", "source", "business_name", "is_synthetic"]

    def _normalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        out = df.copy()
        if "review_id" not in out.columns:
            out["review_id"] = [f"{source_name.upper()}_{i}" for i in range(len(out))]
        if "is_synthetic" not in out.columns:
            out["is_synthetic"] = False
        for col in required_cols:
            if col not in out.columns:
                out[col] = None
        return out[required_cols]

    frames = [_normalize(yelp_df, "yelp")]

    if airbnb_df is not None and not airbnb_df.empty:
        frames.append(_normalize(airbnb_df, "airbnb"))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].astype(str).str.strip().str.len() > 5]
    merged["text"] = merged["text"].astype(str).str.strip()
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(3.0).clip(1, 5)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").fillna(pd.Timestamp("2022-01-01"))

    logger.info("Merged dataset: %d records from sources: %s",
                len(merged), merged["source"].value_counts().to_dict())
    return merged


def load_all_data(base_dir: str = ".") -> pd.DataFrame:
    """Load all available data sources and return merged DataFrame."""
    base = Path(base_dir)
    yelp_path = base / "data" / "raw" / "yelp_academic_dataset_review.json"
    airbnb_candidates = [
        base / "Dataset" / "Cleaned Data" / "reviews.csv",
        base / "Dataset" / "User Review" / "User_reviews.csv",
        base / "reviews.csv",
        base / "User_reviews.csv",
    ]

    yelp_df = load_yelp_data(str(yelp_path))

    airbnb_df = None
    for candidate in airbnb_candidates:
        if candidate.exists():
            logger.info("Found Airbnb data at %s", candidate)
            airbnb_df = load_airbnb_data(str(candidate))
            if not airbnb_df.empty:
                break

    return merge_sources(yelp_df, airbnb_df)
