"""
Multi-touch attribution: Shapley-value, last-touch, first-touch, linear,
and time-decay models for campaign channel credit allocation.
"""

import numpy as np
import pandas as pd
import math
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

CHANNELS = ["email", "social", "paid_search", "display", "organic", "affiliate", "sms"]


def build_journey_data(df: pd.DataFrame, n_customers: int = 100_000) -> pd.DataFrame:
    """
    Reconstruct marketing journeys from events dataframe.
    Each row = one customer's ordered touchpoint sequence.
    """
    if len(df) > n_customers:
        customers = df["customer_id"].drop_duplicates().sample(n_customers, random_state=42)
        df = df[df["customer_id"].isin(customers)]

    journeys = (
        df.groupby("customer_id")
        .apply(lambda g: {
            "touchpoints": list(g["channel"]),
            "converted": int(g["conversions"].sum() > 0),
            "revenue": float(g["monetary"].sum()),
        })
        .reset_index()
    )
    journeys = pd.json_normalize(journeys[0])
    journeys["customer_id"] = df["customer_id"].drop_duplicates().head(len(journeys)).values
    return journeys


def last_touch_attribution(journeys: pd.DataFrame) -> pd.Series:
    """100% credit to the last touchpoint before conversion."""
    converted = journeys[journeys["converted"] == 1]
    last = converted["touchpoints"].apply(lambda x: x[-1] if x else None)
    revenue = converted["revenue"]
    result = pd.Series(dtype=float, index=CHANNELS).fillna(0.0)
    for ch, rev in zip(last, revenue):
        if ch and ch in result.index:
            result[ch] += rev
    return result


def first_touch_attribution(journeys: pd.DataFrame) -> pd.Series:
    """100% credit to the first touchpoint."""
    converted = journeys[journeys["converted"] == 1]
    first = converted["touchpoints"].apply(lambda x: x[0] if x else None)
    revenue = converted["revenue"]
    result = pd.Series(dtype=float, index=CHANNELS).fillna(0.0)
    for ch, rev in zip(first, revenue):
        if ch and ch in result.index:
            result[ch] += rev
    return result


def linear_attribution(journeys: pd.DataFrame) -> pd.Series:
    """Equal credit across all touchpoints."""
    converted = journeys[journeys["converted"] == 1]
    result = pd.Series(dtype=float, index=CHANNELS).fillna(0.0)
    for _, row in converted.iterrows():
        touches = [t for t in row["touchpoints"] if t in CHANNELS]
        if not touches:
            continue
        credit = row["revenue"] / len(touches)
        for ch in touches:
            result[ch] += credit
    return result


def time_decay_attribution(journeys: pd.DataFrame, half_life: float = 7.0) -> pd.Series:
    """More credit to touchpoints closer to conversion."""
    converted = journeys[journeys["converted"] == 1]
    result = pd.Series(dtype=float, index=CHANNELS).fillna(0.0)
    for _, row in converted.iterrows():
        touches = [t for t in row["touchpoints"] if t in CHANNELS]
        if not touches:
            continue
        n = len(touches)
        weights = np.array([2 ** ((i - n + 1) / half_life) for i in range(n)])
        weights /= weights.sum()
        for ch, w in zip(touches, weights):
            result[ch] += row["revenue"] * w
    return result


def shapley_attribution(journeys: pd.DataFrame, sample: int = 10_000) -> pd.Series:
    """
    Shapley-value data-driven attribution.
    Approximated: for each coalition, estimate conversion probability as
    fraction of journeys containing that subset that converted.
    """
    if len(journeys) > sample:
        journeys = journeys.sample(sample, random_state=42)

    converted = journeys[journeys["converted"] == 1]
    total_rev = converted["revenue"].sum()
    result = pd.Series(0.0, index=CHANNELS)

    # Pre-compute conversion rate by channel set (frozenset)
    def coalition_value(coalition: frozenset) -> float:
        if not coalition:
            return 0.0
        mask = journeys["touchpoints"].apply(
            lambda t: bool(set(t) & coalition)
        )
        subset = journeys[mask]
        if len(subset) == 0:
            return 0.0
        return subset["converted"].mean()

    channels_present = set(ch for row in journeys["touchpoints"] for ch in row if ch in CHANNELS)

    for ch in channels_present:
        others = [c for c in channels_present if c != ch]
        shapley_val = 0.0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                s = frozenset(subset)
                marginal = coalition_value(s | {ch}) - coalition_value(s)
                weight = (
                    math.factorial(size)
                    * math.factorial(len(channels_present) - size - 1)
                    / math.factorial(len(channels_present))
                )
                shapley_val += weight * marginal
        result[ch] = shapley_val * total_rev

    result = result.clip(lower=0)
    return result


def run_attribution_analysis(save: bool = True) -> pd.DataFrame:
    """Run all attribution models and compare results."""
    path = DATA_PROC / "synthetic_events.parquet"
    if not path.exists():
        raise FileNotFoundError("Run data_pipeline.prepare_all() first.")

    df = pd.read_parquet(path)
    print("Building customer journeys ...")
    journeys = build_journey_data(df, n_customers=50_000)

    print("Running attribution models ...")
    results = pd.DataFrame({
        "last_touch":  last_touch_attribution(journeys),
        "first_touch": first_touch_attribution(journeys),
        "linear":      linear_attribution(journeys),
        "time_decay":  time_decay_attribution(journeys),
        "shapley":     shapley_attribution(journeys, sample=5_000),
    }).fillna(0)

    # Normalise to percentages
    results_pct = results.div(results.sum(axis=0), axis=1) * 100
    results_pct = results_pct.round(2)

    if save:
        results.to_csv(DATA_PROC / "attribution_revenue.csv")
        results_pct.to_csv(DATA_PROC / "attribution_pct.csv")
        print("Attribution results saved.")

    print("\nAttribution % by channel:")
    print(results_pct.to_string())
    return results_pct


if __name__ == "__main__":
    run_attribution_analysis()
