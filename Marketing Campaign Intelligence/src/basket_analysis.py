"""
Market basket analysis using FP-Growth algorithm.
Mines association rules from product-channel purchase patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"


def build_transaction_data(df: pd.DataFrame, max_rows: int = 500_000) -> pd.DataFrame:
    """
    Build a binary transaction matrix: each customer × product_category.
    Limits to max_rows for FP-Growth memory efficiency.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    # Create basket: which product categories did each customer interact with?
    # Group by customer, collect all product categories they engaged with
    baskets = (
        df.groupby("customer_id")["product_category"]
        .apply(list)
        .reset_index()
    )

    # Add channel as items too (channel-product affinity)
    channel_baskets = (
        df.groupby("customer_id")["channel"]
        .apply(lambda x: [f"ch:{v}" for v in set(x)])
        .reset_index()
    )

    merged = baskets.merge(channel_baskets, on="customer_id")
    merged["items"] = merged["product_category"] + merged["channel"]

    te = TransactionEncoder()
    te_array = te.fit_transform(merged["items"].tolist())
    basket_df = pd.DataFrame(te_array, columns=te.columns_)
    return basket_df


def run_fpgrowth(basket_df: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """Run FP-Growth and return frequent itemsets."""
    print(f"Running FP-Growth on {len(basket_df):,} transactions, min_support={min_support} ...")
    frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"Found {len(frequent_itemsets):,} frequent itemsets")
    return frequent_itemsets


def extract_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "lift",
    min_threshold: float = 1.2,
) -> pd.DataFrame:
    """Extract association rules from frequent itemsets."""
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    print(f"Extracted {len(rules):,} association rules (lift >= {min_threshold})")
    return rules


def top_product_affinities(rules: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top N product-product rules excluding channel items."""
    product_rules = rules[
        ~rules["antecedents_str"].str.contains("ch:") &
        ~rules["consequents_str"].str.contains("ch:")
    ]
    return product_rules.head(top_n)[
        ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    ]


def run_basket_analysis(save: bool = True) -> dict:
    """End-to-end basket analysis pipeline."""
    path = DATA_PROC / "synthetic_events.parquet"
    if not path.exists():
        raise FileNotFoundError("Run data_pipeline.prepare_all() first.")

    df = pd.read_parquet(path, columns=["customer_id", "product_category", "channel"])
    basket_df = build_transaction_data(df)
    frequent_itemsets = run_fpgrowth(basket_df)
    rules = extract_rules(frequent_itemsets)
    top_affinities = top_product_affinities(rules)

    if save:
        rules.to_parquet(DATA_PROC / "association_rules.parquet", index=False)
        top_affinities.to_csv(DATA_PROC / "top_product_affinities.csv", index=False)
        print("Basket analysis saved.")

    return {"frequent_itemsets": frequent_itemsets, "rules": rules, "top_affinities": top_affinities}


if __name__ == "__main__":
    result = run_basket_analysis()
    print("\nTop 10 Product Affinities:")
    print(result["top_affinities"].head(10).to_string())
