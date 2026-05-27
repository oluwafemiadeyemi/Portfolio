"""
Data Loader for Customer Lifetime Value & Retention Platform.
Supports KKBox, IBM Telco, existing CSV, and synthetic streaming data generation.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def load_kkbox_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and merge KKBox competition files: train.csv, members.csv, transactions.csv."""
    data_dir = Path(data_dir)
    required = ["train.csv", "members_v3.csv", "transactions_v2.csv"]
    alt_names = {
        "members_v3.csv": ["members.csv", "members_v2.csv"],
        "transactions_v2.csv": ["transactions.csv", "user_logs_v2.csv", "user_logs.csv"],
    }

    def find_file(name: str, alts: list[str]) -> Optional[Path]:
        for fname in [name] + alts:
            p = data_dir / fname
            if p.exists():
                return p
        return None

    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {data_dir}")

    logger.info("Loading KKBox train.csv...")
    train_df = pd.read_csv(train_path)

    members_path = find_file("members_v3.csv", alt_names["members_v3.csv"])
    if members_path:
        logger.info(f"Loading members from {members_path.name}...")
        members_df = pd.read_csv(members_path)
        train_df = train_df.merge(members_df, on="msno", how="left")

    txn_path = find_file("transactions_v2.csv", alt_names["transactions_v2.csv"])
    if txn_path:
        logger.info(f"Loading transactions from {txn_path.name}...")
        txn_df = pd.read_csv(txn_path, nrows=5_000_000)
        agg = aggregate_transactions(txn_df, train_df)
        train_df = train_df.merge(agg, on="msno", how="left")

    train_df.rename(columns={"is_churn": "churn_flag"}, inplace=True)
    logger.info(f"KKBox dataset loaded: {len(train_df):,} users")
    return train_df


def load_telco_churn(filepath: str | Path) -> pd.DataFrame:
    """Load IBM Telco Customer Churn CSV and standardize column names."""
    filepath = Path(filepath)
    logger.info(f"Loading IBM Telco Churn from {filepath}...")
    df = pd.read_csv(filepath)

    rename_map = {
        "customerID": "user_id",
        "tenure": "tenure_days",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn_flag",
        "gender": "gender",
        "SeniorCitizen": "is_senior",
        "Partner": "has_partner",
        "Dependents": "has_dependents",
        "PhoneService": "has_phone",
        "InternetService": "internet_service",
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "PaperlessBilling": "paperless_billing",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "churn_flag" in df.columns:
        df["churn_flag"] = df["churn_flag"].map({"Yes": 1, "No": 0}).fillna(df["churn_flag"])

    if "tenure_days" in df.columns:
        df["tenure_days"] = df["tenure_days"] * 30  # months → days approx

    df["total_charges"] = pd.to_numeric(df.get("total_charges", pd.Series(dtype=float)), errors="coerce")
    logger.info(f"Telco dataset loaded: {len(df):,} customers, churn rate={df['churn_flag'].mean():.1%}")
    return df


def load_existing_churn(filepath: str | Path) -> pd.DataFrame:
    """Load existing customer_churn.csv from project root and standardize."""
    filepath = Path(filepath)
    logger.info(f"Loading existing churn data from {filepath}...")
    df = pd.read_csv(filepath)

    # Standardize whatever columns exist
    col_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=col_lower, inplace=True)

    churn_candidates = [c for c in df.columns if "churn" in c.lower()]
    if churn_candidates and "churn_flag" not in df.columns:
        df.rename(columns={churn_candidates[0]: "churn_flag"}, inplace=True)

    if "churn_flag" in df.columns:
        df["churn_flag"] = pd.to_numeric(df["churn_flag"], errors="coerce").fillna(0).astype(int)

    if "user_id" not in df.columns:
        if "names" in df.columns:
            df.rename(columns={"names": "user_id"}, inplace=True)
        else:
            df["user_id"] = df.index.astype(str)

    logger.info(f"Existing dataset loaded: {len(df):,} rows, cols={list(df.columns)}")
    return df


def generate_synthetic_streaming_data(n_users: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic music streaming churn dataset.
    Churn patterns:
      - days_since_last_login > 30 increases churn risk
      - avg_completion_rate < 0.3 increases churn risk
      - plan_list_price == 98 AND low engagement increases churn risk
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating synthetic streaming dataset: {n_users:,} users...")

    user_ids = [f"U{i:07d}" for i in range(n_users)]
    reg_dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, n_users), unit="D"
    )

    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
              "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    genders = ["M", "F", "Other"]
    plans = [98, 148, 198]
    payment_methods = list(range(1, 42))
    reg_via = [3, 4, 7]

    city_arr = rng.choice(cities, n_users)
    gender_arr = rng.choice(genders, n_users, p=[0.48, 0.49, 0.03])
    plan_arr = rng.choice(plans, n_users, p=[0.45, 0.35, 0.20])
    payment_arr = rng.choice(payment_methods, n_users)
    reg_via_arr = rng.choice(reg_via, n_users, p=[0.35, 0.45, 0.20])
    age_arr = rng.integers(18, 66, n_users)

    sub_start = reg_dates + pd.to_timedelta(rng.integers(0, 30, n_users), unit="D")
    tenure_days = rng.integers(1, 1826, n_users)  # up to 5 years
    sub_end = sub_start + pd.to_timedelta(tenure_days, unit="D")

    # Listening behavior
    total_secs = rng.integers(0, 2_592_000, n_users).astype(float)  # up to 30 days of seconds
    num_25 = rng.integers(0, 300, n_users)
    num_50 = (num_25 * rng.uniform(0.3, 0.9, n_users)).astype(int)
    num_75 = (num_50 * rng.uniform(0.3, 0.9, n_users)).astype(int)
    num_985 = (num_75 * rng.uniform(0.3, 0.9, n_users)).astype(int)
    num_100 = (num_985 * rng.uniform(0.3, 0.95, n_users)).astype(int)
    num_unq = rng.integers(1, 500, n_users)

    total_plays = num_25 + num_50 + num_75 + num_985 + num_100 + 1
    avg_completion_rate = (
        0.25 * num_25 + 0.50 * num_50 + 0.75 * num_75 + 0.985 * num_985 + 1.0 * num_100
    ) / total_plays

    days_since_last_login = rng.integers(0, 181, n_users)
    engagement_score = total_secs / (30 * 86400)  # fraction of full month

    # Churn probability logic
    churn_prob = np.zeros(n_users)
    churn_prob += np.where(days_since_last_login > 60, 0.40, 0.0)
    churn_prob += np.where(days_since_last_login > 30, 0.20, 0.0)
    churn_prob += np.where(avg_completion_rate < 0.3, 0.25, 0.0)
    churn_prob += np.where(avg_completion_rate < 0.5, 0.10, 0.0)
    churn_prob += np.where((plan_arr == 98) & (engagement_score < 0.1), 0.20, 0.0)
    churn_prob += np.where(plan_arr == 98, 0.05, 0.0)
    churn_prob += np.where(tenure_days < 30, 0.10, 0.0)  # onboarding drop-off
    churn_prob += rng.uniform(0.0, 0.10, n_users)
    churn_prob = np.clip(churn_prob, 0.02, 0.92)

    churn_flag = (rng.uniform(0, 1, n_users) < churn_prob).astype(int)
    actual_churn_rate = churn_flag.mean()
    logger.info(f"Synthetic data: churn rate = {actual_churn_rate:.1%}")

    df = pd.DataFrame({
        "user_id": user_ids,
        "registration_date": reg_dates,
        "city": city_arr,
        "gender": gender_arr,
        "registered_via": reg_via_arr,
        "bd": age_arr,
        "plan_list_price": plan_arr,
        "payment_method_id": payment_arr,
        "subscription_start_date": sub_start,
        "subscription_end_date": sub_end,
        "tenure_days": tenure_days,
        "num_25": num_25,
        "num_50": num_50,
        "num_75": num_75,
        "num_985": num_985,
        "num_100": num_100,
        "num_unq": num_unq,
        "total_secs": total_secs,
        "days_since_last_login": days_since_last_login,
        "avg_completion_rate": avg_completion_rate,
        "engagement_score": engagement_score,
        "churn_flag": churn_flag,
    })

    return df


def aggregate_transactions(
    transactions_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute RFM-style aggregates per user from transaction logs."""
    id_col = "msno" if "msno" in transactions_df.columns else "user_id"

    agg_dict: dict = {}

    if "transaction_date" in transactions_df.columns:
        transactions_df["transaction_date"] = pd.to_datetime(
            transactions_df["transaction_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        max_date = transactions_df["transaction_date"].max()
        last_txn = transactions_df.groupby(id_col)["transaction_date"].max()
        agg_dict["days_since_last_transaction"] = (max_date - last_txn).dt.days
        agg_dict["transaction_count"] = transactions_df.groupby(id_col).size()

    if "actual_amount_paid" in transactions_df.columns:
        agg_dict["total_amount_paid"] = transactions_df.groupby(id_col)["actual_amount_paid"].sum()
        agg_dict["avg_amount_paid"] = transactions_df.groupby(id_col)["actual_amount_paid"].mean()

    if "plan_list_price" in transactions_df.columns and id_col in transactions_df.columns:
        agg_dict["plan_list_price_mode"] = transactions_df.groupby(id_col)["plan_list_price"].agg(
            lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan
        )

    if "is_auto_renew" in transactions_df.columns:
        agg_dict["auto_renew_rate"] = transactions_df.groupby(id_col)["is_auto_renew"].mean()

    if "is_cancel" in transactions_df.columns:
        agg_dict["cancel_rate"] = transactions_df.groupby(id_col)["is_cancel"].mean()

    if not agg_dict:
        logger.warning("No recognizable transaction columns found for aggregation.")
        return pd.DataFrame({id_col: users_df[id_col] if id_col in users_df.columns else []})

    result = pd.DataFrame(agg_dict).reset_index()
    result.rename(columns={"index": id_col}, inplace=True)
    logger.info(f"Transaction aggregation complete: {len(result):,} users, {result.shape[1]} features")
    return result


def load_data(
    data_dir: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    """
    Load data from multiple sources in priority order:
    1. KKBox (train.csv + members + transactions)
    2. IBM Telco (telco.csv or WA_Fn-UseC_-Telco-Customer-Churn.csv)
    3. Existing customer_churn.csv in project root
    4. Synthetic streaming data (fallback)
    """
    data_dir = Path(data_dir)
    project_root = Path(project_root)

    # Try KKBox
    if (data_dir / "train.csv").exists():
        try:
            logger.info("Found KKBox train.csv — loading KKBox dataset...")
            return load_kkbox_data(data_dir)
        except Exception as exc:
            logger.warning(f"KKBox load failed: {exc}")

    # Try IBM Telco
    telco_candidates = [
        data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        data_dir / "telco.csv",
        data_dir / "telco_churn.csv",
        data_dir / "Telco-Customer-Churn.csv",
    ]
    for telco_path in telco_candidates:
        if telco_path.exists():
            try:
                logger.info(f"Found Telco CSV at {telco_path.name}...")
                return load_telco_churn(telco_path)
            except Exception as exc:
                logger.warning(f"Telco load failed: {exc}")

    # Try existing project root churn CSV
    root_candidates = [
        project_root / "customer_churn.csv",
        project_root / "churn.csv",
        data_dir / "churn.csv",
    ]
    for root_path in root_candidates:
        if root_path.exists():
            try:
                logger.info(f"Found existing churn CSV at {root_path}...")
                return load_existing_churn(root_path)
            except Exception as exc:
                logger.warning(f"Existing CSV load failed: {exc}")

    # Fallback: synthetic
    logger.info("No existing data found — generating synthetic streaming dataset (200k users)...")
    return generate_synthetic_streaming_data(n_users=200_000)
