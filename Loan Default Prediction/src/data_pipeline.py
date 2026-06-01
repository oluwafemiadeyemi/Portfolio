"""
Data pipeline: generates 2.5M synthetic loan applications (Lending Club scale)
with realistic default patterns across demographic groups.
Supports loading real Lending Club CSV if placed in data/raw/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

LOAN_PURPOSE   = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                  "medical", "car", "vacation", "small_business", "other"]
LOAN_GRADES    = ["A", "B", "C", "D", "E", "F", "G"]
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]
STATES         = ["CA", "TX", "FL", "NY", "OH", "PA", "IL", "GA", "NC", "MI",
                  "WA", "AZ", "CO", "MA", "TN", "MN", "IN", "MO", "WI", "NV"]
EMPLOYMENT     = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                  "6 years", "7 years", "8 years", "9 years", "10+ years"]

# Default probability per grade
DEFAULT_PROB = {"A": 0.04, "B": 0.08, "C": 0.14, "D": 0.22, "E": 0.30, "F": 0.40, "G": 0.50}


def _fico_from_grade(grade: str) -> tuple:
    ranges = {"A": (750, 800), "B": (700, 750), "C": (650, 700),
              "D": (600, 650), "E": (560, 600), "F": (520, 560), "G": (490, 520)}
    lo, hi = ranges[grade]
    return lo, hi


def generate_synthetic_loans(n: int = 2_500_000) -> pd.DataFrame:
    """Generate n synthetic loan applications with realistic default labels."""
    print(f"Generating {n:,} loan applications ...")
    chunk_size = 250_000
    frames = []

    grade_probs = [0.22, 0.28, 0.24, 0.14, 0.07, 0.03, 0.02]

    for start in tqdm(range(0, n, chunk_size), desc="Chunks"):
        end = min(start + chunk_size, n)
        sz = end - start

        grade = RNG.choice(LOAN_GRADES, size=sz, p=grade_probs)
        fico_low  = np.array([_fico_from_grade(g)[0] for g in grade])
        fico_high = np.array([_fico_from_grade(g)[1] for g in grade])
        fico = (fico_low + RNG.uniform(0, 1, sz) * (fico_high - fico_low)).astype(int)

        loan_amnt = np.clip(RNG.lognormal(9.5, 0.8, sz), 1000, 40_000).round(-2).astype(int)
        int_rate  = np.array([
            RNG.uniform({"A": 5, "B": 8, "C": 11, "D": 15, "E": 19, "F": 22, "G": 25}[g],
                        {"A": 8, "B": 11, "C": 15, "D": 19, "E": 22, "F": 25, "G": 29}[g])
            for g in grade
        ]).round(2)
        annual_inc = np.clip(RNG.lognormal(11.0, 0.7, sz), 20_000, 500_000).round(-2)
        dti        = np.clip(RNG.normal(18, 8, sz), 0, 50).round(2)
        emp_length = RNG.choice(EMPLOYMENT, size=sz)
        home       = RNG.choice(HOME_OWNERSHIP, size=sz, p=[0.45, 0.12, 0.40, 0.03])
        purpose    = RNG.choice(LOAN_PURPOSE, size=sz,
                                p=[0.40, 0.20, 0.10, 0.07, 0.05, 0.04, 0.03, 0.04, 0.07])
        state      = RNG.choice(STATES, size=sz)
        term       = RNG.choice([36, 60], size=sz, p=[0.60, 0.40])
        open_acc   = np.clip(RNG.poisson(11, sz), 1, 40)
        total_acc  = np.clip(open_acc + RNG.poisson(8, sz), open_acc, 60)
        revol_bal  = np.clip(RNG.lognormal(8.5, 1.0, sz), 0, 100_000).round(0)
        revol_util = np.clip(RNG.normal(50, 22, sz), 0, 100).round(1)
        delinq_2yrs = RNG.choice([0, 1, 2, 3], size=sz, p=[0.72, 0.15, 0.08, 0.05])
        pub_rec     = RNG.choice([0, 1, 2], size=sz, p=[0.85, 0.12, 0.03])
        inq_last_6m = RNG.choice([0, 1, 2, 3, 4, 5], size=sz,
                                  p=[0.30, 0.28, 0.20, 0.12, 0.06, 0.04])
        installment = (loan_amnt * (int_rate / 100 / 12) /
                       (1 - (1 + int_rate / 100 / 12) ** -term)).round(2)
        issue_year  = RNG.integers(2010, 2024, sz)

        # Default probability: grade baseline + modifiers
        p_default = np.array([DEFAULT_PROB[g] for g in grade])
        p_default += 0.002 * dti
        p_default += 0.001 * (100 - fico) / 100 * 5
        p_default += 0.005 * delinq_2yrs
        p_default += 0.003 * pub_rec
        p_default = np.clip(p_default, 0.01, 0.75)
        default = (RNG.uniform(size=sz) < p_default).astype(int)

        # Race/gender proxies for fairness analysis (NOT used in credit model)
        race   = RNG.choice(["White", "Black", "Hispanic", "Asian", "Other"],
                             size=sz, p=[0.60, 0.13, 0.17, 0.07, 0.03])
        gender = RNG.choice(["Male", "Female"], size=sz, p=[0.58, 0.42])

        chunk = pd.DataFrame({
            "loan_id":       np.arange(start, end),
            "loan_amnt":     loan_amnt,
            "term":          term,
            "int_rate":      int_rate,
            "installment":   installment,
            "grade":         grade,
            "emp_length":    emp_length,
            "home_ownership": home,
            "annual_inc":    annual_inc,
            "purpose":       purpose,
            "addr_state":    state,
            "dti":           dti,
            "delinq_2yrs":   delinq_2yrs,
            "inq_last_6mths": inq_last_6m,
            "open_acc":      open_acc,
            "pub_rec":       pub_rec,
            "revol_bal":     revol_bal,
            "revol_util":    revol_util,
            "total_acc":     total_acc,
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "fico_avg":      fico,
            "issue_year":    issue_year,
            "loan_default":  default,
            # Protected attributes (for fairness only)
            "race_proxy":    race,
            "gender_proxy":  gender,
        })
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    default_rate = df["loan_default"].mean() * 100
    print(f"Generated {len(df):,} loans | Default rate: {default_rate:.1f}%")
    return df


def load_real_lending_club() -> pd.DataFrame:
    """Load Lending Club CSV if available in data/raw/."""
    for name in ["accepted_2007_to_2018Q4.csv", "loan.csv", "lending_club.csv"]:
        p = DATA_RAW / name
        if p.exists():
            print(f"Loading Lending Club dataset: {p}")
            df = pd.read_csv(p, low_memory=False)
            return df
    return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for credit model."""
    df = df.copy()
    df["debt_to_income_ratio"] = (df["loan_amnt"] / (df["annual_inc"] + 1)).round(4)
    df["installment_to_income"] = (df["installment"] / (df["annual_inc"] / 12 + 1)).round(4)
    df["credit_utilisation"]   = df["revol_util"] / 100
    df["derog_flag"]           = ((df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)).astype(int)
    df["credit_history_score"] = (df["fico_avg"] - 400) / 450   # normalised
    df["risk_band"]            = pd.cut(
        df["fico_avg"], bins=[0, 580, 620, 660, 700, 740, 850],
        labels=["Very Poor", "Poor", "Fair", "Good", "Very Good", "Exceptional"],
    )
    df["log_annual_inc"]  = np.log1p(df["annual_inc"])
    df["log_revol_bal"]   = np.log1p(df["revol_bal"])

    # Encode categoricals
    for col in ["grade", "emp_length", "home_ownership", "purpose", "addr_state", "risk_band"]:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str = "loan_default",
                   bins: int = 10) -> pd.DataFrame:
    """Weight-of-Evidence and Information Value for credit scorecards."""
    df = df.copy()
    if df[feature].dtype in [float, np.float64]:
        df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    else:
        df["bin"] = df[feature]

    total_events   = df[target].sum()
    total_nonevents = len(df) - total_events

    woe_df = df.groupby("bin")[target].agg(
        events="sum", total="count"
    ).reset_index()
    woe_df["nonevents"] = woe_df["total"] - woe_df["events"]
    woe_df["event_rate"]    = woe_df["events"] / total_events
    woe_df["nonevent_rate"] = woe_df["nonevents"] / total_nonevents
    woe_df["woe"]  = np.log(woe_df["event_rate"] / (woe_df["nonevent_rate"] + 1e-10))
    woe_df["iv"]   = (woe_df["event_rate"] - woe_df["nonevent_rate"]) * woe_df["woe"]
    woe_df["IV_total"] = woe_df["iv"].sum()
    return woe_df


def prepare_all(n: int = 2_500_000) -> dict:
    """Full pipeline: load or generate data, engineer features, split."""
    real = load_real_lending_club()
    if len(real) > 0:
        df = real
    else:
        df = generate_synthetic_loans(n)

    df = engineer_features(df)
    df.to_parquet(DATA_PROC / "loans.parquet", index=False)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.15, stratify=df["loan_default"], random_state=42)
    train.to_parquet(DATA_PROC / "train.parquet", index=False)
    test.to_parquet(DATA_PROC / "test.parquet", index=False)
    print(f"Train: {len(train):,} | Test: {len(test):,} | Default rate: {train['loan_default'].mean()*100:.1f}%")
    return {"df": df, "train": train, "test": test}


if __name__ == "__main__":
    prepare_all()
