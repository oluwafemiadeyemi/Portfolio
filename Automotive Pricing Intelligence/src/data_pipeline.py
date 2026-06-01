"""
Data pipeline: generates 3M+ synthetic used car listings (Craigslist-scale)
with realistic pricing patterns. Supports loading real Craigslist CSV if available.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

MAKES = {
    "Toyota":  {"models": ["Camry", "Corolla", "RAV4", "Tacoma", "Prius", "Highlander"], "base": 22000, "prestige": 1.05},
    "Honda":   {"models": ["Civic", "Accord", "CR-V", "Pilot", "Fit", "Odyssey"],        "base": 21000, "prestige": 1.04},
    "Ford":    {"models": ["F-150", "Mustang", "Explorer", "Escape", "Focus", "Edge"],   "base": 28000, "prestige": 1.03},
    "Chevrolet": {"models": ["Silverado", "Equinox", "Malibu", "Tahoe", "Traverse"],     "base": 27000, "prestige": 1.02},
    "BMW":     {"models": ["3 Series", "5 Series", "X3", "X5", "M3", "7 Series"],        "base": 45000, "prestige": 1.30},
    "Mercedes": {"models": ["C-Class", "E-Class", "GLE", "GLC", "S-Class", "A-Class"],   "base": 50000, "prestige": 1.40},
    "Tesla":   {"models": ["Model 3", "Model Y", "Model S", "Model X"],                  "base": 55000, "prestige": 1.50},
    "Hyundai": {"models": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona"],           "base": 19000, "prestige": 0.98},
    "Nissan":  {"models": ["Altima", "Rogue", "Sentra", "Maxima", "Pathfinder"],         "base": 20000, "prestige": 1.00},
    "Jeep":    {"models": ["Wrangler", "Cherokee", "Grand Cherokee", "Compass"],          "base": 32000, "prestige": 1.10},
    "Audi":    {"models": ["A4", "A6", "Q5", "Q7", "A3", "e-tron"],                     "base": 43000, "prestige": 1.28},
    "Volkswagen": {"models": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf"],             "base": 24000, "prestige": 1.05},
}

STATES = ["CA", "TX", "FL", "NY", "OH", "PA", "IL", "GA", "NC", "MI",
          "WA", "AZ", "CO", "MA", "TN", "MN", "IN", "MO", "WI", "NV"]

CONDITIONS = ["new", "like new", "excellent", "good", "fair", "salvage"]
FUEL_TYPES = ["gas", "diesel", "hybrid", "electric", "other"]
TRANSMISSIONS = ["automatic", "manual", "cvt"]
DRIVE_TYPES = ["fwd", "rwd", "4wd", "awd"]
PAINT_COLORS = ["white", "black", "silver", "gray", "blue", "red", "green", "brown", "custom"]
TITLE_STATUS = ["clean", "rebuilt", "lien", "missing", "parts only", "salvage"]
VEHICLE_TYPES = ["sedan", "SUV", "truck", "coupe", "hatchback", "wagon", "van", "convertible", "pickup"]


def _depreciation_curve(age_years: float) -> float:
    """Standard exponential depreciation: ~15%/yr for first 5y, ~10%/yr after."""
    if age_years <= 5:
        return np.exp(-0.16 * age_years)
    return np.exp(-0.16 * 5) * np.exp(-0.10 * (age_years - 5))


def _mileage_penalty(odometer: float) -> float:
    """Price decreases logarithmically with mileage past 20k."""
    baseline = 20_000
    if odometer <= baseline:
        return 1.0
    return max(0.3, 1.0 - 0.12 * np.log(odometer / baseline))


def generate_synthetic_listings(n: int = 3_000_000) -> pd.DataFrame:
    """Generate n used car listings with realistic price modelling."""
    print(f"Generating {n:,} vehicle listings ...")
    chunk_size = 300_000
    frames = []

    makes_list = list(MAKES.keys())

    for start in tqdm(range(0, n, chunk_size), desc="Chunks"):
        end = min(start + chunk_size, n)
        sz = end - start

        make_idx = RNG.choice(len(makes_list), size=sz)
        makes = [makes_list[i] for i in make_idx]
        models = [RNG.choice(MAKES[m]["models"]) for m in makes]
        base_prices = np.array([MAKES[m]["base"] for m in makes])
        prestige = np.array([MAKES[m]["prestige"] for m in makes])

        year = RNG.integers(2005, 2025, size=sz)
        current_year = 2025
        age = current_year - year

        odometer = np.clip(
            RNG.normal(age * 12_000, age * 3_000, sz), 0, 350_000
        ).astype(int)

        dep = np.array([_depreciation_curve(a) for a in age])
        mil = np.array([_mileage_penalty(o) for o in odometer])

        condition = RNG.choice(CONDITIONS, size=sz,
                               p=[0.05, 0.15, 0.25, 0.35, 0.15, 0.05])
        cond_mult = {"new": 1.00, "like new": 0.97, "excellent": 0.92,
                     "good": 0.82, "fair": 0.65, "salvage": 0.35}
        cond_factor = np.array([cond_mult[c] for c in condition])

        fuel = RNG.choice(FUEL_TYPES, size=sz, p=[0.68, 0.08, 0.12, 0.10, 0.02])
        fuel_mult = {"gas": 1.0, "diesel": 1.05, "hybrid": 1.08, "electric": 1.15, "other": 0.95}
        fuel_factor = np.array([fuel_mult[f] for f in fuel])

        transmission = RNG.choice(TRANSMISSIONS, size=sz, p=[0.75, 0.10, 0.15])
        drive = RNG.choice(DRIVE_TYPES, size=sz, p=[0.40, 0.20, 0.20, 0.20])
        state = RNG.choice(STATES, size=sz)
        color = RNG.choice(PAINT_COLORS, size=sz)
        title = RNG.choice(TITLE_STATUS, size=sz, p=[0.82, 0.06, 0.04, 0.03, 0.02, 0.03])
        title_mult = {"clean": 1.0, "rebuilt": 0.80, "lien": 0.90,
                      "missing": 0.70, "parts only": 0.30, "salvage": 0.35}
        title_factor = np.array([title_mult[t] for t in title])

        v_type = RNG.choice(VEHICLE_TYPES, size=sz)
        cylinders = RNG.choice([4, 6, 8, 3], size=sz, p=[0.45, 0.35, 0.15, 0.05])

        true_price = (
            base_prices * prestige * dep * mil *
            cond_factor * fuel_factor * title_factor *
            RNG.uniform(0.92, 1.08, sz)   # market noise
        )
        true_price = np.clip(true_price, 500, 150_000)

        chunk = pd.DataFrame({
            "id":           np.arange(start, end),
            "make":         makes,
            "model":        models,
            "year":         year,
            "age_years":    age,
            "odometer":     odometer,
            "condition":    condition,
            "fuel":         fuel,
            "transmission": transmission,
            "drive":        drive,
            "type":         v_type,
            "cylinders":    cylinders,
            "state":        state,
            "paint_color":  color,
            "title_status": title,
            "price":        true_price.round(0).astype(int),
        })
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"Generated {len(df):,} listings | Price range: ${df['price'].min():,}–${df['price'].max():,}")
    return df


def load_real_listings() -> pd.DataFrame:
    """Load Craigslist used cars CSV if placed in data/raw/."""
    for name in ["vehicles.csv", "craigslist_vehicles.csv", "used_cars.csv"]:
        p = DATA_RAW / name
        if p.exists():
            print(f"Loading real dataset: {p}")
            df = pd.read_csv(p, low_memory=False)
            df = df.dropna(subset=["price"])
            df = df[(df["price"] > 500) & (df["price"] < 200_000)]
            return df
    return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from raw listings."""
    df = df.copy()
    df["price_per_mile"] = (df["price"] / (df["odometer"] + 1)).round(4)
    df["age_x_mileage"]  = df["age_years"] * df["odometer"]
    df["is_luxury"]      = df["make"].isin(["BMW", "Mercedes", "Tesla", "Audi"]).astype(int)
    df["is_electric"]    = (df["fuel"] == "electric").astype(int)
    df["is_clean_title"] = (df["title_status"] == "clean").astype(int)
    df["mileage_bucket"] = pd.cut(
        df["odometer"],
        bins=[0, 15_000, 50_000, 100_000, 150_000, 200_000, 999_999],
        labels=["brand_new", "low", "moderate", "high", "very_high", "extreme"],
    )

    cat_cols = ["make", "model", "condition", "fuel", "transmission",
                "drive", "type", "state", "paint_color", "title_status", "mileage_bucket"]
    for col in cat_cols:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def prepare_all(n: int = 3_000_000) -> dict:
    """Full pipeline: generate/load data, engineer features, train/test split."""
    real = load_real_listings()
    if len(real) > 0:
        print(f"Using real dataset: {len(real):,} listings")
        df = real
    else:
        df = generate_synthetic_listings(n)

    df = engineer_features(df)
    df.to_parquet(DATA_PROC / "vehicles.parquet", index=False)
    print(f"Saved {len(df):,} listings to {DATA_PROC / 'vehicles.parquet'}")

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    train.to_parquet(DATA_PROC / "train.parquet", index=False)
    test.to_parquet(DATA_PROC / "test.parquet", index=False)
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    return {"df": df, "train": train, "test": test}


if __name__ == "__main__":
    prepare_all()
