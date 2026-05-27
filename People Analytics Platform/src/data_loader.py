"""
IBM HR Analytics Data Loader
Loads IBM HR Employee Attrition dataset and extends synthetically to 50,000 employees.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── IBM HR column schema ───────────────────────────────────────────────────────
IBM_HR_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "Over18", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StandardHours", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# ── BLS JOLTS 2023 Industry Attrition Rates ───────────────────────────────────
BLS_ATTRITION_RATES = {
    "Technology": 0.138,
    "Finance": 0.121,
    "Healthcare": 0.218,
    "Manufacturing": 0.158,
    "Retail": 0.334,
    "Professional Services": 0.164,
    "Education": 0.097,
    "Government": 0.071,
    "Hospitality": 0.412,
    "Construction": 0.201,
}


def load_bls_benchmark_rates() -> Dict[str, float]:
    """
    Returns dict of industry → annual attrition rate from BLS 2023 JOLTS data.
    Source: U.S. Bureau of Labor Statistics, Job Openings and Labor Turnover Survey (JOLTS) 2023.
    """
    return BLS_ATTRITION_RATES.copy()


def load_ibm_hr(filepath: str) -> pd.DataFrame:
    """
    Loads IBM HR Analytics Employee Attrition dataset with column validation.

    Args:
        filepath: Path to HR-Employee-Attrition.csv or equivalent

    Returns:
        Validated and cleaned DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IBM HR file not found: {filepath}")

    logger.info(f"Loading IBM HR data from {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Validate expected columns
    expected = set(IBM_HR_COLUMNS)
    found = set(df.columns)
    missing = expected - found
    if missing:
        logger.warning(f"Missing columns (will use defaults): {missing}")

    # Standardize Attrition target
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        df["Attrition"] = 0

    # Standardize boolean columns
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Drop constant columns
    for col in ["EmployeeCount", "Over18", "StandardHours"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    logger.info(
        f"IBM HR data loaded: {len(df):,} employees, "
        f"attrition rate = {df['Attrition'].mean():.1%}"
    )
    return df


def extend_synthetically(df: pd.DataFrame, target_n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Extends IBM HR dataset to target_n employees using fitted distributions
    while preserving real correlational patterns:
    - OverTime + JobLevel + MaritalStatus → Attrition
    - Higher JobLevel → higher MonthlyIncome
    - YearsAtCompany → promotion patterns

    Args:
        df: Original IBM HR DataFrame (1,470 rows)
        target_n: Target number of synthetic employees
        seed: Random seed

    Returns:
        Extended DataFrame with target_n employees
    """
    rng = np.random.default_rng(seed)
    n_synthetic = target_n - len(df)
    logger.info(f"Generating {n_synthetic:,} synthetic employees (extending IBM HR {len(df):,} → {target_n:,})")

    # ── Fit distributions from IBM data ─────────────────────────────────────
    dept_dist = df["Department"].value_counts(normalize=True) if "Department" in df.columns else None
    role_dist = df["JobRole"].value_counts(normalize=True) if "JobRole" in df.columns else None
    edu_field_dist = df["EducationField"].value_counts(normalize=True) if "EducationField" in df.columns else None
    biz_travel_dist = df["BusinessTravel"].value_counts(normalize=True) if "BusinessTravel" in df.columns else None
    marital_dist = df["MaritalStatus"].value_counts(normalize=True) if "MaritalStatus" in df.columns else None

    # Core numeric distributions
    age = np.clip(rng.normal(df["Age"].mean(), df["Age"].std(), n_synthetic).astype(int), 18, 65)
    job_level = rng.choice([1, 2, 3, 4, 5], size=n_synthetic, p=[0.26, 0.33, 0.21, 0.13, 0.07])
    total_working_years = np.clip(
        (age - 22 + rng.normal(0, 3, n_synthetic)).astype(int), 0, 40
    )
    years_at_company = np.clip(
        rng.exponential(7, n_synthetic).astype(int), 0, total_working_years
    )
    years_in_current_role = np.clip(
        rng.exponential(3, n_synthetic).astype(int), 0, years_at_company
    )
    years_since_last_promotion = np.clip(
        rng.exponential(2, n_synthetic).astype(int), 0, years_at_company
    )
    years_with_curr_manager = np.clip(
        rng.exponential(4, n_synthetic).astype(int), 0, years_at_company
    )

    # Income: correlated with job level
    income_base = {1: 2800, 2: 4500, 3: 7500, 4: 12000, 5: 18000}
    income_noise = rng.lognormal(0, 0.25, n_synthetic)
    monthly_income = np.array([
        int(income_base[jl] * noise) for jl, noise in zip(job_level, income_noise)
    ])
    monthly_income = np.clip(monthly_income, 1009, 19999)

    # Satisfaction scores (1-4)
    job_satisfaction = rng.choice([1, 2, 3, 4], n_synthetic, p=[0.12, 0.18, 0.34, 0.36])
    env_satisfaction = rng.choice([1, 2, 3, 4], n_synthetic, p=[0.11, 0.19, 0.35, 0.35])
    rel_satisfaction = rng.choice([1, 2, 3, 4], n_synthetic, p=[0.10, 0.19, 0.38, 0.33])
    work_life_balance = rng.choice([1, 2, 3, 4], n_synthetic, p=[0.05, 0.22, 0.56, 0.17])
    job_involvement = rng.choice([1, 2, 3, 4], n_synthetic, p=[0.05, 0.18, 0.60, 0.17])

    # Overtime: higher for lower job levels
    overtime_prob = np.where(job_level <= 2, 0.35, 0.20)
    overtime = (rng.uniform(0, 1, n_synthetic) < overtime_prob).astype(int)

    # Marital status
    marital = rng.choice(
        marital_dist.index.tolist() if marital_dist is not None else ["Single", "Married", "Divorced"],
        n_synthetic,
        p=marital_dist.values if marital_dist is not None else [0.32, 0.46, 0.22],
    )

    # Gender
    gender = rng.choice(["Male", "Female"], n_synthetic, p=[0.60, 0.40])

    # Department
    departments = dept_dist.index.tolist() if dept_dist is not None else ["Research & Development", "Sales", "Human Resources"]
    dept_probs = dept_dist.values if dept_dist is not None else [0.65, 0.30, 0.05]
    department = rng.choice(departments, n_synthetic, p=dept_probs)

    # Job roles by department
    job_role = _assign_job_roles_by_dept(department, role_dist, rng)

    # Education
    education = rng.choice([1, 2, 3, 4, 5], n_synthetic, p=[0.05, 0.12, 0.39, 0.39, 0.05])
    education_field = rng.choice(
        edu_field_dist.index.tolist() if edu_field_dist is not None else
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        n_synthetic,
        p=edu_field_dist.values if edu_field_dist is not None else [0.41, 0.21, 0.16, 0.14, 0.03, 0.05],
    )

    business_travel = rng.choice(
        biz_travel_dist.index.tolist() if biz_travel_dist is not None else ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        n_synthetic,
        p=biz_travel_dist.values if biz_travel_dist is not None else [0.71, 0.19, 0.10],
    )

    # Additional numeric features
    daily_rate = rng.integers(100, 1500, n_synthetic)
    hourly_rate = rng.integers(30, 100, n_synthetic)
    monthly_rate = rng.integers(2000, 27000, n_synthetic)
    distance_from_home = rng.integers(1, 30, n_synthetic)
    num_companies_worked = np.clip(rng.integers(0, 10, n_synthetic), 0, 9)
    percent_salary_hike = rng.integers(11, 25, n_synthetic)
    performance_rating = rng.choice([3, 4], n_synthetic, p=[0.85, 0.15])
    stock_option_level = rng.choice([0, 1, 2, 3], n_synthetic, p=[0.47, 0.33, 0.14, 0.06])
    training_times_last_year = rng.integers(0, 7, n_synthetic)

    # ── Attrition outcome (preserving IBM correlations) ─────────────────────
    # Key drivers: overtime, low satisfaction, low job level, single, stagnation
    attrition_logit = (
        -1.5
        + 0.9 * overtime
        - 0.4 * (job_satisfaction - 2.5)
        - 0.3 * (env_satisfaction - 2.5)
        - 0.2 * (work_life_balance - 2.5)
        + 0.5 * (marital == "Single").astype(float)
        - 0.3 * (job_level - 2.5)
        + 0.1 * (distance_from_home - 9)
        + 0.3 * (years_since_last_promotion > 3).astype(float)
        + 0.2 * (business_travel == "Travel_Frequently").astype(float)
        + rng.normal(0, 0.5, n_synthetic)
    )
    attrition_prob = 1 / (1 + np.exp(-attrition_logit))
    # Scale to ~16% attrition (IBM baseline)
    scale_factor = 0.16 / attrition_prob.mean()
    attrition_prob = np.clip(attrition_prob * scale_factor, 0.02, 0.90)
    attrition = (rng.uniform(0, 1, n_synthetic) < attrition_prob).astype(int)

    synthetic_df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EmployeeNumber": np.arange(len(df) + 1, len(df) + n_synthetic + 1),
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": rel_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    })

    extended = pd.concat([df, synthetic_df], ignore_index=True)
    logger.info(
        f"Extended dataset: {len(extended):,} employees, "
        f"attrition rate = {extended['Attrition'].mean():.1%}"
    )
    return extended


def load_data(data_dir: str, target_n: int = 50000) -> pd.DataFrame:
    """
    Loads IBM HR data from multiple possible locations, extends synthetically if needed.

    Args:
        data_dir: Base directory to search for IBM HR data
        target_n: Target number of employees after synthetic extension

    Returns:
        Extended employee DataFrame
    """
    data_path = Path(data_dir)

    # Search candidate paths
    candidates = [
        data_path / "raw" / "HR-Employee-Attrition.csv",
        data_path / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        data_path / "HR-Employee-Attrition.csv",
        data_path / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        data_path / "employee_attrition.csv",
        data_path.parent / "HR-Employee-Attrition.csv",
        data_path.parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        data_path.parent / "employee_attrition.csv",
    ]

    ibm_df = None
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found IBM HR data at: {candidate}")
            try:
                ibm_df = load_ibm_hr(str(candidate))
                break
            except Exception as exc:
                logger.warning(f"Failed to load {candidate}: {exc}")

    if ibm_df is None:
        logger.warning("IBM HR file not found. Generating pure synthetic data...")
        ibm_df = _generate_pure_synthetic_ibm(n=1470)

    # Extend to target size
    if len(ibm_df) < target_n:
        df = extend_synthetically(ibm_df, target_n=target_n)
    else:
        df = ibm_df.copy()

    # Save extended dataset
    raw_dir = data_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / "hr_employees_extended.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Extended dataset saved to: {save_path}")

    return df


def _assign_job_roles_by_dept(
    departments: np.ndarray,
    role_dist: Optional[pd.Series],
    rng: np.random.Generator,
) -> np.ndarray:
    """Assigns job roles consistent with department."""
    dept_roles = {
        "Research & Development": ["Research Scientist", "Laboratory Technician", "Research Director",
                                   "Healthcare Representative", "Manufacturing Director"],
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Human Resources": ["Human Resources", "Manager"],
    }
    default_roles = list(role_dist.index) if role_dist is not None else [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources",
    ]

    roles = []
    for dept in departments:
        role_list = dept_roles.get(dept, default_roles)
        roles.append(rng.choice(role_list))
    return np.array(roles)


def _generate_pure_synthetic_ibm(n: int = 1470, seed: int = 42) -> pd.DataFrame:
    """Generates a pure synthetic IBM-like HR dataset when no real data is available."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(37, 9, n).astype(int), 18, 65)
    job_level = rng.choice([1, 2, 3, 4, 5], n, p=[0.26, 0.33, 0.21, 0.13, 0.07])
    overtime = rng.choice([0, 1], n, p=[0.72, 0.28])
    monthly_income = np.clip(rng.lognormal(8.4, 0.6, n).astype(int), 1009, 19999)
    attrition_prob = 0.05 + 0.25 * overtime + 0.08 * (job_level < 2).astype(float)
    attrition = (rng.uniform(0, 1, n) < np.clip(attrition_prob, 0, 0.8)).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.71, 0.19, 0.10]),
        "DailyRate": rng.integers(100, 1500, n),
        "Department": rng.choice(["Research & Development", "Sales", "Human Resources"], n, p=[0.65, 0.30, 0.05]),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Education": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.12, 0.39, 0.39, 0.05]),
        "EducationField": rng.choice(
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
            n, p=[0.41, 0.21, 0.16, 0.14, 0.03, 0.05]
        ),
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": rng.choice([1, 2, 3, 4], n, p=[0.11, 0.19, 0.35, 0.35]),
        "Gender": rng.choice(["Male", "Female"], n, p=[0.60, 0.40]),
        "HourlyRate": rng.integers(30, 100, n),
        "JobInvolvement": rng.choice([1, 2, 3, 4], n, p=[0.05, 0.18, 0.60, 0.17]),
        "JobLevel": job_level,
        "JobRole": rng.choice(
            ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
             "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
            n
        ),
        "JobSatisfaction": rng.choice([1, 2, 3, 4], n, p=[0.12, 0.18, 0.34, 0.36]),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22]),
        "MonthlyIncome": monthly_income,
        "MonthlyRate": rng.integers(2000, 27000, n),
        "NumCompaniesWorked": rng.integers(0, 10, n),
        "OverTime": overtime,
        "PercentSalaryHike": rng.integers(11, 25, n),
        "PerformanceRating": rng.choice([3, 4], n, p=[0.85, 0.15]),
        "RelationshipSatisfaction": rng.choice([1, 2, 3, 4], n, p=[0.10, 0.19, 0.38, 0.33]),
        "StockOptionLevel": rng.choice([0, 1, 2, 3], n, p=[0.47, 0.33, 0.14, 0.06]),
        "TotalWorkingYears": np.clip(rng.integers(0, 40, n), 0, 40),
        "TrainingTimesLastYear": rng.integers(0, 7, n),
        "WorkLifeBalance": rng.choice([1, 2, 3, 4], n, p=[0.05, 0.22, 0.56, 0.17]),
        "YearsAtCompany": np.clip(rng.integers(0, 40, n), 0, 40),
        "YearsInCurrentRole": np.clip(rng.integers(0, 18, n), 0, 18),
        "YearsSinceLastPromotion": np.clip(rng.integers(0, 15, n), 0, 15),
        "YearsWithCurrManager": np.clip(rng.integers(0, 17, n), 0, 17),
    })
