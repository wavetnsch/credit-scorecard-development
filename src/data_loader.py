import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

DATA_PATH = Path(__file__).parent.parent / "data" / "cs-training.csv"


def load_credit_data(path: str = None) -> pd.DataFrame:
    """Load and clean Give Me Some Credit dataset (Kaggle 2011 competition)."""
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path, index_col=0)

    df = df.rename(columns={
        "SeriousDlqin2yrs":                       "default",
        "RevolvingUtilizationOfUnsecuredLines":    "credit_utilization",
        "NumberOfOpenCreditLinesAndLoans":         "num_credit_accounts",
        "DebtRatio":                               "debt_to_income",
        "MonthlyIncome":                           "annual_income",
        "NumberOfDependents":                      "num_dependents",
        "NumberRealEstateLoansOrLines":            "num_real_estate_loans",
        "NumberOfTime30-59DaysPastDueNotWorse":    "num_30_59_dpd",
        "NumberOfTime60-89DaysPastDueNotWorse":    "num_60_89_dpd",
        "NumberOfTimes90DaysLate":                 "num_90day_late",
    })

    # Convert monthly → annual income
    df["annual_income"] = df["annual_income"] * 12

    # Aggregate all delinquency counts into one feature
    df["num_delinquencies"] = (
        df["num_30_59_dpd"] + df["num_60_89_dpd"] + df["num_90day_late"]
    ).clip(upper=10)

    # Cap outliers
    df["credit_utilization"] = df["credit_utilization"].clip(0, 1)
    df["debt_to_income"] = df["debt_to_income"].clip(0, 10)
    df["annual_income"] = df["annual_income"].clip(upper=12_000_000)

    df = df.drop(columns=["num_30_59_dpd", "num_60_89_dpd", "num_90day_late"])

    return df


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split preserving row order (mirrors original train_test_split_time API)."""
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
