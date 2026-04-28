import numpy as np
import pandas as pd
from typing import Tuple


def generate_credit_data(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic credit application data for scorecard development."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 76, n_samples)
    annual_income = rng.lognormal(np.log(480_000), 0.7, n_samples).clip(60_000, 5_000_000)
    employment_years = rng.exponential(6, n_samples).clip(0, 40)
    loan_amount = rng.lognormal(np.log(300_000), 0.9, n_samples).clip(20_000, 5_000_000)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples, p=[0.10, 0.20, 0.35, 0.20, 0.15])
    credit_utilization = rng.beta(2, 5, n_samples)
    num_delinquencies = rng.poisson(0.4, n_samples)
    num_credit_accounts = rng.integers(1, 21, n_samples)
    debt_to_income = (loan_amount / annual_income).clip(0, 5)

    log_odds = (
        -4.0
        + 0.015 * (38 - age)
        - 0.6 * np.log((annual_income / 480_000).clip(1e-6))
        - 0.08 * employment_years
        + 1.2 * debt_to_income
        + 2.0 * credit_utilization
        + 0.9 * num_delinquencies
        + 0.02 * loan_term / 12
        - 0.05 * num_credit_accounts
    )
    prob_default = (1 / (1 + np.exp(-log_odds)) + rng.normal(0, 0.02, n_samples)).clip(0.001, 0.999)
    default = (rng.uniform(0, 1, n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.astype(int),
        "employment_years": employment_years.round(1),
        "loan_amount": loan_amount.astype(int),
        "loan_term": loan_term,
        "credit_utilization": credit_utilization.round(4),
        "num_delinquencies": num_delinquencies,
        "num_credit_accounts": num_credit_accounts,
        "debt_to_income": debt_to_income.round(4),
        "default": default,
    })

    # Simulate realistic missing values (~3%)
    for col in ["employment_years", "annual_income"]:
        idx = rng.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        df.loc[idx, col] = np.nan

    return df


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split preserving temporal order to avoid data leakage."""
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
