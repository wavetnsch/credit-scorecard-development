import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov statistic.
    KS = max | CDF_defaults(s) − CDF_non_defaults(s) |
    Returns (ks_value, score_at_ks).
    """
    df = (
        pd.DataFrame({"score": y_score, "y": y_true})
        .sort_values("score")
        .reset_index(drop=True)
    )
    n_e = y_true.sum()
    n_ne = len(y_true) - n_e

    df["cum_e"] = df["y"].cumsum() / n_e
    df["cum_ne"] = (1 - df["y"]).cumsum() / n_ne
    df["ks"] = (df["cum_e"] - df["cum_ne"]).abs()

    idx = df["ks"].idxmax()
    return df.loc[idx, "ks"], df.loc[idx, "score"]


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 × AUC − 1."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def population_stability_index(base: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI = Σ (A% − E%) × ln(A% / E%)

    Interpretation:
        PSI < 0.10 : Stable
        0.10–0.25  : Minor shift — monitor
        PSI > 0.25 : Major shift — investigate / redevelop
    """
    breakpoints = np.percentile(base, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    base_pct = np.histogram(base, bins=breakpoints)[0] / len(base)
    curr_pct = np.histogram(current, bins=breakpoints)[0] / len(current)

    base_pct = base_pct.clip(1e-10)
    curr_pct = curr_pct.clip(1e-10)

    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def validation_report(y_true: np.ndarray, y_score: np.ndarray, label: str = "Model") -> pd.DataFrame:
    auc = roc_auc_score(y_true, y_score)
    ks, ks_thr = ks_statistic(y_true, y_score)
    return pd.DataFrame([{
        "Dataset": label,
        "AUC": round(auc, 4),
        "Gini": round(2 * auc - 1, 4),
        "KS": round(ks, 4),
        "KS Score Threshold": round(ks_thr, 2),
        "Default Rate (%)": round(float(y_true.mean()) * 100, 2),
        "N": len(y_true),
    }])


def plot_roc_ks(y_true: np.ndarray, y_score: np.ndarray, title: str = "Model Validation", figsize=(14, 5)):
    """Side-by-side ROC curve and KS plot."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1

    axes[0].plot(fpr, tpr, color="#2980b9", lw=2,
                 label=f"AUC = {auc:.3f}  |  Gini = {gini:.3f}")
    axes[0].fill_between(fpr, tpr, alpha=0.10, color="#2980b9")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title}\nROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # --- KS ---
    df = pd.DataFrame({"score": y_score, "y": y_true}).sort_values("score")
    n_e, n_ne = y_true.sum(), len(y_true) - y_true.sum()
    df["cum_e"] = df["y"].cumsum() / n_e
    df["cum_ne"] = (1 - df["y"]).cumsum() / n_ne
    ks_val, _ = ks_statistic(y_true, y_score)

    axes[1].plot(df["score"], df["cum_e"], color="#e74c3c", lw=2, label="Cumulative Defaults")
    axes[1].plot(df["score"], df["cum_ne"], color="#2980b9", lw=2, label="Cumulative Non-Defaults")
    axes[1].set_xlabel("Predicted Score / Probability")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title(f"{title}\nKS Plot  (KS = {ks_val:.3f})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig
