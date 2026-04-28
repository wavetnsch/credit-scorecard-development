import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


class CreditScorecard:
    """
    Logistic regression credit scorecard with PDO-based score scaling.

    Score = Offset + Factor × ln(odds_of_non_default)

    Where:
        Factor = PDO / ln(2)
        Offset = Base Score − Factor × ln(Base Odds)
        PDO    = Points to Double the Odds (industry standard: 20)
    """

    def __init__(
        self,
        base_score: int = 600,
        base_odds: float = 50.0,
        pdo: int = 20,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self._factor = pdo / np.log(2)
        self._offset = base_score - self._factor * np.log(base_odds)

        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self.scaler = StandardScaler()
        self.features_: List[str] = []
        self.score_points_: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features_ = X.columns.tolist()
        X_sc = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_sc, y)
        self._build_score_card(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self.scaler.transform(X[self.features_].fillna(0))
        return self.model.predict_proba(X_sc)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Map PD to a credit score (higher score = lower risk)."""
        pd_hat = self.predict_proba(X)[:, 1].clip(1e-9, 1 - 1e-9)
        log_odds = np.log((1 - pd_hat) / pd_hat)
        return np.round(self._offset + self._factor * log_odds).astype(int)

    def score_summary(self) -> pd.DataFrame:
        rows = [
            {
                "feature": feat,
                "coefficient": info["coef"],
                "points_per_unit_woe": info["pts_per_woe"],
            }
            for feat, info in self.score_points_.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("pts_per_woe", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    def _build_score_card(self, X: pd.DataFrame):
        coefs = self.model.coef_[0]
        for feat, coef, scale in zip(self.features_, coefs, self.scaler.scale_):
            self.score_points_[feat] = {
                "coef": coef,
                "pts_per_woe": -self._factor * coef / scale,
            }
