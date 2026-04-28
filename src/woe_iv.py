import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class WoEBinner:
    """
    Weight of Evidence (WoE) transformer for credit scorecard development.

    WoE_i  = ln( Dist_Events_i / Dist_Non_Events_i )
    IV     = Σ  (Dist_Events_i − Dist_Non_Events_i) × WoE_i

    IV interpretation (Siddiqi 2006):
        < 0.02  : Useless predictor
        0.02–0.1: Weak predictor
        0.1–0.3 : Medium predictor
        0.3–0.5 : Strong predictor
        > 0.5   : Suspicious (possible data leakage)
    """

    _IV_LABELS = [(0.0, 0.02, "Useless"), (0.02, 0.1, "Weak"),
                  (0.1, 0.3, "Medium"), (0.3, 0.5, "Strong"), (0.5, 9e9, "Suspicious")]

    def __init__(self, max_bins: int = 10, min_bin_pct: float = 0.05):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.woe_tables_: Dict[str, pd.DataFrame] = {}
        self.iv_values_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, features: Optional[List[str]] = None):
        features = features or X.columns.tolist()
        for feat in features:
            table, iv = self._compute(X[feat], y)
            self.woe_tables_[feat] = table
            self.iv_values_[feat] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for feat, table in self.woe_tables_.items():
            if feat in out.columns:
                out[f"{feat}_woe"] = out[feat].apply(lambda v: self._lookup_woe(v, table))
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, features: Optional[List[str]] = None):
        return self.fit(X, y, features).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        rows = [
            {"feature": f, "iv": iv, "interpretation": self._interpret(iv)}
            for f, iv in self.iv_values_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def plot_woe(self, feature: str, figsize: Tuple = (12, 4)):
        if feature not in self.woe_tables_:
            raise ValueError(f"Feature '{feature}' not found. Run fit() first.")

        table = self.woe_tables_[feature]
        iv = self.iv_values_[feature]
        labels = [str(b)[:20] for b in table["bin"]]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        colors = ["#e74c3c" if w < 0 else "#2980b9" for w in table["woe"]]
        axes[0].bar(range(len(table)), table["woe"], color=colors, alpha=0.85)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_xticks(range(len(table)))
        axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axes[0].set_title(f"{feature} — WoE per Bin  (IV = {iv:.4f} · {self._interpret(iv)})")
        axes[0].set_ylabel("Weight of Evidence")

        axes[1].bar(range(len(table)), table["event_rate"] * 100, color="#e67e22", alpha=0.85)
        axes[1].set_xticks(range(len(table)))
        axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axes[1].set_title(f"{feature} — Default Rate (%) per Bin")
        axes[1].set_ylabel("Default Rate (%)")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self, series: pd.Series, target: pd.Series) -> Tuple[pd.DataFrame, float]:
        df = pd.DataFrame({"x": series, "y": target}).dropna()
        total_e = df["y"].sum()
        total_ne = len(df) - total_e
        if total_e == 0 or total_ne == 0:
            return pd.DataFrame(), 0.0

        if df["x"].dtype == object or df["x"].nunique() <= 10:
            df["bin"] = df["x"].astype(str)
        else:
            try:
                df["bin"] = pd.qcut(df["x"], q=self.max_bins, duplicates="drop").astype(str)
            except ValueError:
                df["bin"] = pd.cut(df["x"], bins=self.max_bins).astype(str)

        tbl = (
            df.groupby("bin")["y"]
            .agg(count="count", events="sum")
            .assign(non_events=lambda x: x["count"] - x["events"])
            .assign(
                dist_e=lambda x: (x["events"] / total_e).clip(lower=1e-10),
                dist_ne=lambda x: (x["non_events"] / total_ne).clip(lower=1e-10),
            )
        )
        tbl["woe"] = np.log(tbl["dist_e"] / tbl["dist_ne"])
        tbl["iv_contrib"] = (tbl["dist_e"] - tbl["dist_ne"]) * tbl["woe"]
        tbl["event_rate"] = tbl["events"] / tbl["count"]

        return tbl.reset_index(), tbl["iv_contrib"].sum()

    def _lookup_woe(self, value, table: pd.DataFrame) -> float:
        if pd.isna(value):
            return 0.0
        for _, row in table.iterrows():
            if str(value) == row["bin"] or self._in_interval(value, row["bin"]):
                return row["woe"]
        return 0.0

    @staticmethod
    def _in_interval(value, bin_str: str) -> bool:
        m = re.match(r"[\(\[](.*),\s*(.*)[)\]]", str(bin_str))
        if not m:
            return False
        try:
            lo, hi = float(m.group(1)), float(m.group(2))
            return lo <= value <= hi
        except ValueError:
            return False

    def _interpret(self, iv: float) -> str:
        for lo, hi, label in self._IV_LABELS:
            if lo <= iv < hi:
                return label
        return "Unknown"
