"""
FairLoop Core Fairness Metrics Engine
Computes all 7 statistical fairness metrics for batch-level evaluation.
Uses numpy for fast computation — no external fairness library dependency for core metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FairnessReport:
    """Complete fairness evaluation of a data batch."""
    batch_id: str
    sample_count: int
    protected_attribute: str
    metrics: Dict[str, float]
    flagged_proxies: List[str]
    passes_thresholds: bool
    failed_metrics: List[str]
    details: Dict[str, any]


class FairnessMetricsEngine:
    """
    Computes 7 statistical fairness metrics on a data batch:
    1. Demographic Parity Difference (DPD)
    2. Disparate Impact Ratio (DIR)
    3. Equal Opportunity Difference (EOD)
    4. Predictive Parity Difference (PPD)
    5. Individual Fairness Score (IFS)
    6. Representation Balance (RB)
    7. Proxy Variable Penalty (PVP)
    """

    def __init__(self, thresholds=None):
        from core.config import FairnessThresholds
        self.thresholds = thresholds or FairnessThresholds()

    def compute_all(
        self,
        df: pd.DataFrame,
        target_col: str,
        protected_attr: str,
        privileged_value: str,
        favorable_label,
        batch_id: str = "unknown",
        predicted_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> FairnessReport:
        """
        Compute all 7 fairness metrics on a batch.

        Args:
            df: The data batch as a DataFrame
            target_col: Name of the label/outcome column
            protected_attr: Name of the protected attribute column (e.g. 'sex')
            privileged_value: Value indicating the privileged group (e.g. 'Male')
            favorable_label: The favorable outcome value (e.g. '>50K')
            batch_id: Unique identifier for this batch
            predicted_col: Optional column with model predictions (for EOD/PPD)
            feature_cols: Columns to check for proxy variable correlation
        """
        # Binary encode
        y = (df[target_col].astype(str).str.strip() == str(favorable_label).strip()).astype(int).values
        s = (df[protected_attr].astype(str).str.strip() == str(privileged_value).strip()).astype(int).values

        priv_mask = s == 1
        unpriv_mask = s == 0

        if predicted_col and predicted_col in df.columns:
            y_pred = (df[predicted_col].astype(str).str.strip() == str(favorable_label).strip()).astype(int).values
        else:
            y_pred = y  # use labels as proxy when no predictions available

        metrics = {}
        failed = []

        # 1. Demographic Parity Difference
        dpd = self._demographic_parity_diff(y, priv_mask, unpriv_mask)
        metrics["demographic_parity_diff"] = dpd
        if abs(dpd) > self.thresholds.demographic_parity_diff:
            failed.append("demographic_parity_diff")

        # 2. Disparate Impact Ratio
        di = self._disparate_impact_ratio(y, priv_mask, unpriv_mask)
        metrics["disparate_impact_ratio"] = di
        if di < self.thresholds.disparate_impact_ratio:
            failed.append("disparate_impact_ratio")

        # 3. Equal Opportunity Difference
        eod = self._equal_opportunity_diff(y, y_pred, priv_mask, unpriv_mask)
        metrics["equal_opportunity_diff"] = eod
        if abs(eod) > self.thresholds.equal_opportunity_diff:
            failed.append("equal_opportunity_diff")

        # 4. Predictive Parity Difference
        ppd = self._predictive_parity_diff(y, y_pred, priv_mask, unpriv_mask)
        metrics["predictive_parity_diff"] = ppd
        if abs(ppd) > self.thresholds.predictive_parity_diff:
            failed.append("predictive_parity_diff")

        # 5. Individual Fairness Score
        ifs = self._individual_fairness_score(df, y, protected_attr, feature_cols)
        metrics["individual_fairness_score"] = ifs
        if ifs < self.thresholds.individual_fairness_score:
            failed.append("individual_fairness_score")

        # 6. Representation Balance
        rb = self._representation_balance(priv_mask, unpriv_mask)
        metrics["representation_balance"] = rb
        if rb < self.thresholds.representation_balance:
            failed.append("representation_balance")

        # 7. Proxy Variable Penalty
        proxies = self._detect_proxy_variables(
            df, protected_attr, feature_cols,
            threshold=self.thresholds.proxy_variable_penalty
        )
        metrics["proxy_variable_count"] = len(proxies)
        if len(proxies) > 0:
            failed.append("proxy_variable_penalty")

        passes = len(failed) == 0

        return FairnessReport(
            batch_id=batch_id,
            sample_count=len(df),
            protected_attribute=protected_attr,
            metrics=metrics,
            flagged_proxies=proxies,
            passes_thresholds=passes,
            failed_metrics=failed,
            details={
                "privileged_count": int(priv_mask.sum()),
                "unprivileged_count": int(unpriv_mask.sum()),
                "privileged_positive_rate": float(y[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0,
                "unprivileged_positive_rate": float(y[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0,
            }
        )

    @staticmethod
    def _demographic_parity_diff(y, priv_mask, unpriv_mask) -> float:
        """SPD = P(ŷ=1|unprivileged) - P(ŷ=1|privileged). Ideal: 0."""
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 0.0
        p_priv = y[priv_mask].mean()
        p_unpriv = y[unpriv_mask].mean()
        return float(p_unpriv - p_priv)

    @staticmethod
    def _disparate_impact_ratio(y, priv_mask, unpriv_mask) -> float:
        """DI = P(ŷ=1|unprivileged) / P(ŷ=1|privileged). Ideal: 1.0. Biased if <0.8."""
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 1.0
        p_priv = y[priv_mask].mean()
        p_unpriv = y[unpriv_mask].mean()
        if p_priv == 0:
            return 1.0 if p_unpriv == 0 else 0.0
        return float(p_unpriv / (p_priv + 1e-10))

    @staticmethod
    def _equal_opportunity_diff(y_true, y_pred, priv_mask, unpriv_mask) -> float:
        """EOD = TPR_unprivileged - TPR_privileged. Ideal: 0."""
        def tpr(mask):
            positives = y_true[mask] == 1
            if positives.sum() == 0:
                return 0.0
            return float(y_pred[mask][positives].mean())

        return tpr(unpriv_mask) - tpr(priv_mask)

    @staticmethod
    def _predictive_parity_diff(y_true, y_pred, priv_mask, unpriv_mask) -> float:
        """PPD = Precision_unprivileged - Precision_privileged. Ideal: 0."""
        def precision(mask):
            predicted_pos = y_pred[mask] == 1
            if predicted_pos.sum() == 0:
                return 0.0
            return float(y_true[mask][predicted_pos].mean())

        return precision(unpriv_mask) - precision(priv_mask)

    @staticmethod
    def _individual_fairness_score(
        df: pd.DataFrame, y, protected_attr: str,
        feature_cols: Optional[List[str]] = None,
        sample_size: int = 200
    ) -> float:
        """
        Individual fairness: similar individuals should receive similar outcomes.
        Computes pairwise feature similarity vs outcome similarity on a sample.
        Returns score in [0, 1]. Higher = more individually fair.
        """
        if feature_cols is None:
            # Use numeric columns only, exclude protected attribute
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != protected_attr]

        if len(feature_cols) == 0:
            return 1.0  # No features to compare

        # Sample for computational efficiency
        n = min(sample_size, len(df))
        idx = np.random.choice(len(df), n, replace=False)

        # Normalize features
        X = df.iloc[idx][feature_cols].values.astype(float)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_norm = (X - X_mean) / X_std

        y_sample = y[idx]

        # Compute pairwise distances (feature space)
        from scipy.spatial.distance import pdist, squareform
        feat_dists = squareform(pdist(X_norm, 'euclidean'))

        # Outcome differences
        outcome_diffs = np.abs(y_sample[:, None] - y_sample[None, :]).astype(float)

        # For similar individuals (feature distance < median), outcome should be similar
        median_dist = np.median(feat_dists[feat_dists > 0]) if (feat_dists > 0).any() else 1.0
        similar_mask = (feat_dists < median_dist) & (feat_dists > 0)

        if similar_mask.sum() == 0:
            return 1.0

        # Fraction of similar-feature pairs with same outcome
        score = 1.0 - outcome_diffs[similar_mask].mean()
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _representation_balance(priv_mask, unpriv_mask) -> float:
        """
        Representation balance: ratio of minority group size to majority group size.
        Ideal: 1.0 (equal representation). Range: [0, 1].
        """
        n_priv = priv_mask.sum()
        n_unpriv = unpriv_mask.sum()
        if n_priv == 0 and n_unpriv == 0:
            return 1.0
        minority = min(n_priv, n_unpriv)
        majority = max(n_priv, n_unpriv)
        return float(minority / (majority + 1e-10))

    @staticmethod
    def _detect_proxy_variables(
        df: pd.DataFrame,
        protected_attr: str,
        feature_cols: Optional[List[str]] = None,
        threshold: float = 0.60,
    ) -> List[str]:
        """
        Detect columns that are statistical proxies for the protected attribute.
        A proxy is any feature with |correlation| > threshold with the protected attr.
        """
        proxies = []
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != protected_attr]

        # Encode protected attribute as numeric
        s_encoded = pd.Categorical(df[protected_attr]).codes.astype(float)

        for col in feature_cols:
            if col == protected_attr:
                continue
            try:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    col_vals = df[col].astype(float)
                else:
                    col_vals = pd.Categorical(df[col]).codes.astype(float)

                corr = np.corrcoef(s_encoded, col_vals)[0, 1]
                if abs(corr) > threshold:
                    proxies.append(col)
            except (ValueError, TypeError):
                continue

        return proxies


def compute_reweighing_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """
    Kamiran & Calders (2012) Reweighing algorithm.
    Computes per-sample weights to equalize P(y|s) across groups.

    Args:
        y: Binary labels (0/1)
        sensitive: Binary sensitive attribute (0/1)

    Returns:
        Sample weights array
    """
    n = len(y)
    weights = np.ones(n)

    for y_val in [0, 1]:
        for s_val in np.unique(sensitive):
            p_y = np.mean(y == y_val)
            p_s = np.mean(sensitive == s_val)
            p_ys = np.mean((y == y_val) & (sensitive == s_val))

            mask = (y == y_val) & (sensitive == s_val)
            if p_ys > 0:
                weights[mask] = (p_y * p_s) / (p_ys + 1e-10)

    return weights
