"""
FairLoop Feature Agent
Transforms raw features for model ingestion and runs proxy variable detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureAgent:
    """
    Feature engineering + proxy variable detection.
    Transforms raw data into model-ready features while detecting
    columns that are statistical proxies for protected attributes.
    """

    def __init__(self, config=None):
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.fitted: bool = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        protected_attrs: List[str] = None,
        exclude_cols: List[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit encoders on data and transform features.
        Returns (transformed_df, feature_report).
        """
        target_col = target_col or self.config.target_column
        protected_attrs = protected_attrs or self.config.protected_attributes
        exclude_cols = exclude_cols or []

        df_out = df.copy()
        feature_report = {
            "original_columns": list(df.columns),
            "numeric_columns": [],
            "encoded_columns": [],
            "dropped_columns": [],
            "proxy_variables": [],
        }

        # Identify column types
        cat_cols = df_out.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

        # Remove columns to exclude
        for col in exclude_cols:
            if col in df_out.columns:
                df_out = df_out.drop(columns=[col])
                feature_report["dropped_columns"].append(col)

        # Encode categorical columns
        for col in cat_cols:
            if col in exclude_cols:
                continue
            if col not in self.label_encoders:
                le = LabelEncoder()
                # Handle unseen values by fitting on all unique values
                all_vals = df_out[col].astype(str).unique().tolist()
                le.fit(all_vals)
                self.label_encoders[col] = le

            le = self.label_encoders[col]
            # Handle unseen values at transform time
            known = set(le.classes_)
            df_out[col] = df_out[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
            feature_report["encoded_columns"].append(col)

        feature_report["numeric_columns"] = num_cols

        # Proxy variable detection
        for attr in protected_attrs:
            if attr in df_out.columns:
                proxies = self._detect_proxies(df_out, attr)
                for proxy_col, corr_val in proxies:
                    feature_report["proxy_variables"].append({
                        "column": proxy_col,
                        "protected_attribute": attr,
                        "correlation": round(corr_val, 4),
                    })

        self.fitted = True
        return df_out, feature_report

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders."""
        if not self.fitted:
            raise ValueError("FeatureAgent not fitted. Call fit_transform first.")

        df_out = df.copy()
        for col, le in self.label_encoders.items():
            if col in df_out.columns:
                known = set(le.classes_)
                df_out[col] = df_out[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
        return df_out

    def _detect_proxies(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        threshold: float = None,
    ) -> List[Tuple[str, float]]:
        """
        Detect columns correlated with a protected attribute above threshold.
        Returns list of (column_name, correlation_value) tuples.
        """
        threshold = threshold or self.config.thresholds.proxy_variable_penalty
        proxies = []

        if protected_attr not in df.columns:
            return proxies

        try:
            s_vals = df[protected_attr].astype(float)
        except (ValueError, TypeError):
            s_vals = pd.Categorical(df[protected_attr]).codes.astype(float)

        for col in df.columns:
            if col == protected_attr:
                continue
            try:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    col_vals = df[col].astype(float)
                else:
                    col_vals = pd.Categorical(df[col]).codes.astype(float)

                # Handle NaN
                valid = ~(np.isnan(col_vals) | np.isnan(s_vals))
                if valid.sum() < 10:
                    continue

                corr = np.corrcoef(s_vals[valid], col_vals[valid])[0, 1]
                if abs(corr) > threshold:
                    proxies.append((col, abs(corr)))
            except (ValueError, TypeError):
                continue

        return sorted(proxies, key=lambda x: x[1], reverse=True)

    def get_feature_importance_for_fairness(
        self, df: pd.DataFrame, target_col: str, protected_attr: str
    ) -> Dict[str, float]:
        """
        Compute how much each feature contributes to differential outcomes
        between privileged and unprivileged groups.
        """
        importance = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col in [target_col, protected_attr]:
                continue
            try:
                # Compute mean difference of feature between groups × correlation with outcome
                groups = df.groupby(protected_attr)[col].mean()
                if len(groups) >= 2:
                    group_diff = groups.max() - groups.min()
                    outcome_corr = abs(df[col].corr(df[target_col].astype(float)))
                    importance[col] = float(group_diff * outcome_corr)
            except (ValueError, TypeError):
                continue

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
