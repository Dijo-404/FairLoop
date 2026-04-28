"""
FairLoop LLM Learner Agent
Receives only Validator-approved batches and trains the model.
Demo mode: sklearn LogisticRegression (fast, visual).
Production mode: LoRA/QLoRA fine-tuning on Llama/Gemini.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class LearnerAgent:
    """
    The model being trained. Receives ONLY Validator-approved batches.
    Emits loss and fairness metrics after each update.
    """

    def __init__(self, config=None):
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()

        # Demo learner: Logistic Regression with warm_start for incremental learning
        self.model = LogisticRegression(
            max_iter=200,
            warm_start=True,   # allows incremental fitting
            C=1.0,
            solver='lbfgs',
            random_state=42,
        )
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.fitted = False
        self.training_history: List[Dict] = []
        self.iteration = 0

        # Test set for evaluation
        self.X_test = None
        self.y_test = None
        self.sensitive_test = None

    def set_test_data(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        sensitive_features: Dict[str, np.ndarray],
    ):
        """Set held-out test data for evaluation after each training update."""
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_test = sensitive_features

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to numeric features and labels."""
        target_col = self.config.target_column
        protected_attrs = self.config.protected_attributes

        # Separate target
        y = (df[target_col].astype(str).str.strip() == str(self.config.favorable_label).strip()).astype(int).values

        # Feature columns (exclude target and protected attributes)
        feature_cols = [c for c in df.columns
                       if c not in [target_col] + protected_attrs
                       and not c.startswith("_")]

        # Encode features
        X_parts = []
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                X_parts.append(df[col].values.astype(float).reshape(-1, 1))
            else:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str).unique())
                    self.label_encoders[col] = le
                le = self.label_encoders[col]
                known = set(le.classes_)
                encoded = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                ).values.reshape(-1, 1)
                X_parts.append(encoded.astype(float))

        if not X_parts:
            raise ValueError("No features to train on")

        X = np.hstack(X_parts)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        return X, y

    def train_on_batch(
        self,
        batch: Dict,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Train on a single Validator-approved batch.
        Returns training metrics.
        """
        df = batch["data"]
        X, y = self.prepare_features(df)

        # Scale features
        if not self.fitted:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Train
        if sample_weights is not None and len(sample_weights) == len(X):
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

        self.fitted = True
        self.iteration += 1

        # Compute training metrics
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred)
        train_loss = self._compute_log_loss(X, y)

        metrics = {
            "iteration": self.iteration,
            "batch_id": batch["batch_id"],
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
            "batch_size": len(df),
        }

        # Evaluate on test set if available
        if self.X_test is not None:
            eval_metrics = self.evaluate()
            metrics.update(eval_metrics)

        self.training_history.append(metrics)
        print(f"[Learner] Iteration {self.iteration}: acc={train_acc:.4f}, loss={train_loss:.4f}")

        return metrics

    def _compute_log_loss(self, X, y) -> float:
        """Compute log loss for the current batch."""
        try:
            probs = self.model.predict_proba(X)
            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(y * np.log(probs[:, 1]) + (1 - y) * np.log(probs[:, 0]))
            return float(loss)
        except Exception:
            return 0.0

    def evaluate(self) -> Dict:
        """Evaluate model on held-out test data."""
        if self.X_test is None:
            return {}

        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)

        eval_metrics = {
            "eval_accuracy": float(accuracy_score(self.y_test, y_pred)),
        }

        # Compute fairness metrics on test predictions
        if self.sensitive_test:
            from core.fairness_metrics import FairnessMetricsEngine
            engine = FairnessMetricsEngine(self.config.thresholds)

            for attr_name, attr_values in self.sensitive_test.items():
                priv_val = self.config.privileged_groups.get(attr_name)
                if priv_val is None:
                    continue

                priv_mask = attr_values == 1
                unpriv_mask = attr_values == 0

                di = engine._disparate_impact_ratio(y_pred, priv_mask, unpriv_mask)
                dpd = engine._demographic_parity_diff(y_pred, priv_mask, unpriv_mask)
                eod = engine._equal_opportunity_diff(self.y_test, y_pred, priv_mask, unpriv_mask)

                eval_metrics[f"eval_{attr_name}_disparate_impact"] = float(di)
                eval_metrics[f"eval_{attr_name}_demographic_parity_diff"] = float(dpd)
                eval_metrics[f"eval_{attr_name}_equal_opportunity_diff"] = float(eod)

        return eval_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.fitted:
            raise ValueError("Model not trained yet")
        X, _ = self.prepare_features(df)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def get_training_history(self) -> List[Dict]:
        """Return full training history for dashboard."""
        return self.training_history
