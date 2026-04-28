"""
FairLoop Synthetic Data Agent
Generates realistic synthetic samples to fill representation gaps.
Uses lightweight statistical resampling for the demo, with SDV CTGAN for production.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class SyntheticDataAgent:
    """
    Generates synthetic data to fill representation gaps identified by the Validator.
    Demo mode uses statistical resampling; production mode uses SDV CTGAN.
    """

    def __init__(self, config=None, mode: str = "resampling"):
        """
        Args:
            mode: "resampling" for lightweight demo, "ctgan" for SDV-based generation
        """
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.mode = mode
        self.synthesizer = None
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit the synthetic data generator on the full training data."""
        self.training_data = df.copy()
        self.fitted = True

        if self.mode == "ctgan":
            try:
                from sdv.single_table import CTGANSynthesizer
                from sdv.metadata import SingleTableMetadata

                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df)
                self.synthesizer = CTGANSynthesizer(
                    metadata,
                    epochs=self.config.synth_epochs,
                    verbose=True,
                )
                self.synthesizer.fit(df)
                print("[SynthAgent] CTGAN fitted successfully")
            except ImportError:
                print("[SynthAgent] SDV not installed, falling back to resampling mode")
                self.mode = "resampling"

        print(f"[SynthAgent] Fitted on {len(df)} rows (mode={self.mode})")

    def generate(
        self,
        num_rows: int,
        conditions: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples, optionally conditioned on specific attribute values.

        Args:
            num_rows: Number of synthetic samples to generate
            conditions: Dict of {column: value} to condition generation on
                       e.g. {"sex": "Female", "race": "Black"}
        """
        if not self.fitted:
            raise ValueError("SynthAgent not fitted. Call fit() first.")

        if self.mode == "ctgan" and self.synthesizer is not None:
            return self._generate_ctgan(num_rows, conditions)
        else:
            return self._generate_resampling(num_rows, conditions)

    def _generate_resampling(
        self, num_rows: int, conditions: Optional[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Generate synthetic data via stratified resampling with noise injection.
        Lightweight, fast, good enough for demo.
        """
        df = self.training_data.copy()

        # Filter to matching conditions
        if conditions:
            mask = pd.Series(True, index=df.index)
            for col, val in conditions.items():
                if col in df.columns:
                    mask &= df[col].astype(str).str.strip() == str(val).strip()
            subset = df[mask]
            if len(subset) < 5:
                # Not enough matching samples, use full data but set conditions
                subset = df.copy()
        else:
            subset = df

        # Resample with replacement
        synthetic = subset.sample(n=num_rows, replace=True, random_state=None).reset_index(drop=True)

        # Add noise to numeric columns (±5% of std) to avoid exact duplicates
        for col in synthetic.select_dtypes(include=[np.number]).columns:
            std = synthetic[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.05, len(synthetic))
                synthetic[col] = synthetic[col] + noise
                # Round integer columns back
                if df[col].dtype in [np.int64, int]:
                    synthetic[col] = synthetic[col].round().astype(int)

        # Override conditions
        if conditions:
            for col, val in conditions.items():
                if col in synthetic.columns:
                    synthetic[col] = val

        # Tag as synthetic
        synthetic["_is_synthetic"] = True

        return synthetic

    def _generate_ctgan(
        self, num_rows: int, conditions: Optional[Dict[str, str]]
    ) -> pd.DataFrame:
        """Generate using SDV CTGAN synthesizer."""
        if conditions:
            from sdv.sampling import Condition
            cond = Condition(num_rows=num_rows, column_values=conditions)
            synthetic = self.synthesizer.sample_from_conditions([cond])
        else:
            synthetic = self.synthesizer.sample(num_rows=num_rows)

        synthetic["_is_synthetic"] = True
        return synthetic

    def fill_representation_gap(
        self,
        batch_df: pd.DataFrame,
        protected_attr: str,
        target_balance: float = 0.80,
    ) -> pd.DataFrame:
        """
        Analyze a batch for underrepresentation and generate synthetic samples
        to bring minority groups up to target_balance ratio.

        Returns the augmented batch.
        """
        if not self.fitted:
            raise ValueError("SynthAgent not fitted. Call fit() first.")

        value_counts = batch_df[protected_attr].value_counts()
        majority_count = value_counts.max()
        target_count = int(majority_count * target_balance)

        augmented = batch_df.copy()
        generated_count = 0

        for group_val, group_count in value_counts.items():
            if group_count < target_count:
                needed = target_count - group_count
                synthetic = self.generate(
                    num_rows=needed,
                    conditions={protected_attr: str(group_val)},
                )
                augmented = pd.concat([augmented, synthetic], ignore_index=True)
                generated_count += needed
                print(f"[SynthAgent] Generated {needed} synthetic samples for {protected_attr}={group_val}")

        if generated_count > 0:
            print(f"[SynthAgent] Total synthetic samples added: {generated_count}")

        return augmented

    def generate_counterfactuals(
        self,
        batch_df: pd.DataFrame,
        protected_attr: str,
    ) -> pd.DataFrame:
        """
        Generate counterfactual samples by flipping the protected attribute
        for each sample in the batch. Used for counterfactual augmentation.
        """
        counterfactuals = batch_df.copy()
        unique_vals = batch_df[protected_attr].unique()

        if len(unique_vals) != 2:
            print(f"[SynthAgent] Counterfactual generation requires binary attribute, got {len(unique_vals)} values")
            return pd.DataFrame()

        val_a, val_b = unique_vals[0], unique_vals[1]
        counterfactuals[protected_attr] = counterfactuals[protected_attr].apply(
            lambda x: val_b if x == val_a else val_a
        )
        counterfactuals["_is_synthetic"] = True
        counterfactuals["_is_counterfactual"] = True

        return counterfactuals
