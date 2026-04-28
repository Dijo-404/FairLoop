"""
FairLoop Remediation Agent
Applies a hierarchy of repair strategies to biased batches.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.fairness_metrics import compute_reweighing_weights


class RemediationAgent:
    """
    Receives batches flagged by the Validator and applies repair strategies:
    1. Re-weighting (Kamiran & Calders 2012)
    2. Disparate Impact Removal
    3. Counterfactual Augmentation
    4. Synthetic Infill
    5. Hard Drop (if all fail)

    Max 2 remediation cycles per batch.
    """

    def __init__(self, config=None, synth_agent=None):
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.synth_agent = synth_agent
        self.remediation_history: List[Dict] = []

    def remediate(
        self,
        batch: Dict,
        failed_metrics: List[str],
        protected_attr: str,
        cycle: int = 0,
    ) -> Tuple[Dict, str]:
        """
        Apply remediation strategies to a biased batch.

        Args:
            batch: The batch dict with "data" DataFrame
            failed_metrics: List of metric names that failed
            protected_attr: The protected attribute to remediate for
            cycle: Current remediation cycle (max 2)

        Returns:
            (remediated_batch, strategy_applied)
        """
        df = batch["data"].copy()
        strategies_applied = []

        print(f"[Remediation] Cycle {cycle + 1} for batch {batch['batch_id']}")
        print(f"[Remediation] Failed metrics: {failed_metrics}")

        # Strategy 1: Re-weighting
        if any(m in failed_metrics for m in [
            "disparate_impact_ratio", "demographic_parity_diff"
        ]):
            df, weights = self._apply_reweighting(df, protected_attr)
            batch["sample_weights"] = weights
            strategies_applied.append("reweighting")
            print(f"[Remediation] Applied reweighting")

        # Strategy 2: Disparate Impact Removal (repair feature distributions)
        if "disparate_impact_ratio" in failed_metrics:
            df = self._apply_disparate_impact_removal(df, protected_attr)
            strategies_applied.append("disparate_impact_removal")
            print(f"[Remediation] Applied disparate impact removal")

        # Strategy 3: Counterfactual Augmentation
        if any(m in failed_metrics for m in [
            "equal_opportunity_diff", "predictive_parity_diff"
        ]):
            if self.synth_agent is not None:
                counterfactuals = self.synth_agent.generate_counterfactuals(
                    df, protected_attr
                )
                if len(counterfactuals) > 0:
                    # Add a fraction of counterfactuals
                    n_add = min(len(counterfactuals), len(df) // 4)
                    df = pd.concat(
                        [df, counterfactuals.head(n_add)], ignore_index=True
                    )
                    strategies_applied.append("counterfactual_augmentation")
                    print(f"[Remediation] Added {n_add} counterfactual samples")

        # Strategy 4: Synthetic Infill (for representation balance)
        if "representation_balance" in failed_metrics:
            if self.synth_agent is not None and self.synth_agent.fitted:
                df = self.synth_agent.fill_representation_gap(
                    df, protected_attr, target_balance=0.80
                )
                strategies_applied.append("synthetic_infill")
                print(f"[Remediation] Applied synthetic infill")

        # Strategy 5: If nothing worked or no strategies matched, try balanced resampling
        if not strategies_applied:
            df = self._balanced_resample(df, protected_attr)
            strategies_applied.append("balanced_resampling")
            print(f"[Remediation] Applied balanced resampling")

        # Update batch
        batch["data"] = df
        batch["sample_count"] = len(df)

        strategy_str = " + ".join(strategies_applied)

        self.remediation_history.append({
            "batch_id": batch["batch_id"],
            "cycle": cycle,
            "strategies": strategies_applied,
            "original_size": batch.get("sample_count", len(df)),
            "remediated_size": len(df),
        })

        return batch, strategy_str

    def _apply_reweighting(
        self, df: pd.DataFrame, protected_attr: str
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply Kamiran & Calders reweighing to equalize outcome rates across groups."""
        target_col = self.config.target_column
        favorable = self.config.favorable_label

        y = (df[target_col].astype(str).str.strip() == str(favorable).strip()).astype(int).values
        priv_val = self.config.privileged_groups.get(protected_attr, df[protected_attr].value_counts().index[0])
        s = (df[protected_attr].astype(str).str.strip() == str(priv_val).strip()).astype(int).values

        weights = compute_reweighing_weights(y, s)

        # Use weights to oversample underrepresented positive outcomes
        # Normalize weights
        weights = weights / weights.mean()

        # Probabilistic resampling based on weights
        probs = weights / weights.sum()
        n_resample = len(df)
        indices = np.random.choice(len(df), size=n_resample, p=probs, replace=True)
        df_reweighted = df.iloc[indices].reset_index(drop=True)

        return df_reweighted, weights

    def _apply_disparate_impact_removal(
        self, df: pd.DataFrame, protected_attr: str
    ) -> pd.DataFrame:
        """
        Repair feature distributions to reduce disparate impact.
        Simplified version of AIF360's DisparateImpactRemover.
        Adjusts numeric features toward group-conditional medians.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = self.config.target_column

        for col in numeric_cols:
            if col == target_col:
                continue

            try:
                # Compute per-group medians
                group_medians = df.groupby(protected_attr)[col].median()
                overall_median = df[col].median()

                # Shift each group's values toward the overall median
                for group_val in df[protected_attr].unique():
                    mask = df[protected_attr] == group_val
                    group_median = group_medians.get(group_val, overall_median)
                    shift = overall_median - group_median

                    # Apply partial repair (repair_level = 0.8)
                    repair_level = 0.8
                    df.loc[mask, col] = df.loc[mask, col] + (shift * repair_level)
            except (ValueError, TypeError):
                continue

        return df

    def _balanced_resample(
        self, df: pd.DataFrame, protected_attr: str
    ) -> pd.DataFrame:
        """
        Balance representation by oversampling minority groups.
        Simple but effective fallback strategy.
        """
        groups = df.groupby(protected_attr)
        max_count = groups.size().max()

        balanced_parts = []
        for group_val, group_df in groups:
            if len(group_df) < max_count:
                # Oversample
                oversampled = group_df.sample(
                    n=max_count, replace=True, random_state=42
                )
                balanced_parts.append(oversampled)
            else:
                balanced_parts.append(group_df)

        return pd.concat(balanced_parts, ignore_index=True)

    def should_hard_drop(self, cycle: int) -> bool:
        """Check if we should drop the batch (exceeded max remediation cycles)."""
        return cycle >= self.config.max_remediation_cycles
