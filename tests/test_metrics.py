"""
FairLoop Test Suite
Tests for the core fairness metrics engine and validator.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from core.fairness_metrics import FairnessMetricsEngine, compute_reweighing_weights
from core.config import FairnessThresholds


@pytest.fixture
def biased_data():
    """Create a biased dataset: males get positive outcomes 3x more than females."""
    np.random.seed(42)
    n = 500
    sex = np.random.choice(["Male", "Female"], n, p=[0.67, 0.33])
    # Biased: Males get >50K 30% of time, Females only 10%
    income = []
    for s in sex:
        if s == "Male":
            income.append(">50K" if np.random.random() < 0.30 else "<=50K")
        else:
            income.append(">50K" if np.random.random() < 0.10 else "<=50K")

    return pd.DataFrame({
        "sex": sex,
        "age": np.random.randint(20, 70, n),
        "hours_per_week": np.random.randint(20, 60, n),
        "income": income,
    })


@pytest.fixture
def fair_data():
    """Create a fair dataset: equal outcomes across groups."""
    np.random.seed(42)
    n = 500
    sex = np.random.choice(["Male", "Female"], n, p=[0.50, 0.50])
    income = np.random.choice([">50K", "<=50K"], n, p=[0.25, 0.75])
    return pd.DataFrame({
        "sex": sex,
        "age": np.random.randint(20, 70, n),
        "hours_per_week": np.random.randint(20, 60, n),
        "income": income,
    })


class TestFairnessMetrics:
    def test_biased_data_fails(self, biased_data):
        engine = FairnessMetricsEngine()
        report = engine.compute_all(
            biased_data, "income", "sex", "Male", ">50K", batch_id="test_biased"
        )
        assert not report.passes_thresholds
        assert "disparate_impact_ratio" in report.failed_metrics
        assert report.metrics["disparate_impact_ratio"] < 0.8

    def test_fair_data_passes(self, fair_data):
        engine = FairnessMetricsEngine()
        report = engine.compute_all(
            fair_data, "income", "sex", "Male", ">50K", batch_id="test_fair"
        )
        # Fair data should pass most metrics
        assert report.metrics["disparate_impact_ratio"] > 0.5
        assert abs(report.metrics["demographic_parity_diff"]) < 0.2

    def test_disparate_impact_ratio(self, biased_data):
        engine = FairnessMetricsEngine()
        y = (biased_data["income"] == ">50K").astype(int).values
        priv = (biased_data["sex"] == "Male").values
        unpriv = ~priv

        di = engine._disparate_impact_ratio(y, priv, unpriv)
        # Males ~30% positive, Females ~10% → DI ≈ 0.33
        assert 0.1 < di < 0.6
        assert di < 0.8  # Fails 80% rule

    def test_reweighing_weights(self, biased_data):
        y = (biased_data["income"] == ">50K").astype(int).values
        s = (biased_data["sex"] == "Male").astype(int).values

        weights = compute_reweighing_weights(y, s)
        assert len(weights) == len(y)
        assert weights.mean() > 0.5  # Weights should be positive
        # Unprivileged positive samples should get higher weight
        unpriv_pos = (s == 0) & (y == 1)
        priv_pos = (s == 1) & (y == 1)
        if unpriv_pos.sum() > 0 and priv_pos.sum() > 0:
            assert weights[unpriv_pos].mean() > weights[priv_pos].mean()

    def test_representation_balance(self, biased_data):
        engine = FairnessMetricsEngine()
        priv = (biased_data["sex"] == "Male").values
        unpriv = ~priv

        rb = engine._representation_balance(priv, unpriv)
        assert 0 < rb < 1  # Males are 67%, females 33%, so ratio ≈ 0.5


class TestReweighing:
    def test_weights_balance_outcome_rates(self, biased_data):
        y = (biased_data["income"] == ">50K").astype(int).values
        s = (biased_data["sex"] == "Male").astype(int).values
        weights = compute_reweighing_weights(y, s)

        # Weighted positive rates should be more equal
        priv = s == 1
        unpriv = s == 0

        orig_priv_rate = y[priv].mean()
        orig_unpriv_rate = y[unpriv].mean()
        orig_gap = abs(orig_priv_rate - orig_unpriv_rate)

        weighted_priv_rate = np.average(y[priv], weights=weights[priv])
        weighted_unpriv_rate = np.average(y[unpriv], weights=weights[unpriv])
        weighted_gap = abs(weighted_priv_rate - weighted_unpriv_rate)

        # Weighted gap should be smaller than original
        assert weighted_gap < orig_gap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
