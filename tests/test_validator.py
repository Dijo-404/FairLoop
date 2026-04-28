"""
FairLoop Test Suite — Validator Agent Tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from core.config import PipelineConfig
from agents.validator_agent import NeutralValidatorAgent


@pytest.fixture
def config():
    return PipelineConfig(
        protected_attributes=["sex"],
        privileged_groups={"sex": "Male"},
        target_column="income",
        favorable_label=">50K",
        enable_semantic_layer=False,  # Disable Gemini for tests
    )


@pytest.fixture
def biased_batch():
    np.random.seed(42)
    n = 200
    sex = np.random.choice(["Male", "Female"], n, p=[0.7, 0.3])
    income = []
    for s in sex:
        if s == "Male":
            income.append(">50K" if np.random.random() < 0.35 else "<=50K")
        else:
            income.append(">50K" if np.random.random() < 0.08 else "<=50K")
    df = pd.DataFrame({"sex": sex, "age": np.random.randint(20, 70, n), "income": income})
    return {"batch_id": "test_biased_001", "data": df, "sample_count": n}


@pytest.fixture
def fair_batch():
    np.random.seed(42)
    n = 200
    sex = np.random.choice(["Male", "Female"], n, p=[0.5, 0.5])
    income = np.random.choice([">50K", "<=50K"], n, p=[0.25, 0.75])
    df = pd.DataFrame({"sex": sex, "age": np.random.randint(20, 70, n), "income": income})
    return {"batch_id": "test_fair_001", "data": df, "sample_count": n}


class TestValidatorAgent:
    def test_flags_biased_batch(self, config, biased_batch):
        validator = NeutralValidatorAgent(config)
        result = validator.validate_batch(biased_batch)
        assert result.verdict in ["REMEDIATE", "REJECT"]
        assert len(result.fairness_report.failed_metrics) > 0

    def test_approves_fair_batch(self, config, fair_batch):
        validator = NeutralValidatorAgent(config)
        result = validator.validate_batch(fair_batch)
        # Fair data should likely be approved or at worst remediated
        assert result.verdict in ["APPROVE", "REMEDIATE"]

    def test_verdict_has_reason(self, config, biased_batch):
        validator = NeutralValidatorAgent(config)
        result = validator.validate_batch(biased_batch)
        assert len(result.reason) > 0
        assert result.confidence > 0

    def test_metrics_reported(self, config, biased_batch):
        validator = NeutralValidatorAgent(config)
        result = validator.validate_batch(biased_batch)
        metrics = result.fairness_report.metrics
        assert "disparate_impact_ratio" in metrics
        assert "demographic_parity_diff" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
