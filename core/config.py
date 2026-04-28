"""
FairLoop Configuration Module
Central configuration for all agents, thresholds, and system parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FairnessThresholds:
    """Fairness metric thresholds — batch fails if any threshold is breached."""
    demographic_parity_diff: float = 0.10       # max acceptable gap
    disparate_impact_ratio: float = 0.80        # min acceptable ratio (80% rule)
    equal_opportunity_diff: float = 0.10        # max TPR gap
    predictive_parity_diff: float = 0.10        # max precision gap
    individual_fairness_score: float = 0.85     # min similarity-fairness score
    representation_balance: float = 0.75        # min minority representation ratio
    proxy_variable_penalty: float = 0.60        # correlation threshold for proxy detection
    semantic_bias_approve: float = 0.20         # below this → APPROVE
    semantic_bias_remediate: float = 0.50       # below this → REMEDIATE, above → REJECT


@dataclass
class PipelineConfig:
    """Configuration for the FairLoop training pipeline."""
    # Dataset
    dataset_name: str = "scikit-learn/adult-census-income"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    target_column: str = "income"
    protected_attributes: List[str] = field(default_factory=lambda: ["sex", "race"])
    favorable_label: str = ">50K"
    unfavorable_label: str = "<=50K"
    privileged_groups: Dict[str, str] = field(default_factory=lambda: {"sex": "Male"})

    # Pipeline
    batch_size: int = 256
    max_iterations: int = 20
    max_remediation_cycles: int = 2

    # Fairness
    thresholds: FairnessThresholds = field(default_factory=FairnessThresholds)

    # Gemini Validator (semantic layer)
    gemini_model: str = "gemini-1.5-pro"
    gemini_api_key: Optional[str] = None        # set via env GOOGLE_API_KEY
    enable_semantic_layer: bool = True

    # Learner
    learner_type: str = "sklearn"               # "sklearn" for demo, "llm" for production
    learner_model_name: str = "logistic_regression"

    # Synthetic Data
    synth_method: str = "ctgan"                 # "ctgan" or "gaussian_copula"
    synth_epochs: int = 100

    # Audit
    audit_db_path: str = "fairloop_audit.db"

    # Dashboard
    dashboard_port: int = 3000
    api_port: int = 8000


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
