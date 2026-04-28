"""
FairLoop Neutral Validator Agent ⭐ Core Innovation
An independent evaluator with NO shared weights with the Learner.
Runs two layers: statistical (fast) + semantic (LLM-powered, on flagged batches).
"""
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from core.fairness_metrics import FairnessMetricsEngine, FairnessReport


@dataclass
class ValidationVerdict:
    """Complete validation result for a batch."""
    batch_id: str
    verdict: str                    # "APPROVE", "REMEDIATE", "REJECT"
    fairness_report: FairnessReport
    semantic_bias_score: Optional[float]
    semantic_issues: List[str]
    reason: str
    confidence: float               # 0-1


class NeutralValidatorAgent:
    """
    The fairness firewall. Structurally independent from the Learner.

    Layer 1 — Statistical (runs on every batch):
        7 fairness metrics computed locally.

    Layer 2 — Semantic (runs on flagged batches, via Gemini API):
        Detects stereotyped language, coded bias, harmful associations.
    """

    def __init__(self, config=None):
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.metrics_engine = FairnessMetricsEngine(self.config.thresholds)
        self.gemini_model = None
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini API for semantic analysis."""
        api_key = self.config.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key and self.config.enable_semantic_layer:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
                print("[Validator] Gemini semantic layer initialized")
            except (ImportError, Exception) as e:
                print(f"[Validator] Gemini not available: {e}. Semantic layer disabled.")
                self.gemini_model = None
        else:
            print("[Validator] Semantic layer disabled (no API key or disabled in config)")

    def validate_batch(
        self,
        batch: Dict,
        predicted_col: Optional[str] = None,
    ) -> ValidationVerdict:
        """
        Run full validation on a batch.
        Returns a verdict: APPROVE, REMEDIATE, or REJECT.
        """
        df = batch["data"]
        batch_id = batch["batch_id"]

        # === LAYER 1: Statistical Analysis (always runs) ===
        fairness_reports = []
        for attr in self.config.protected_attributes:
            if attr not in df.columns:
                continue

            priv_val = self.config.privileged_groups.get(attr)
            if priv_val is None:
                # Auto-detect majority group
                priv_val = df[attr].value_counts().index[0]

            report = self.metrics_engine.compute_all(
                df=df,
                target_col=self.config.target_column,
                protected_attr=attr,
                privileged_value=priv_val,
                favorable_label=self.config.favorable_label,
                batch_id=batch_id,
                predicted_col=predicted_col,
            )
            fairness_reports.append(report)

        # Aggregate across protected attributes
        all_pass = all(r.passes_thresholds for r in fairness_reports)
        all_failed = []
        all_proxies = []
        agg_metrics = {}

        for report in fairness_reports:
            all_failed.extend(report.failed_metrics)
            all_proxies.extend(report.flagged_proxies)
            for k, v in report.metrics.items():
                key = f"{report.protected_attribute}_{k}"
                agg_metrics[key] = v

        primary_report = fairness_reports[0] if fairness_reports else FairnessReport(
            batch_id=batch_id, sample_count=len(df),
            protected_attribute="none", metrics={},
            flagged_proxies=[], passes_thresholds=True,
            failed_metrics=[], details={}
        )

        # === LAYER 2: Semantic Analysis (only on flagged batches) ===
        semantic_score = None
        semantic_issues = []

        if not all_pass and self.gemini_model is not None:
            semantic_score, semantic_issues = self._semantic_analysis(df, batch_id)
        elif not all_pass:
            # Mock semantic score when Gemini is not available
            semantic_score = self._mock_semantic_score(df, primary_report)
            if semantic_score > 0.2:
                semantic_issues.append("Elevated bias indicators detected via statistical proxy")

        # === VERDICT LOGIC ===
        verdict, reason, confidence = self._decide_verdict(
            all_pass, all_failed, all_proxies, semantic_score, agg_metrics
        )

        return ValidationVerdict(
            batch_id=batch_id,
            verdict=verdict,
            fairness_report=primary_report,
            semantic_bias_score=semantic_score,
            semantic_issues=semantic_issues,
            reason=reason,
            confidence=confidence,
        )

    def _decide_verdict(
        self,
        statistical_pass: bool,
        failed_metrics: List[str],
        proxies: List[str],
        semantic_score: Optional[float],
        metrics: Dict[str, float],
    ) -> Tuple[str, str, float]:
        """Apply verdict logic based on both statistical and semantic layers."""
        thresholds = self.config.thresholds

        # APPROVE: all statistical pass AND semantic clean
        if statistical_pass and (semantic_score is None or semantic_score < thresholds.semantic_bias_approve):
            return (
                "APPROVE",
                "All fairness metrics within acceptable thresholds.",
                0.95,
            )

        # REJECT: severe statistical failures OR high semantic bias
        severe_failures = [
            m for m in failed_metrics
            if m in ["disparate_impact_ratio", "demographic_parity_diff"]
        ]

        if semantic_score is not None and semantic_score >= thresholds.semantic_bias_remediate:
            return (
                "REJECT",
                f"Severe bias detected. Semantic score: {semantic_score:.2f}. "
                f"Failed metrics: {', '.join(failed_metrics)}. "
                f"Batch is unrecoverable and will be dropped.",
                0.85,
            )

        # Check for extremely biased data (near-zero representation)
        # Only REJECT when data is truly unrecoverable (one group has ~0 positive outcomes)
        for key, val in metrics.items():
            if "disparate_impact_ratio" in key and val < 0.05:
                return (
                    "REJECT",
                    f"Near-zero disparate impact: {val:.3f}. "
                    f"One group has virtually no positive outcomes — unrecoverable.",
                    0.90,
                )

        # REMEDIATE: fixable issues
        reason_parts = []
        if failed_metrics:
            reason_parts.append(f"Failed metrics: {', '.join(set(failed_metrics))}")
        if proxies:
            reason_parts.append(f"Proxy variables detected: {', '.join(set(proxies))}")
        if semantic_score is not None:
            reason_parts.append(f"Semantic bias score: {semantic_score:.2f}")

        # Include specific metric values
        for key, val in metrics.items():
            if "disparate_impact_ratio" in key:
                reason_parts.append(f"{key}={val:.3f} (threshold: ≥0.80)")
            elif "demographic_parity_diff" in key:
                reason_parts.append(f"{key}={val:.3f} (threshold: ≤0.10)")

        return (
            "REMEDIATE",
            "Batch has fixable bias issues. " + ". ".join(reason_parts),
            0.75,
        )

    def _semantic_analysis(
        self, df: pd.DataFrame, batch_id: str
    ) -> Tuple[float, List[str]]:
        """Run Gemini-powered semantic bias analysis on text fields."""
        # Sample text data from the batch
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not text_cols:
            return 0.0, []

        # Create a representative sample
        sample_size = min(10, len(df))
        sample = df.sample(sample_size).to_dict(orient='records')

        prompt = f"""You are a fairness auditor. Analyze the following training data sample for:
1. Stereotyped or biased language targeting protected groups
2. Coded language that correlates with demographic attributes
3. Harmful associations between group identity and negative outcomes

Return ONLY valid JSON:
{{
  "semantic_bias_score": 0.0-1.0,
  "detected_issues": ["issue1", "issue2"],
  "flagged_phrases": ["phrase1"],
  "recommendation": "APPROVE | REMEDIATE | REJECT",
  "reason": "plain English explanation"
}}

Data sample (batch {batch_id}):
{json.dumps(sample[:5], indent=2, default=str)}
"""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0, "response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            score = float(result.get("semantic_bias_score", 0.0))
            issues = result.get("detected_issues", [])
            return score, issues
        except Exception as e:
            print(f"[Validator] Semantic analysis failed: {e}")
            return 0.0, []

    def _mock_semantic_score(self, df: pd.DataFrame, report: FairnessReport) -> float:
        """
        When Gemini is unavailable, estimate semantic bias from statistical metrics.
        This is a rough heuristic, not a replacement for LLM analysis.
        """
        score = 0.0

        # Base score from disparate impact
        di = report.metrics.get("disparate_impact_ratio", 1.0)
        if di < 0.8:
            score += (0.8 - di) * 0.5  # Maps 0.8→0, 0.3→0.25

        # Bonus from demographic parity
        dpd = abs(report.metrics.get("demographic_parity_diff", 0.0))
        score += dpd * 0.3

        # Proxy penalty
        if report.flagged_proxies:
            score += 0.1 * len(report.flagged_proxies)

        return min(score, 1.0)
