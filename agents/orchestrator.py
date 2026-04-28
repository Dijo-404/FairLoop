"""
FairLoop Orchestrator Agent
The conductor of the entire system. Manages the training loop state machine
using LangGraph, spawns and coordinates all sub-agents.
"""
import json
import time
import uuid
from typing import Dict, List, Optional, TypedDict, Literal, Annotated
from dataclasses import dataclass, field, asdict

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.synth_agent import SyntheticDataAgent
from agents.validator_agent import NeutralValidatorAgent
from agents.remediation_agent import RemediationAgent
from agents.learner_agent import LearnerAgent
from core.audit_log import AuditLog
from core.config import PipelineConfig


# ===== LangGraph State =====
class FairLoopState(TypedDict):
    """Central state for the FairLoop pipeline."""
    iteration: int
    phase: str                      # current phase
    batch: Optional[Dict]           # current batch being processed
    batch_id: str
    verdict: str                    # APPROVE / REMEDIATE / REJECT
    remediation_cycle: int
    strategy_applied: str
    failed_metrics: List[str]
    metrics: Dict                   # current fairness metrics
    learner_metrics: Dict           # training metrics from learner
    global_metrics: List[Dict]      # metrics across all iterations
    audit_entries: List[Dict]       # audit log entries
    total_approved: int
    total_remediated: int
    total_rejected: int
    converged: bool
    error: Optional[str]


class Orchestrator:
    """
    Central orchestrator that manages the FairLoop pipeline.
    Provides both a LangGraph-based mode and a direct Python loop mode.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Initialize all agents
        self.data_agent = DataAgent(self.config)
        self.feature_agent = FeatureAgent(self.config)
        self.synth_agent = SyntheticDataAgent(self.config, mode="resampling")
        self.validator = NeutralValidatorAgent(self.config)
        self.remediation_agent = RemediationAgent(self.config, self.synth_agent)
        self.learner = LearnerAgent(self.config)
        self.audit_log = AuditLog(self.config.audit_db_path)

        # State
        self.state: Dict = {
            "iteration": 0,
            "global_metrics": [],
            "total_approved": 0,
            "total_remediated": 0,
            "total_rejected": 0,
        }

        # Event callbacks (for dashboard/API)
        self.event_callbacks: List[callable] = []

    def on_event(self, callback: callable):
        """Register a callback for pipeline events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict):
        """Emit an event to all registered callbacks."""
        event = {"type": event_type, "data": data, "timestamp": time.time()}
        for cb in self.event_callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[Orchestrator] Event callback error: {e}")

    def run(self, max_iterations: int = None) -> Dict:
        """
        Run the full FairLoop pipeline.
        Returns final metrics and audit summary.
        """
        max_iterations = max_iterations or self.config.max_iterations

        print("=" * 70)
        print("  FairLoop — In-Loop Bias Prevention System")
        print("=" * 70)

        # === Phase 1: Data Ingestion ===
        print("\n[Phase 1] Data Ingestion")
        df = self.data_agent.load_dataset()

        # === Phase 2: Feature Engineering & Proxy Detection ===
        print("\n[Phase 2] Feature Engineering")
        df_transformed, feature_report = self.feature_agent.fit_transform(
            df,
            exclude_cols=["fnlwgt", "education"],  # Remove redundant/noisy cols
        )
        if feature_report["proxy_variables"]:
            print(f"[Feature] Proxy variables detected: {feature_report['proxy_variables']}")

        # === Phase 3: Prepare test set ===
        print("\n[Phase 3] Preparing test set")
        test_frac = 0.2
        n_test = int(len(df) * test_frac)
        test_df = df.tail(n_test).copy()
        train_df = df.head(len(df) - n_test).copy()

        # Prepare test features for evaluation
        X_test, y_test = self.learner.prepare_features(test_df)
        sensitive_test = {}
        for attr in self.config.protected_attributes:
            priv_val = self.config.privileged_groups.get(attr)
            if priv_val and attr in test_df.columns:
                sensitive_test[attr] = (
                    test_df[attr].astype(str).str.strip() == str(priv_val).strip()
                ).astype(int).values

        self.learner.set_test_data(X_test, y_test, sensitive_test)

        # === Phase 4: Fit Synthetic Agent ===
        print("\n[Phase 4] Fitting Synthetic Data Generator")
        self.synth_agent.fit(train_df)

        # === Phase 5: Chunk into batches ===
        print("\n[Phase 5] Chunking data into batches")
        self.data_agent.raw_data = train_df
        batches = self.data_agent.chunk_into_batches()

        # === Phase 6: Training Loop ===
        print(f"\n[Phase 6] Starting FairLoop training ({max_iterations} iterations)")
        print("-" * 70)

        iterations_run = 0
        for i in range(min(max_iterations, len(batches))):
            iterations_run += 1
            self.state["iteration"] = i + 1
            batch = batches[i]

            print(f"\n--- Iteration {i + 1}/{max_iterations} | Batch: {batch['batch_id']} ---")

            # Validate batch
            verdict_result = self.validator.validate_batch(batch)
            verdict = verdict_result.verdict
            print(f"[Validator] Verdict: {verdict} | Reason: {verdict_result.reason}")

            # Process verdict
            strategy_applied = None
            remediation_cycle = 0

            if verdict == "APPROVE":
                self.state["total_approved"] += 1

            elif verdict == "REMEDIATE":
                # Remediation loop (max 2 cycles)
                current_batch = batch
                while verdict == "REMEDIATE" and remediation_cycle < self.config.max_remediation_cycles:
                    current_batch, strategy = self.remediation_agent.remediate(
                        current_batch,
                        verdict_result.fairness_report.failed_metrics,
                        self.config.protected_attributes[0],
                        cycle=remediation_cycle,
                    )
                    strategy_applied = strategy
                    remediation_cycle += 1

                    # Re-validate
                    verdict_result = self.validator.validate_batch(current_batch)
                    verdict = verdict_result.verdict
                    print(f"[Validator] Re-validation: {verdict} (cycle {remediation_cycle})")

                if verdict == "APPROVE":
                    batch = current_batch
                    self.state["total_approved"] += 1
                    self.state["total_remediated"] += 1
                elif verdict == "REJECT":
                    self.state["total_rejected"] += 1
                    print(f"[Remediation] Batch rejected after {remediation_cycle} cycles")
                else:
                    # Still REMEDIATE after max cycles — use remediated batch as-is
                    # (partially improved is better than fully biased)
                    batch = current_batch
                    self.state["total_approved"] += 1
                    self.state["total_remediated"] += 1
                    verdict = "APPROVE"
                    print(f"[Remediation] Force-approved after {remediation_cycle} remediation cycles (partially improved)")

            elif verdict == "REJECT":
                self.state["total_rejected"] += 1

            # Log to audit
            self.audit_log.log(
                iteration=i + 1,
                batch_id=batch["batch_id"],
                sample_count=batch["sample_count"],
                protected_attributes=self.config.protected_attributes,
                metrics=verdict_result.fairness_report.metrics,
                verdict=verdict,
                reason=verdict_result.reason,
                semantic_bias_score=verdict_result.semantic_bias_score,
                flagged_proxies=verdict_result.fairness_report.flagged_proxies,
                remediation_applied=strategy_applied,
                final_verdict=verdict,
            )

            # Train on approved batches
            if verdict == "APPROVE":
                sample_weights = batch.get("sample_weights")
                learner_metrics = self.learner.train_on_batch(batch, sample_weights)
                self.state["global_metrics"].append(learner_metrics)

                # Emit event
                self._emit_event("training_update", {
                    "iteration": i + 1,
                    "metrics": learner_metrics,
                    "verdict": verdict,
                })

            # Check convergence
            if self._check_convergence():
                print(f"\nConvergence reached at iteration {i + 1}!")
                break

        # === Phase 7: Final Summary ===
        print("\n" + "=" * 70)
        print("  FairLoop Training Complete")
        print("=" * 70)

        summary = self._generate_summary(iterations_run)
        self._print_summary(summary)

        return summary

    def _check_convergence(self) -> bool:
        """Check if fairness metrics have converged to acceptable levels."""
        if len(self.state["global_metrics"]) < 5:
            return False

        # Check last 3 iterations — all must have DI in [0.8, 1.2] range
        recent = self.state["global_metrics"][-3:]
        for m in recent:
            for attr in self.config.protected_attributes:
                di_key = f"eval_{attr}_disparate_impact"
                if di_key in m:
                    di = m[di_key]
                    if di < 0.80 or di > 1.20:
                        return False
        return True

    def _generate_summary(self, iterations: int) -> Dict:
        """Generate a complete summary of the FairLoop run."""
        audit_summary = self.audit_log.get_summary()
        training_history = self.learner.get_training_history()

        # Get final evaluation metrics
        final_eval = self.learner.evaluate() if self.learner.fitted else {}

        return {
            "iterations_completed": iterations,
            "total_approved": self.state["total_approved"],
            "total_remediated": self.state["total_remediated"],
            "total_rejected": self.state["total_rejected"],
            "final_evaluation": final_eval,
            "training_history": training_history,
            "audit_summary": audit_summary,
            "config": {
                "dataset": self.config.dataset_name,
                "protected_attributes": self.config.protected_attributes,
                "batch_size": self.config.batch_size,
                "max_iterations": self.config.max_iterations,
            },
        }

    def _print_summary(self, summary: Dict):
        """Pretty-print the run summary."""
        print(f"\nIterations: {summary['iterations_completed']}")
        print(f"Batches — Approved: {summary['total_approved']}, "
              f"Remediated: {summary['total_remediated']}, "
              f"Rejected: {summary['total_rejected']}")
        print(f"\nFinal Evaluation:")
        for key, val in summary["final_evaluation"].items():
            print(f"  {key}: {val:.4f}")

        # Show metrics trend
        if summary["training_history"]:
            first = summary["training_history"][0]
            last = summary["training_history"][-1]
            print(f"\nMetrics Trend (first → last):")
            print(f"  Accuracy: {first.get('eval_accuracy', 0):.4f} → {last.get('eval_accuracy', 0):.4f}")
            for attr in self.config.protected_attributes:
                di_key = f"eval_{attr}_disparate_impact"
                if di_key in first and di_key in last:
                    print(f"  {attr} DI: {first[di_key]:.4f} → {last[di_key]:.4f}")


def build_langgraph_pipeline(config: PipelineConfig = None):
    """
    Build the FairLoop pipeline as a LangGraph StateGraph.
    This provides the graph-based execution mode with visual debugging.
    """
    try:
        from langgraph.graph import StateGraph, END, START
    except ImportError:
        print("LangGraph not installed. Using direct orchestration mode.")
        return None

    config = config or PipelineConfig()
    orchestrator = Orchestrator(config)

    # Define node functions
    def ingest_node(state: FairLoopState) -> dict:
        """Data ingestion and preparation."""
        df = orchestrator.data_agent.load_dataset()
        orchestrator.synth_agent.fit(df)
        batches = orchestrator.data_agent.chunk_into_batches()
        return {"phase": "validate", "batch": batches[0] if batches else None}

    def validate_node(state: FairLoopState) -> dict:
        """Validate current batch."""
        batch = state.get("batch")
        if batch is None:
            return {"phase": "done", "converged": True}
        result = orchestrator.validator.validate_batch(batch)
        return {
            "verdict": result.verdict,
            "failed_metrics": result.fairness_report.failed_metrics,
            "metrics": result.fairness_report.metrics,
            "phase": "route",
        }

    def remediate_node(state: FairLoopState) -> dict:
        """Remediate biased batch."""
        batch = state.get("batch")
        cycle = state.get("remediation_cycle", 0)
        batch, strategy = orchestrator.remediation_agent.remediate(
            batch, state.get("failed_metrics", []),
            config.protected_attributes[0], cycle
        )
        return {
            "batch": batch,
            "remediation_cycle": cycle + 1,
            "strategy_applied": strategy,
            "phase": "validate",  # Re-validate
        }

    def train_node(state: FairLoopState) -> dict:
        """Train learner on approved batch."""
        batch = state.get("batch")
        metrics = orchestrator.learner.train_on_batch(batch)
        return {
            "learner_metrics": metrics,
            "iteration": state.get("iteration", 0) + 1,
            "phase": "next_batch",
        }

    def route(state: FairLoopState) -> str:
        """Route based on verdict."""
        verdict = state.get("verdict", "")
        cycle = state.get("remediation_cycle", 0)
        if verdict == "APPROVE":
            return "train"
        elif verdict == "REMEDIATE" and cycle < config.max_remediation_cycles:
            return "remediate"
        else:
            return "next_batch"

    def next_batch_node(state: FairLoopState) -> dict:
        """Get next batch."""
        batch = orchestrator.data_agent.get_next_batch()
        iteration = state.get("iteration", 0)
        if batch is None or iteration >= config.max_iterations:
            return {"phase": "done", "converged": True}
        return {"batch": batch, "remediation_cycle": 0, "phase": "validate"}

    # Build graph
    builder = StateGraph(FairLoopState)
    builder.add_node("ingest", ingest_node)
    builder.add_node("validate", validate_node)
    builder.add_node("remediate", remediate_node)
    builder.add_node("train", train_node)
    builder.add_node("next_batch", next_batch_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "validate")
    builder.add_conditional_edges("validate", route, {
        "train": "train",
        "remediate": "remediate",
        "next_batch": "next_batch",
    })
    builder.add_edge("remediate", "validate")
    builder.add_edge("train", "next_batch")
    builder.add_edge("next_batch", "validate")

    graph = builder.compile()
    return graph, orchestrator
