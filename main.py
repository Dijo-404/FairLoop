#!/usr/bin/env python3
"""
FairLoop — In-Loop Bias Prevention System
Main entry point. Run the full pipeline or start the API server.

Usage:
    python main.py run          # Run the full FairLoop pipeline
    python main.py server       # Start FastAPI server + dashboard
    python main.py demo         # Quick demo (5 iterations)
    python main.py baseline     # Train WITHOUT FairLoop (for comparison)
"""
import os
import sys
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import PipelineConfig
from agents.orchestrator import Orchestrator


def run_pipeline(args):
    """Run the full FairLoop pipeline."""
    config = PipelineConfig(
        max_iterations=args.iterations,
        batch_size=args.batch_size,
        protected_attributes=args.protected_attrs.split(","),
        enable_semantic_layer=args.semantic,
    )

    orchestrator = Orchestrator(config)
    result = orchestrator.run(max_iterations=args.iterations)

    # Save results
    output_path = args.output or "fairloop_results.json"
    serializable = {
        "iterations_completed": result["iterations_completed"],
        "total_approved": result["total_approved"],
        "total_remediated": result["total_remediated"],
        "total_rejected": result["total_rejected"],
        "final_evaluation": result["final_evaluation"],
        "training_history": result["training_history"],
        "config": result["config"],
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Generate compliance report
    report = orchestrator.audit_log.export_compliance_report()
    report_path = output_path.replace(".json", "_compliance.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Compliance report saved to {report_path}")

    return result


def run_baseline(args):
    """Train WITHOUT FairLoop for comparison (biased baseline)."""
    from agents.data_agent import DataAgent
    from agents.learner_agent import LearnerAgent
    from core.config import PipelineConfig
    import numpy as np

    config = PipelineConfig(
        protected_attributes=args.protected_attrs.split(","),
    )

    print("=" * 70)
    print("  BASELINE — Training WITHOUT FairLoop (biased)")
    print("=" * 70)

    # Load data
    data_agent = DataAgent(config)
    df = data_agent.load_dataset()

    # Split
    n_test = int(len(df) * 0.2)
    test_df = df.tail(n_test)
    train_df = df.head(len(df) - n_test)

    # Prepare learner
    learner = LearnerAgent(config)
    X_test, y_test = learner.prepare_features(test_df)

    sensitive_test = {}
    for attr in config.protected_attributes:
        priv_val = config.privileged_groups.get(attr)
        if priv_val and attr in test_df.columns:
            sensitive_test[attr] = (
                test_df[attr].astype(str).str.strip() == str(priv_val).strip()
            ).astype(int).values
    learner.set_test_data(X_test, y_test, sensitive_test)

    # Train on ALL data (no filtering)
    batches = data_agent.chunk_into_batches(batch_size=args.batch_size)
    for i, batch in enumerate(batches[:args.iterations]):
        metrics = learner.train_on_batch(batch)

    print("\nBASELINE Results (NO FairLoop):")
    eval_metrics = learner.evaluate()
    for k, v in eval_metrics.items():
        flag = "WARN" if "disparate_impact" in k and v < 0.8 else "    "
        print(f"  {flag} {k}: {v:.4f}")

    return eval_metrics


def run_server(args):
    """Start the FastAPI server."""
    import uvicorn
    print("Starting FairLoop API server...")
    print(f"  API: http://localhost:{args.port}")
    print(f"  Dashboard: open dashboard/index.html in browser")
    uvicorn.run("api.main:app", host="0.0.0.0", port=args.port, reload=True)


def run_demo(args):
    """Quick demo: run baseline then FairLoop, show comparison."""
    print("\n" + "-" * 25)
    print("  STEP 1: Training WITHOUT FairLoop (biased baseline)")
    print("-" * 25)
    args.iterations = 5
    baseline = run_baseline(args)

    print("\n\n" + "-" * 25)
    print("  STEP 2: Training WITH FairLoop (bias prevention)")
    print("-" * 25)
    args.iterations = 5
    fairloop = run_pipeline(args)

    print("\n\n" + "=" * 70)
    print("  COMPARISON: Baseline vs FairLoop")
    print("=" * 70)

    fl_eval = fairloop.get("final_evaluation", {})
    for key in sorted(set(list(baseline.keys()) + list(fl_eval.keys()))):
        b_val = baseline.get(key, "N/A")
        f_val = fl_eval.get(key, "N/A")
        b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
        f_str = f"{f_val:.4f}" if isinstance(f_val, float) else str(f_val)

        improved = ""
        if isinstance(b_val, float) and isinstance(f_val, float):
            if "disparate_impact" in key:
                improved = "OK" if f_val > b_val else "WARN"
            elif "accuracy" in key:
                improved = "OK" if f_val >= b_val * 0.95 else "WARN"

        print(f"  {improved} {key:45s} | Baseline: {b_str:>8s} | FairLoop: {f_str:>8s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairLoop — In-Loop Bias Prevention")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run FairLoop pipeline")
    run_parser.add_argument("--iterations", type=int, default=20)
    run_parser.add_argument("--batch-size", type=int, default=256)
    run_parser.add_argument("--protected-attrs", default="sex")
    run_parser.add_argument("--semantic", action="store_true")
    run_parser.add_argument("--output", default="fairloop_results.json")

    # Baseline command
    base_parser = subparsers.add_parser("baseline", help="Train without FairLoop")
    base_parser.add_argument("--iterations", type=int, default=20)
    base_parser.add_argument("--batch-size", type=int, default=256)
    base_parser.add_argument("--protected-attrs", default="sex")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--port", type=int, default=8000)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Quick comparison demo")
    demo_parser.add_argument("--batch-size", type=int, default=256)
    demo_parser.add_argument("--protected-attrs", default="sex")
    demo_parser.add_argument("--output", default="fairloop_results.json")
    demo_parser.add_argument("--semantic", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "server":
        run_server(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()
