"""
FairLoop FastAPI Backend
Serves the pipeline status, metrics, and audit log to the dashboard.
"""
import os
import sys
import json
import asyncio
import threading
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import Orchestrator
from core.config import PipelineConfig
from core.audit_log import AuditLog

app = FastAPI(
    title="FairLoop API",
    description="In-Loop Bias Prevention System — Real-time fairness monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_state = {
    "status": "idle",      # idle, running, completed, error
    "orchestrator": None,
    "events": [],
    "ws_clients": [],
}


class PipelineStartRequest(BaseModel):
    max_iterations: int = 20
    batch_size: int = 256
    dataset: str = "scikit-learn/adult-census-income"
    protected_attributes: List[str] = ["sex"]
    enable_semantic: bool = False


# === WebSocket for live updates ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline_state["ws_clients"].append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        pipeline_state["ws_clients"].remove(websocket)


async def broadcast_event(event: Dict):
    """Send event to all connected WebSocket clients."""
    for ws in pipeline_state["ws_clients"]:
        try:
            await ws.send_json(event)
        except Exception:
            pass


def sync_event_handler(event: Dict):
    """Synchronous event handler that queues events for WebSocket broadcast."""
    pipeline_state["events"].append(event)


# === REST Endpoints ===
@app.get("/")
def root():
    return {
        "name": "FairLoop API",
        "version": "1.0.0",
        "status": pipeline_state["status"],
    }


@app.post("/pipeline/start")
def start_pipeline(request: PipelineStartRequest):
    """Start the FairLoop training pipeline."""
    if pipeline_state["status"] == "running":
        return JSONResponse(
            status_code=409,
            content={"error": "Pipeline already running"}
        )

    config = PipelineConfig(
        max_iterations=request.max_iterations,
        batch_size=request.batch_size,
        dataset_name=request.dataset,
        protected_attributes=request.protected_attributes,
        enable_semantic_layer=request.enable_semantic,
    )

    orchestrator = Orchestrator(config)
    orchestrator.on_event(sync_event_handler)
    pipeline_state["orchestrator"] = orchestrator
    pipeline_state["status"] = "running"
    pipeline_state["events"] = []

    # Run in background thread
    def run_pipeline():
        try:
            result = orchestrator.run()
            pipeline_state["status"] = "completed"
            pipeline_state["result"] = result
        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return {"status": "started", "config": json.loads(json.dumps(request.dict()))}


@app.get("/pipeline/status")
def pipeline_status():
    """Get current pipeline status."""
    result = {
        "status": pipeline_state["status"],
        "events_count": len(pipeline_state["events"]),
    }

    orch = pipeline_state.get("orchestrator")
    if orch:
        result["state"] = {
            "iteration": orch.state.get("iteration", 0),
            "total_approved": orch.state.get("total_approved", 0),
            "total_remediated": orch.state.get("total_remediated", 0),
            "total_rejected": orch.state.get("total_rejected", 0),
        }

    if pipeline_state["status"] == "completed":
        result["result"] = pipeline_state.get("result", {})

    return result


@app.get("/pipeline/metrics")
def pipeline_metrics():
    """Get training metrics over time."""
    orch = pipeline_state.get("orchestrator")
    if not orch:
        return {"metrics": []}

    history = orch.learner.get_training_history()
    return {"metrics": history}


@app.get("/pipeline/events")
def pipeline_events(since: int = 0):
    """Get pipeline events since a given index."""
    return {"events": pipeline_state["events"][since:]}


@app.get("/audit/entries")
def audit_entries(
    iteration: Optional[int] = None,
    verdict: Optional[str] = None,
    limit: int = 50,
):
    """Get audit log entries."""
    orch = pipeline_state.get("orchestrator")
    if not orch:
        return {"entries": []}

    entries = orch.audit_log.get_entries(
        iteration=iteration, verdict=verdict, limit=limit
    )
    return {"entries": entries}


@app.get("/audit/summary")
def audit_summary():
    """Get audit log summary statistics."""
    orch = pipeline_state.get("orchestrator")
    if not orch:
        return {"summary": {}}

    return {"summary": orch.audit_log.get_summary()}


@app.get("/audit/compliance-report")
def compliance_report():
    """Generate a compliance report."""
    orch = pipeline_state.get("orchestrator")
    if not orch:
        return {"error": "No pipeline data available"}

    return orch.audit_log.export_compliance_report()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
