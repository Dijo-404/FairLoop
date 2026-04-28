"""
FairLoop Audit Log System
Immutable, append-only audit log for all Validator decisions.
SQLite backend for local deployments. Exportable as compliance reports.
"""
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class AuditEntry:
    """A single audit log entry for a Validator decision."""
    id: str
    iteration: int
    batch_id: str
    timestamp: str
    sample_count: int
    protected_attributes: List[str]
    metrics: Dict[str, float]
    semantic_bias_score: Optional[float]
    flagged_proxies: List[str]
    verdict: str                    # APPROVE, REMEDIATE, REJECT
    reason: str
    remediation_applied: Optional[str]
    final_verdict: Optional[str]
    approved_batch_id: Optional[str]


class AuditLog:
    """
    Immutable, append-only audit log backed by SQLite.
    Supports querying, filtering, and compliance report generation.
    """

    def __init__(self, db_path: str = "fairloop_audit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the audit table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                iteration INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                protected_attributes TEXT NOT NULL,
                metrics TEXT NOT NULL,
                semantic_bias_score REAL,
                flagged_proxies TEXT NOT NULL,
                verdict TEXT NOT NULL,
                reason TEXT NOT NULL,
                remediation_applied TEXT,
                final_verdict TEXT,
                approved_batch_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_iteration ON audit_log(iteration)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_verdict ON audit_log(verdict)
        """)
        conn.commit()
        conn.close()

    def log(
        self,
        iteration: int,
        batch_id: str,
        sample_count: int,
        protected_attributes: List[str],
        metrics: Dict[str, float],
        verdict: str,
        reason: str,
        semantic_bias_score: Optional[float] = None,
        flagged_proxies: Optional[List[str]] = None,
        remediation_applied: Optional[str] = None,
        final_verdict: Optional[str] = None,
        approved_batch_id: Optional[str] = None,
    ) -> AuditEntry:
        """Append an immutable entry to the audit log."""
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            iteration=iteration,
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_count=sample_count,
            protected_attributes=protected_attributes,
            metrics=metrics,
            semantic_bias_score=semantic_bias_score,
            flagged_proxies=flagged_proxies or [],
            verdict=verdict,
            reason=reason,
            remediation_applied=remediation_applied,
            final_verdict=final_verdict,
            approved_batch_id=approved_batch_id,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO audit_log (
                id, iteration, batch_id, timestamp, sample_count,
                protected_attributes, metrics, semantic_bias_score,
                flagged_proxies, verdict, reason, remediation_applied,
                final_verdict, approved_batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.iteration,
            entry.batch_id,
            entry.timestamp,
            entry.sample_count,
            json.dumps(entry.protected_attributes),
            json.dumps(entry.metrics),
            entry.semantic_bias_score,
            json.dumps(entry.flagged_proxies),
            entry.verdict,
            entry.reason,
            entry.remediation_applied,
            entry.final_verdict,
            entry.approved_batch_id,
        ))
        conn.commit()
        conn.close()

        return entry

    def get_entries(
        self,
        iteration: Optional[int] = None,
        verdict: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """Query audit log entries with optional filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if iteration is not None:
            query += " AND iteration = ?"
            params.append(iteration)
        if verdict is not None:
            query += " AND verdict = ?"
            params.append(verdict)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        conn.close()

        results = []
        for row in rows:
            entry = dict(row)
            entry["protected_attributes"] = json.loads(entry["protected_attributes"])
            entry["metrics"] = json.loads(entry["metrics"])
            entry["flagged_proxies"] = json.loads(entry["flagged_proxies"])
            results.append(entry)

        return results

    def get_summary(self) -> Dict:
        """Get summary statistics of the audit log."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
        approved = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE verdict='APPROVE'").fetchone()["c"]
        remediated = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE verdict='REMEDIATE'").fetchone()["c"]
        rejected = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE verdict='REJECT'").fetchone()["c"]

        # Average metrics over time
        metrics_rows = conn.execute(
            "SELECT metrics FROM audit_log ORDER BY iteration"
        ).fetchall()

        conn.close()

        metrics_trend = []
        for row in metrics_rows:
            m = json.loads(row["metrics"])
            metrics_trend.append(m)

        return {
            "total_decisions": total,
            "approved": approved,
            "remediated": remediated,
            "rejected": rejected,
            "approval_rate": approved / max(total, 1),
            "metrics_trend": metrics_trend,
        }

    def export_compliance_report(self) -> Dict:
        """Generate a compliance report suitable for EU AI Act / EEOC reporting."""
        summary = self.get_summary()
        entries = self.get_entries(limit=10000)

        return {
            "report_type": "FairLoop Fairness Compliance Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system": "FairLoop v1.0 — In-Loop Bias Prevention",
            "summary": {
                "total_batches_evaluated": summary["total_decisions"],
                "batches_approved": summary["approved"],
                "batches_remediated": summary["remediated"],
                "batches_rejected": summary["rejected"],
                "approval_rate_percent": round(summary["approval_rate"] * 100, 2),
            },
            "fairness_metrics_applied": [
                "Demographic Parity Difference (threshold ≤ 0.10)",
                "Disparate Impact Ratio (threshold ≥ 0.80, EEOC 80% rule)",
                "Equal Opportunity Difference (threshold ≤ 0.10)",
                "Predictive Parity Difference (threshold ≤ 0.10)",
                "Individual Fairness Score (threshold ≥ 0.85)",
                "Representation Balance (threshold ≥ 0.75)",
                "Proxy Variable Detection (correlation > 0.60 flagged)",
            ],
            "audit_trail": entries[:100],  # Last 100 entries
            "compliance_standards": [
                "EU AI Act (Article 10 — Data Governance)",
                "US EEOC Uniform Guidelines on Employee Selection (80% rule)",
                "IEEE 7010-2020 (Well-being Impact Assessment)",
            ],
        }
