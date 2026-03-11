"""
security/audit_logger.py
========================
Comprehensive audit logging for all data access, security events,
and system operations. Integrates with ProvenanceChain for
tamper-evident records.

Compliant with:
  - ISO 27001 A.12.4 (Logging and Monitoring)
  - GDPR Article 30 (Records of Processing Activities)
  - SOC 2 Type II audit trail requirements

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional

from provenance import ProvenanceChain

logger = logging.getLogger("eco_ai.security.audit")


class AuditEventType(str, Enum):
    # Authentication
    LOGIN_SUCCESS   = "LOGIN_SUCCESS"
    LOGIN_FAILURE   = "LOGIN_FAILURE"
    LOGOUT          = "LOGOUT"
    TOKEN_REFRESH   = "TOKEN_REFRESH"
    TOKEN_REVOKED   = "TOKEN_REVOKED"
    MFA_CHALLENGE   = "MFA_CHALLENGE"

    # Authorisation
    PERMISSION_GRANTED = "PERMISSION_GRANTED"
    PERMISSION_DENIED  = "PERMISSION_DENIED"

    # Data access
    DATA_READ       = "DATA_READ"
    DATA_WRITE      = "DATA_WRITE"
    DATA_DELETE     = "DATA_DELETE"
    DATA_EXPORT     = "DATA_EXPORT"
    DATA_UPLOAD     = "DATA_UPLOAD"
    DATA_ENCRYPTED  = "DATA_ENCRYPTED"
    DATA_DECRYPTED  = "DATA_DECRYPTED"

    # Evaluation
    EVALUATION_RUN  = "EVALUATION_RUN"
    SIMULATION_RUN  = "SIMULATION_RUN"

    # Compliance
    COMPLIANCE_SCAN = "COMPLIANCE_SCAN"
    COMPLIANCE_FAIL = "COMPLIANCE_FAIL"

    # System
    SYSTEM_START    = "SYSTEM_START"
    SYSTEM_STOP     = "SYSTEM_STOP"
    CONFIG_CHANGE   = "CONFIG_CHANGE"
    KEY_ROTATION    = "KEY_ROTATION"
    ALERT_FIRED     = "ALERT_FIRED"
    API_CALL        = "API_CALL"

    # Security incidents
    BRUTE_FORCE     = "BRUTE_FORCE"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""
    event_type:   AuditEventType
    actor:        str            # username or "system"
    action:       str            # human-readable description
    resource:     str            # what was accessed/modified
    outcome:      str            # "success" | "failure" | "warning"
    timestamp:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ip_address:   Optional[str] = None
    session_id:   Optional[str] = None
    request_id:   Optional[str] = None
    details:      dict = field(default_factory=dict)
    risk_score:   float = 0.0   # 0-1, computed based on event type


class AuditLogger:
    """
    Thread-safe audit logger with:
      - In-memory rolling buffer (configurable size)
      - File persistence (JSONL format)
      - ProvenanceChain integration (tamper-evident)
      - Risk scoring per event type
      - Alert on suspicious patterns

    Parameters
    ----------
    max_memory_entries : how many entries to keep in RAM
    log_file_path      : path to JSONL audit log file (None = no file)
    provenance         : ProvenanceChain for tamper-evident ledger
    """

    RISK_SCORES = {
        AuditEventType.LOGIN_FAILURE:       0.6,
        AuditEventType.PERMISSION_DENIED:   0.5,
        AuditEventType.DATA_DELETE:         0.7,
        AuditEventType.DATA_EXPORT:         0.4,
        AuditEventType.CONFIG_CHANGE:       0.6,
        AuditEventType.KEY_ROTATION:        0.3,
        AuditEventType.BRUTE_FORCE:         0.95,
        AuditEventType.SUSPICIOUS_ACTIVITY: 0.85,
        AuditEventType.INTEGRITY_VIOLATION: 1.0,
        AuditEventType.COMPLIANCE_FAIL:     0.75,
    }

    def __init__(
        self,
        max_memory_entries: int = 2000,
        log_file_path: Optional[str] = None,
        provenance: Optional[ProvenanceChain] = None,
    ):
        self._entries: Deque[AuditEntry] = deque(maxlen=max_memory_entries)
        self._lock = threading.Lock()
        self._provenance = provenance or ProvenanceChain(model_id="audit-logger")
        self._failed_logins: dict = {}  # username → count for brute force detection
        self._log_file = Path(log_file_path) if log_file_path else None

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

        self.log(
            event_type=AuditEventType.SYSTEM_START,
            actor="system",
            action="Audit logger initialised",
            resource="audit_system",
            outcome="success",
        )

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[dict] = None,
        risk_override: Optional[float] = None,
    ) -> AuditEntry:
        """Record a new audit event."""
        risk_score = risk_override or self.RISK_SCORES.get(event_type, 0.1)

        entry = AuditEntry(
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            ip_address=ip_address,
            session_id=session_id,
            details=details or {},
            risk_score=risk_score,
        )

        with self._lock:
            self._entries.append(entry)

        # Detect patterns
        self._check_brute_force(entry)
        self._persist(entry)

        # High-risk events go to ProvenanceChain
        if risk_score >= 0.5:
            self._provenance.record(f"audit.{event_type.value}", {
                "actor":     actor,
                "action":    action,
                "resource":  resource,
                "outcome":   outcome,
                "risk":      risk_score,
                "timestamp": entry.timestamp,
            })

        log_level = logging.WARNING if risk_score >= 0.5 else logging.INFO
        logger.log(
            log_level,
            "[Audit] %s | actor=%s | %s | outcome=%s | risk=%.2f",
            event_type.value, actor, action, outcome, risk_score,
        )

        return entry

    # Convenience shorthand methods
    def log_login(self, username: str, success: bool, ip: Optional[str] = None) -> AuditEntry:
        if success:
            self._failed_logins.pop(username, None)
            return self.log(AuditEventType.LOGIN_SUCCESS, username,
                            f"User '{username}' logged in", "auth", "success", ip_address=ip)
        else:
            self._failed_logins[username] = self._failed_logins.get(username, 0) + 1
            return self.log(AuditEventType.LOGIN_FAILURE, username,
                            f"Failed login for '{username}'", "auth", "failure", ip_address=ip)

    def log_permission(self, username: str, permission: str, granted: bool) -> AuditEntry:
        etype = AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED
        return self.log(etype, username,
                        f"Permission '{permission}' {'granted' if granted else 'denied'}",
                        permission, "success" if granted else "failure")

    def log_data_access(
        self,
        actor: str,
        operation: str,
        resource: str,
        rows: Optional[int] = None,
        encrypted: bool = False,
    ) -> AuditEntry:
        op_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
            "upload": AuditEventType.DATA_UPLOAD,
        }
        etype = op_map.get(operation.lower(), AuditEventType.DATA_READ)
        details = {}
        if rows is not None:
            details["rows"] = rows
        if encrypted:
            details["encrypted"] = True
        return self.log(etype, actor, f"Data {operation}: {resource}", resource,
                        "success", details=details)

    def log_api_call(
        self,
        actor: str,
        method: str,
        endpoint: str,
        status_code: int,
        ip: Optional[str] = None,
    ) -> AuditEntry:
        outcome = "success" if status_code < 400 else "failure"
        risk = 0.3 if status_code >= 400 else 0.1
        return self.log(
            AuditEventType.API_CALL, actor,
            f"{method} {endpoint} → {status_code}",
            endpoint, outcome,
            ip_address=ip,
            risk_override=risk,
            details={"method": method, "status_code": status_code},
        )

    def log_compliance_scan(
        self, actor: str, framework: str, passed: bool, findings: int
    ) -> AuditEntry:
        etype = AuditEventType.COMPLIANCE_SCAN if passed else AuditEventType.COMPLIANCE_FAIL
        return self.log(etype, actor,
                        f"Compliance scan: {framework} | {'PASSED' if passed else 'FAILED'} "
                        f"({findings} findings)",
                        f"compliance.{framework}", "success" if passed else "failure",
                        details={"framework": framework, "passed": passed, "findings": findings})

    # ------------------------------------------------------------------
    # Brute force detection
    # ------------------------------------------------------------------

    def _check_brute_force(self, entry: AuditEntry) -> None:
        if entry.event_type != AuditEventType.LOGIN_FAILURE:
            return
        count = self._failed_logins.get(entry.actor, 0)
        if count >= 5:
            logger.critical(
                "[Audit] BRUTE FORCE DETECTED for user '%s' — %d failed attempts!",
                entry.actor, count,
            )
            self.log(
                AuditEventType.BRUTE_FORCE,
                actor="security_system",
                action=f"Brute force detected for '{entry.actor}' ({count} failures)",
                resource="auth",
                outcome="warning",
                risk_override=0.95,
                details={"target_user": entry.actor, "attempt_count": count},
            )

    # ------------------------------------------------------------------
    # Query and export
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 100) -> List[AuditEntry]:
        with self._lock:
            return list(self._entries)[-n:]

    def get_by_actor(self, actor: str, n: int = 50) -> List[AuditEntry]:
        with self._lock:
            return [e for e in self._entries if e.actor == actor][-n:]

    def get_by_type(self, event_type: AuditEventType, n: int = 50) -> List[AuditEntry]:
        with self._lock:
            return [e for e in self._entries if e.event_type == event_type][-n:]

    def get_high_risk(self, threshold: float = 0.5) -> List[AuditEntry]:
        with self._lock:
            return [e for e in self._entries if e.risk_score >= threshold]

    def export_jsonl(self) -> str:
        """Export all entries as JSONL string."""
        lines = []
        with self._lock:
            for entry in self._entries:
                line = {
                    "timestamp":  entry.timestamp,
                    "event_type": entry.event_type.value,
                    "actor":      entry.actor,
                    "action":     entry.action,
                    "resource":   entry.resource,
                    "outcome":    entry.outcome,
                    "risk_score": entry.risk_score,
                    "ip_address": entry.ip_address,
                    "details":    entry.details,
                }
                lines.append(json.dumps(line))
        return "\n".join(lines)

    def get_summary_stats(self) -> dict:
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return {}
        by_type: dict = {}
        by_outcome: dict = {}
        total_risk = 0.0
        for e in entries:
            by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1
            by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1
            total_risk += e.risk_score
        return {
            "total_entries":    len(entries),
            "by_event_type":    by_type,
            "by_outcome":       by_outcome,
            "avg_risk_score":   round(total_risk / len(entries), 4),
            "high_risk_events": sum(1 for e in entries if e.risk_score >= 0.5),
            "failed_logins":    dict(self._failed_logins),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, entry: AuditEntry) -> None:
        if not self._log_file:
            return
        try:
            line = json.dumps({
                "timestamp":  entry.timestamp,
                "event_type": entry.event_type.value,
                "actor":      entry.actor,
                "action":     entry.action,
                "resource":   entry.resource,
                "outcome":    entry.outcome,
                "risk_score": entry.risk_score,
                "details":    entry.details,
            }, default=str)
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.error("[Audit] Persist error: %s", exc)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        log_path = os.getenv("AUDIT_LOG_FILE", "logs/audit.jsonl")
        _audit_logger = AuditLogger(log_file_path=log_path)
    return _audit_logger
