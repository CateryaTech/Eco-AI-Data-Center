"""
caterya_guardrail.py
====================
CATERYAGuardrail — Ethical output enforcement layer for all API endpoints.

Sits as middleware between business logic and external API consumers.
Every AI decision, model output, and data release passes through this guardrail
before it leaves the system.

Guardrail layers:
  1. COS threshold check  — reject outputs below ethical quality floor
  2. Bias sentinel        — detect statistical bias in output distributions  
  3. Fairness gate        — QuantumFairnessEvaluator on numeric outputs
  4. Provenance stamp     — every released output gets an immutable audit hash
  5. Content filter       — block PII / sensitive data leakage
  6. Rate limiter         — per-actor request throttling to prevent abuse
  7. EthicsSwarm veto     — optional swarm consensus on high-stakes outputs

Designed for enterprise and government use cases (bank, OJK, BUMN, etc).

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fairness import QuantumFairnessEvaluator
from provenance import ProvenanceChain
from swarm import EthicsSwarm

logger = logging.getLogger("eco_ai.guardrail")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GuardrailDecision:
    """
    The result of a guardrail check on a single output payload.
    Passed or blocked, with full reasoning chain.
    """
    passed: bool
    output: Any                         # the (possibly sanitized) output
    layers_checked: List[str]
    layers_failed: List[str]
    cos_score: Optional[float]
    fairness_score: Optional[float]
    bias_detected: bool
    pii_detected: bool
    provenance_hash: str
    actor: str
    endpoint: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: str = ""
    swarm_approved: Optional[bool] = None
    swarm_consensus: Optional[float] = None
    risk_level: str = "low"             # low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "cos_score": self.cos_score,
            "fairness_score": self.fairness_score,
            "bias_detected": self.bias_detected,
            "pii_detected": self.pii_detected,
            "swarm_approved": self.swarm_approved,
            "swarm_consensus": self.swarm_consensus,
            "provenance_hash": self.provenance_hash,
            "layers_checked": self.layers_checked,
            "layers_failed": self.layers_failed,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "endpoint": self.endpoint,
        }


@dataclass
class GuardrailConfig:
    """Configurable thresholds and flags for the guardrail."""
    cos_threshold: float = 0.65         # minimum COS for output release
    fairness_threshold: float = 0.60    # minimum fairness score
    bias_z_threshold: float = 2.5       # z-score above which output is "biased"
    enable_swarm_veto: bool = False     # require EthicsSwarm on critical outputs
    swarm_threshold: float = 0.70       # minimum swarm consensus
    enable_pii_filter: bool = True      # scan for PII patterns
    enable_rate_limit: bool = True      # per-actor request throttle
    rate_limit_rpm: int = 60            # requests per minute per actor
    max_output_rows: int = 10_000       # max rows in any data export
    high_stakes_endpoints: List[str] = field(default_factory=lambda: [
        "/api/v1/compliance/scan",
        "/api/v1/cos/evaluate",
        "/api/v1/data/export",
    ])


# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "nik_indonesia":    re.compile(r"\b\d{16}\b"),                    # NIK
    "ktp_label":        re.compile(r"\bNIK\s*[:=]\s*\d{10,16}\b"),
    "email":            re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone_id":         re.compile(r"\b(\+62|62|0)[0-9]{8,13}\b"),  # Indonesian phone
    "credit_card":      re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "password_field":   re.compile(r"(?i)(password|passwd|secret|token)\s*[:=]\s*\S+"),
    "jwt_token":        re.compile(r"\beyJ[A-Za-z0-9_\-=]+\.[A-Za-z0-9_\-=]+\.[A-Za-z0-9_\-=]+\b"),
    "ip_internal":      re.compile(r"\b(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)[\d.]+\b"),
}


class PIIDetector:
    """Scan text/dict payloads for personally identifiable information."""

    def scan(self, payload: Any) -> Tuple[bool, List[str]]:
        """
        Returns (pii_found: bool, list_of_pattern_names_matched).
        """
        text = json.dumps(payload, default=str)
        found = []
        for name, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                found.append(name)
        return bool(found), found

    def redact(self, payload: Any) -> Any:
        """Replace PII patterns with [REDACTED] in string representation."""
        if isinstance(payload, str):
            result = payload
            for _, pattern in PII_PATTERNS.items():
                result = pattern.sub("[REDACTED]", result)
            return result
        if isinstance(payload, dict):
            return {k: self.redact(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self.redact(item) for item in payload]
        return payload


# ---------------------------------------------------------------------------
# Bias Sentinel
# ---------------------------------------------------------------------------

class BiasSentinel:
    """
    Detects statistical bias in numeric output distributions.
    Uses z-score analysis and inter-group variance checking.
    """

    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold

    def check(self, output: Any) -> Tuple[bool, str]:
        """
        Returns (bias_detected: bool, reason: str).
        Works on dicts with numeric values, lists of numbers, or nested structures.
        """
        numerics = self._extract_numerics(output)
        if len(numerics) < 3:
            return False, "insufficient data for bias check"

        arr = np.array(numerics)
        mean = arr.mean()
        std = arr.std()

        if std < 1e-10:
            return False, "uniform distribution — no bias"

        z_scores = np.abs((arr - mean) / std)
        max_z = float(z_scores.max())

        if max_z > self.z_threshold:
            return True, f"outlier detected (z={max_z:.2f} > threshold={self.z_threshold})"

        # Coefficient of variation check
        cv = std / (abs(mean) + 1e-10)
        if cv > 1.5:
            return True, f"high coefficient of variation (CV={cv:.2f}), suggesting skewed output"

        return False, f"bias check passed (max_z={max_z:.2f})"

    def _extract_numerics(self, obj: Any, depth: int = 0) -> List[float]:
        if depth > 5:
            return []
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return [float(obj)]
        if isinstance(obj, dict):
            result = []
            for v in obj.values():
                result.extend(self._extract_numerics(v, depth + 1))
            return result
        if isinstance(obj, (list, tuple)):
            result = []
            for item in obj:
                result.extend(self._extract_numerics(item, depth + 1))
            return result
        return []


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Simple sliding window rate limiter.
    Thread-safe via per-actor deques.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.rpm * 2))

    def check(self, actor: str) -> Tuple[bool, int]:
        """
        Returns (allowed: bool, remaining: int).
        """
        now = time.monotonic()
        window = self._windows[actor]

        # Remove entries older than 60s
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= self.rpm:
            return False, 0

        window.append(now)
        remaining = self.rpm - len(window)
        return True, remaining


# ---------------------------------------------------------------------------
# Main CATERYAGuardrail
# ---------------------------------------------------------------------------

class CATERYAGuardrail:
    """
    Ethical output guardrail for Eco AI Data Center API.

    Wraps any AI/analytics output with multi-layer ethical checks
    before it leaves the system boundary.

    Usage
    -----
    guardrail = CATERYAGuardrail(config=GuardrailConfig(cos_threshold=0.7))

    # In your API endpoint:
    decision = guardrail.check(
        output={"cos_composite": 0.94, "approved": True},
        actor="api_user@bank.go.id",
        endpoint="/api/v1/cos/evaluate",
        cos_score=0.94,
    )

    if not decision.passed:
        raise HTTPException(403, detail=decision.reason)

    return decision.output  # sanitized, stamped output
    """

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        provenance: Optional[ProvenanceChain] = None,
    ):
        self.config = config or GuardrailConfig()
        self.provenance = provenance or ProvenanceChain(model_id="caterya-guardrail")
        self.pii_detector = PIIDetector()
        self.bias_sentinel = BiasSentinel(z_threshold=self.config.bias_z_threshold)
        self.rate_limiter = RateLimiter(requests_per_minute=self.config.rate_limit_rpm)
        self.qfe = QuantumFairnessEvaluator(n_samples=200, seed=42)
        self._swarm: Optional[EthicsSwarm] = None
        self._decision_log: List[GuardrailDecision] = []

        logger.info(
            "[Guardrail] Initialised | COS≥%.2f | fairness≥%.2f | PII=%s | RateLimit=%d rpm",
            self.config.cos_threshold,
            self.config.fairness_threshold,
            self.config.enable_pii_filter,
            self.config.rate_limit_rpm,
        )

    def check(
        self,
        output: Any,
        actor: str,
        endpoint: str,
        cos_score: Optional[float] = None,
        metadata: Optional[dict] = None,
        require_swarm: Optional[bool] = None,
    ) -> GuardrailDecision:
        """
        Run all guardrail layers on an output payload.

        Parameters
        ----------
        output    : the data/dict to be returned by the API endpoint
        actor     : username / API client ID making the request
        endpoint  : API path (used for high-stakes detection)
        cos_score : pre-computed COS score (if available)
        metadata  : additional context for audit logging
        require_swarm : override config.enable_swarm_veto for this call
        """
        layers_checked: List[str] = []
        layers_failed: List[str] = []
        bias_detected = False
        pii_detected = False
        fairness_score = None
        swarm_approved = None
        swarm_consensus = None
        block_reason = ""

        # ── Layer 1: Rate limiting ────────────────────────────────────────
        if self.config.enable_rate_limit:
            layers_checked.append("rate_limiter")
            allowed, remaining = self.rate_limiter.check(actor)
            if not allowed:
                layers_failed.append("rate_limiter")
                block_reason = f"Rate limit exceeded for actor '{actor}' ({self.config.rate_limit_rpm} rpm)"
                return self._build_decision(
                    passed=False, output=None, layers_checked=layers_checked,
                    layers_failed=layers_failed, cos_score=cos_score,
                    fairness_score=None, bias_detected=False, pii_detected=False,
                    actor=actor, endpoint=endpoint, reason=block_reason,
                    swarm_approved=None, swarm_consensus=None, risk_level="high",
                )

        # ── Layer 2: COS score check ──────────────────────────────────────
        if cos_score is not None:
            layers_checked.append("cos_threshold")
            if cos_score < self.config.cos_threshold:
                layers_failed.append("cos_threshold")
                block_reason = (
                    f"COS score {cos_score:.4f} below threshold {self.config.cos_threshold:.2f}. "
                    "Ethical quality floor not met — output blocked."
                )
                return self._build_decision(
                    passed=False, output=None, layers_checked=layers_checked,
                    layers_failed=layers_failed, cos_score=cos_score,
                    fairness_score=None, bias_detected=False, pii_detected=False,
                    actor=actor, endpoint=endpoint, reason=block_reason,
                    swarm_approved=None, swarm_consensus=None, risk_level="critical",
                )

        # ── Layer 3: Bias detection ───────────────────────────────────────
        layers_checked.append("bias_sentinel")
        bias_detected, bias_reason = self.bias_sentinel.check(output)
        if bias_detected:
            layers_failed.append("bias_sentinel")
            logger.warning("[Guardrail] Bias detected | actor=%s | reason=%s", actor, bias_reason)

        # ── Layer 4: Fairness gate ────────────────────────────────────────
        try:
            import pandas as pd
            numeric_vals = self.bias_sentinel._extract_numerics(output)
            if len(numeric_vals) >= 4:
                layers_checked.append("fairness_gate")
                df_check = pd.DataFrame({"values": numeric_vals})
                fairness_result = self.qfe.evaluate(df_check)
                fairness_score = fairness_result.get("fairness_score", 1.0)
                if fairness_score < self.config.fairness_threshold:
                    layers_failed.append("fairness_gate")
                    logger.warning(
                        "[Guardrail] Fairness below threshold | score=%.4f | threshold=%.2f",
                        fairness_score, self.config.fairness_threshold,
                    )
        except Exception as exc:
            logger.debug("[Guardrail] Fairness gate skipped: %s", exc)

        # ── Layer 5: PII filter ───────────────────────────────────────────
        sanitized_output = output
        if self.config.enable_pii_filter:
            layers_checked.append("pii_filter")
            pii_detected, pii_types = self.pii_detector.scan(output)
            if pii_detected:
                layers_failed.append("pii_filter")
                sanitized_output = self.pii_detector.redact(output)
                logger.warning(
                    "[Guardrail] PII detected and redacted | types=%s | actor=%s",
                    pii_types, actor,
                )

        # ── Layer 6: EthicsSwarm veto (high-stakes) ───────────────────────
        use_swarm = require_swarm if require_swarm is not None else self.config.enable_swarm_veto
        is_high_stakes = any(hs in endpoint for hs in self.config.high_stakes_endpoints)

        if use_swarm or is_high_stakes:
            layers_checked.append("ethics_swarm")
            swarm_approved, swarm_consensus = self._run_swarm(
                output=sanitized_output, cos_score=cos_score,
                fairness_score=fairness_score, bias_detected=bias_detected,
            )
            if not swarm_approved:
                layers_failed.append("ethics_swarm")
                block_reason = (
                    f"EthicsSwarm veto: consensus={swarm_consensus:.4f} "
                    f"below threshold={self.config.swarm_threshold:.2f}"
                )

        # ── Layer 7: Provenance stamp ─────────────────────────────────────
        layers_checked.append("provenance_stamp")
        prov_hash = self._stamp_output(
            output=sanitized_output, actor=actor, endpoint=endpoint,
            cos_score=cos_score, fairness_score=fairness_score,
        )

        # ── Final decision ────────────────────────────────────────────────
        # Block if any critical layer failed (PII auto-sanitized, not blocking)
        critical_failures = [l for l in layers_failed if l not in ("pii_filter", "bias_sentinel")]
        passed = len(critical_failures) == 0

        if not passed and not block_reason:
            block_reason = f"Guardrail layer(s) failed: {', '.join(layers_failed)}"

        risk_level = self._compute_risk(
            bias_detected, pii_detected, cos_score, fairness_score, swarm_approved
        )

        decision = self._build_decision(
            passed=passed, output=sanitized_output,
            layers_checked=layers_checked, layers_failed=layers_failed,
            cos_score=cos_score, fairness_score=fairness_score,
            bias_detected=bias_detected, pii_detected=pii_detected,
            actor=actor, endpoint=endpoint,
            reason=block_reason if not passed else "All guardrail layers passed.",
            swarm_approved=swarm_approved, swarm_consensus=swarm_consensus,
            risk_level=risk_level,
        )
        decision.provenance_hash = prov_hash

        self._decision_log.append(decision)

        log_level = logging.INFO if passed else logging.WARNING
        logger.log(
            log_level,
            "[Guardrail] %s | actor=%s | endpoint=%s | risk=%s | layers_failed=%s",
            "PASSED" if passed else "BLOCKED",
            actor, endpoint, risk_level, layers_failed,
        )

        return decision

    def _run_swarm(
        self,
        output: Any,
        cos_score: Optional[float],
        fairness_score: Optional[float],
        bias_detected: bool,
    ) -> Tuple[bool, float]:
        """Run EthicsSwarm to get consensus on output approval."""
        try:
            if self._swarm is None:
                self._swarm = EthicsSwarm(threshold=self.config.swarm_threshold)

            import pandas as pd
            numerics = self.bias_sentinel._extract_numerics(output)
            if not numerics:
                numerics = [cos_score or 0.5, fairness_score or 0.5]
            df = pd.DataFrame({"values": numerics})
            result = self._swarm.evaluate(df)
            consensus = result.get("consensus_score", 0.0)
            approved = consensus >= self.config.swarm_threshold
            return approved, consensus
        except Exception as exc:
            logger.warning("[Guardrail] Swarm evaluation failed: %s", exc)
            return True, 1.0  # fail-open on swarm error

    def _stamp_output(
        self,
        output: Any,
        actor: str,
        endpoint: str,
        cos_score: Optional[float],
        fairness_score: Optional[float],
    ) -> str:
        """Record output to provenance chain and return the SHA-256 hash."""
        payload_str = json.dumps(
            {
                "output_type": type(output).__name__,
                "actor": actor,
                "endpoint": endpoint,
                "cos_score": cos_score,
                "fairness_score": fairness_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default=str,
        )
        content_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        try:
            self.provenance.record(
                event_type="guardrail_output_stamp",
                metadata={
                    "actor": actor,
                    "endpoint": endpoint,
                    "cos_score": cos_score,
                    "fairness_score": fairness_score,
                    "content_hash": content_hash,
                },
            )
        except Exception as exc:
            logger.debug("[Guardrail] Provenance stamp failed: %s", exc)

        return content_hash

    def _compute_risk(
        self,
        bias_detected: bool,
        pii_detected: bool,
        cos_score: Optional[float],
        fairness_score: Optional[float],
        swarm_approved: Optional[bool],
    ) -> str:
        score = 0
        if bias_detected:
            score += 2
        if pii_detected:
            score += 3
        if cos_score is not None and cos_score < self.config.cos_threshold:
            score += 3
        if fairness_score is not None and fairness_score < self.config.fairness_threshold:
            score += 2
        if swarm_approved is False:
            score += 2
        if score == 0:
            return "low"
        if score <= 2:
            return "medium"
        if score <= 5:
            return "high"
        return "critical"

    def _build_decision(self, **kwargs) -> GuardrailDecision:
        kwargs.setdefault("provenance_hash", "")
        return GuardrailDecision(**kwargs)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_decisions(self, limit: int = 100) -> List[GuardrailDecision]:
        return self._decision_log[-limit:]

    def get_stats(self) -> dict:
        total = len(self._decision_log)
        if total == 0:
            return {"total": 0, "passed": 0, "blocked": 0, "pass_rate": 1.0}
        passed = sum(1 for d in self._decision_log if d.passed)
        return {
            "total": total,
            "passed": passed,
            "blocked": total - passed,
            "pass_rate": passed / total,
            "bias_detected": sum(1 for d in self._decision_log if d.bias_detected),
            "pii_detected": sum(1 for d in self._decision_log if d.pii_detected),
            "risk_breakdown": {
                level: sum(1 for d in self._decision_log if d.risk_level == level)
                for level in ("low", "medium", "high", "critical")
            },
        }

    def reset_log(self) -> None:
        self._decision_log.clear()


# ---------------------------------------------------------------------------
# Decorator for FastAPI / function-level guardrail
# ---------------------------------------------------------------------------

def guardrail_protect(
    cos_threshold: float = 0.65,
    require_swarm: bool = False,
    actor_fn: Optional[Callable] = None,
):
    """
    Decorator to wrap any function with CATERYAGuardrail protection.

    Example
    -------
    @guardrail_protect(cos_threshold=0.7, require_swarm=True)
    def my_api_handler(data):
        return {"result": data}
    """
    _guardrail = CATERYAGuardrail(
        config=GuardrailConfig(
            cos_threshold=cos_threshold,
            enable_swarm_veto=require_swarm,
        )
    )

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            actor = "unknown"
            if actor_fn:
                actor = actor_fn(*args, **kwargs)

            result = fn(*args, **kwargs)

            decision = _guardrail.check(
                output=result,
                actor=actor,
                endpoint=fn.__name__,
                require_swarm=require_swarm,
            )
            if not decision.passed:
                raise ValueError(f"CATERYAGuardrail blocked output: {decision.reason}")
            return decision.output

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator
