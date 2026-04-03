"""
monetization/analytics.py
==========================
Usage analytics & metering for billing.

Tracks per-tenant usage events (API calls, simulations, badge mints, etc.)
and exposes aggregated metrics for Stripe metering + BI export.

Works standalone — no Mixpanel or GA credentials required.
When MIXPANEL_TOKEN is set, events are also forwarded to Mixpanel.
When GA_MEASUREMENT_ID is set, events go to GA4 via Measurement Protocol.

Architecture
------------
  UsageTracker (core)
    └─ UsageEvent (dataclass)
    └─ in-memory ring buffer (thread-safe)
    └─ optional: MixpanelForwarder
    └─ optional: GA4Forwarder
    └─ DailyUsageSummary (aggregate)

All methods are sync and non-blocking. External forwarding happens in
fire-and-forget daemon threads so it never slows down the main API.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eco_ai.monetization.analytics")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None
    _REQUESTS_OK = False


# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------

class UsageEventType(str, Enum):
    # Core
    LOGIN             = "login"
    LOGOUT            = "logout"
    API_CALL          = "api_call"
    # Simulation / evaluation
    SIMULATION_RUN    = "simulation_run"
    COS_EVALUATION    = "cos_evaluation"
    QUANTUM_THERMAL   = "quantum_thermal"
    REALTIME_POLL     = "realtime_poll"
    # Compliance
    COMPLIANCE_SCAN   = "compliance_scan"
    # Blockchain
    BLOCKCHAIN_PUBLISH= "blockchain_publish"
    ESG_BADGE_MINT    = "esg_badge_mint"
    # Reports
    REPORT_GENERATE   = "report_generate"
    REPORT_EXPORT     = "report_export"
    DATA_EXPORT       = "data_export"
    # Billing events
    SUBSCRIPTION_START= "subscription_start"
    SUBSCRIPTION_END  = "subscription_end"
    OVERAGE_CHARGE    = "overage_charge"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UsageEvent:
    event_type:  UsageEventType
    tenant_id:   str
    user_id:     str
    timestamp:   str       = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id:    str       = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint:    str       = ""
    resource:    str       = ""
    units:       float     = 1.0           # billable units for this event
    cost_usd:    float     = 0.0           # computed cost in USD
    metadata:    Dict      = field(default_factory=dict)
    session_id:  str       = ""
    ip_address:  str       = ""


@dataclass
class DailyUsageSummary:
    tenant_id:           str
    date:                str
    total_events:        int   = 0
    simulations:         int   = 0
    api_calls:           int   = 0
    compliance_scans:    int   = 0
    blockchain_publishes: int  = 0
    esg_badges_minted:   int   = 0
    reports_generated:   int   = 0
    total_units:         float = 0.0
    total_cost_usd:      float = 0.0
    unique_users:        int   = 0


@dataclass
class TenantUsageSnapshot:
    """Live rolling snapshot for billing dashboard."""
    tenant_id:         str
    period_start:      str
    period_end:        str
    plan:              str   = "starter"
    simulations_used:  int   = 0
    api_calls_used:    int   = 0
    badges_minted:     int   = 0
    reports_exported:  int   = 0
    total_units_used:  float = 0.0
    total_cost_usd:    float = 0.0
    overage_units:     float = 0.0
    overage_cost_usd:  float = 0.0


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------

# Units per event type → used to calculate metered billing
EVENT_UNIT_COST: Dict[UsageEventType, float] = {
    UsageEventType.LOGIN:              0.0,
    UsageEventType.LOGOUT:             0.0,
    UsageEventType.API_CALL:           0.001,      # $0.001 per API call
    UsageEventType.SIMULATION_RUN:     0.05,       # $0.05 per sim
    UsageEventType.COS_EVALUATION:     0.02,       # $0.02 per eval
    UsageEventType.QUANTUM_THERMAL:    0.08,       # $0.08 per quantum run
    UsageEventType.REALTIME_POLL:      0.0001,     # $0.0001 per poll
    UsageEventType.COMPLIANCE_SCAN:    0.10,       # $0.10 per scan
    UsageEventType.BLOCKCHAIN_PUBLISH: 0.15,       # $0.15 per publish
    UsageEventType.ESG_BADGE_MINT:     0.25,       # $0.25 per badge
    UsageEventType.REPORT_GENERATE:    0.05,       # $0.05 per report
    UsageEventType.REPORT_EXPORT:      0.03,       # $0.03 per export
    UsageEventType.DATA_EXPORT:        0.02,
    UsageEventType.SUBSCRIPTION_START: 0.0,
    UsageEventType.SUBSCRIPTION_END:   0.0,
    UsageEventType.OVERAGE_CHARGE:     0.0,
}

# Plan quotas (units included per month)
PLAN_QUOTAS: Dict[str, Dict[str, Any]] = {
    "starter": {
        "price_usd_month": 0,
        "simulations":     50,
        "api_calls":       1_000,
        "compliance_scans": 5,
        "badges":          5,
        "reports":         10,
        "overage_multiplier": 1.5,
        "description": "Free tier — perfect for pilots",
    },
    "growth": {
        "price_usd_month": 299,
        "simulations":     500,
        "api_calls":       50_000,
        "compliance_scans": 50,
        "badges":          50,
        "reports":         100,
        "overage_multiplier": 1.2,
        "description": "For growing enterprise teams",
    },
    "enterprise": {
        "price_usd_month": 999,
        "simulations":     5_000,
        "api_calls":       500_000,
        "compliance_scans": 500,
        "badges":          500,
        "reports":         1_000,
        "overage_multiplier": 1.0,
        "description": "Unlimited for large institutions",
    },
    "government": {
        "price_usd_month": 1_499,
        "simulations":     -1,   # unlimited
        "api_calls":       -1,
        "compliance_scans": -1,
        "badges":          -1,
        "reports":         -1,
        "overage_multiplier": 0.0,
        "description": "Custom for government & regulatory bodies",
    },
}


# ---------------------------------------------------------------------------
# Mixpanel forwarder
# ---------------------------------------------------------------------------

class MixpanelForwarder:
    """
    Fire-and-forget forwarding to Mixpanel.
    Falls back silently if token not set or requests unavailable.
    """
    ENDPOINT = "https://api.mixpanel.com/track"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("MIXPANEL_TOKEN", "")
        self._available = bool(self.token and _REQUESTS_OK)
        if not self._available:
            logger.debug("[Mixpanel] Not configured — analytics local only.")

    def send(self, event: UsageEvent) -> None:
        if not self._available:
            return
        payload = [{
            "event": event.event_type.value,
            "properties": {
                "token":       self.token,
                "distinct_id": event.user_id,
                "tenant_id":   event.tenant_id,
                "endpoint":    event.endpoint,
                "resource":    event.resource,
                "units":       event.units,
                "cost_usd":    event.cost_usd,
                "time":        int(time.time()),
                **event.metadata,
            }
        }]
        threading.Thread(
            target=self._post, args=(payload,), daemon=True
        ).start()

    def _post(self, payload: list) -> None:
        try:
            _requests.post(
                self.ENDPOINT,
                json={"data": payload},
                timeout=5
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Mixpanel] Forward failed: %s", exc)


# ---------------------------------------------------------------------------
# GA4 forwarder
# ---------------------------------------------------------------------------

class GA4Forwarder:
    """
    Measurement Protocol v2 forwarder for Google Analytics 4.
    Requires GA_MEASUREMENT_ID and GA_API_SECRET env vars.
    """
    ENDPOINT = "https://www.google-analytics.com/mp/collect"

    def __init__(
        self,
        measurement_id: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.measurement_id = measurement_id or os.getenv("GA_MEASUREMENT_ID", "")
        self.api_secret     = api_secret     or os.getenv("GA_API_SECRET", "")
        self._available = bool(self.measurement_id and self.api_secret and _REQUESTS_OK)

    def send(self, event: UsageEvent) -> None:
        if not self._available:
            return
        body = {
            "client_id": event.tenant_id,
            "events": [{
                "name": event.event_type.value,
                "params": {
                    "user_id":   event.user_id,
                    "endpoint":  event.endpoint,
                    "units":     event.units,
                    "cost_usd":  event.cost_usd,
                }
            }]
        }
        params = {
            "measurement_id": self.measurement_id,
            "api_secret":     self.api_secret,
        }
        threading.Thread(
            target=self._post, args=(body, params), daemon=True
        ).start()

    def _post(self, body: dict, params: dict) -> None:
        try:
            _requests.post(self.ENDPOINT, params=params, json=body, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[GA4] Forward failed: %s", exc)


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """
    Thread-safe usage tracker.

    Usage
    -----
    tracker = UsageTracker()
    tracker.track(UsageEventType.SIMULATION_RUN, tenant_id="bni", user_id="analyst")
    snapshot = tracker.get_snapshot("bni")
    daily    = tracker.get_daily_summary("bni", "2026-03-03")
    """

    def __init__(
        self,
        max_events_per_tenant: int = 50_000,
        mixpanel_token: Optional[str] = None,
        ga_measurement_id: Optional[str] = None,
        ga_api_secret: Optional[str] = None,
    ):
        self._lock  = threading.Lock()
        self._max   = max_events_per_tenant
        # tenant_id → list of UsageEvent
        self._store: Dict[str, List[UsageEvent]] = defaultdict(list)
        # tenant_id → plan name
        self._plans: Dict[str, str] = {}
        # external forwarders
        self._mixpanel = MixpanelForwarder(mixpanel_token)
        self._ga4      = GA4Forwarder(ga_measurement_id, ga_api_secret)

    # ── Public API ────────────────────────────────────────────

    def track(
        self,
        event_type: UsageEventType,
        tenant_id: str,
        user_id: str,
        endpoint:   str  = "",
        resource:   str  = "",
        units:      float = 1.0,
        metadata:   Optional[Dict] = None,
        session_id: str  = "",
        ip_address: str  = "",
    ) -> UsageEvent:
        """Record one usage event and return it."""
        cost = EVENT_UNIT_COST.get(event_type, 0.0) * units
        event = UsageEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            endpoint=endpoint,
            resource=resource,
            units=units,
            cost_usd=cost,
            metadata=metadata or {},
            session_id=session_id,
            ip_address=ip_address,
        )
        with self._lock:
            buf = self._store[tenant_id]
            if len(buf) >= self._max:
                buf.pop(0)  # evict oldest
            buf.append(event)

        # Forward to analytics (non-blocking)
        self._mixpanel.send(event)
        self._ga4.send(event)

        logger.debug(
            "[Analytics] %s | tenant=%s user=%s units=%.3f cost=$%.4f",
            event_type.value, tenant_id, user_id, units, cost
        )
        return event

    def set_plan(self, tenant_id: str, plan: str) -> None:
        """Assign a billing plan to a tenant."""
        if plan not in PLAN_QUOTAS:
            raise ValueError(f"Unknown plan '{plan}'. Valid: {list(PLAN_QUOTAS.keys())}")
        with self._lock:
            self._plans[tenant_id] = plan

    def get_plan(self, tenant_id: str) -> str:
        return self._plans.get(tenant_id, "starter")

    def get_events(
        self,
        tenant_id: str,
        event_type: Optional[UsageEventType] = None,
        since: Optional[str] = None,
        limit: int = 1_000,
    ) -> List[UsageEvent]:
        """Return events for tenant, optionally filtered."""
        with self._lock:
            events = list(self._store.get(tenant_id, []))

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_snapshot(self, tenant_id: str) -> TenantUsageSnapshot:
        """
        Rolling current-month usage snapshot for billing API.
        """
        now   = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with self._lock:
            events = [
                e for e in self._store.get(tenant_id, [])
                if e.timestamp >= start.isoformat()
            ]

        plan_name = self._plans.get(tenant_id, "starter")
        plan      = PLAN_QUOTAS[plan_name]

        sims    = sum(1 for e in events if e.event_type == UsageEventType.SIMULATION_RUN)
        api     = sum(1 for e in events if e.event_type == UsageEventType.API_CALL)
        badges  = sum(1 for e in events if e.event_type == UsageEventType.ESG_BADGE_MINT)
        reports = sum(1 for e in events if e.event_type in (
            UsageEventType.REPORT_GENERATE, UsageEventType.REPORT_EXPORT
        ))
        total_units = sum(e.units for e in events)
        total_cost  = sum(e.cost_usd for e in events)

        # Overage calculation
        overage_units = 0.0
        overage_cost  = 0.0
        if plan["simulations"] != -1 and sims > plan["simulations"]:
            over_sims = sims - plan["simulations"]
            overage_units += over_sims
            overage_cost  += over_sims * EVENT_UNIT_COST[UsageEventType.SIMULATION_RUN] \
                             * plan["overage_multiplier"]
        if plan["api_calls"] != -1 and api > plan["api_calls"]:
            over_api = api - plan["api_calls"]
            overage_cost += over_api * EVENT_UNIT_COST[UsageEventType.API_CALL] \
                            * plan["overage_multiplier"]

        return TenantUsageSnapshot(
            tenant_id=tenant_id,
            period_start=start.isoformat(),
            period_end=now.isoformat(),
            plan=plan_name,
            simulations_used=sims,
            api_calls_used=api,
            badges_minted=badges,
            reports_exported=reports,
            total_units_used=total_units,
            total_cost_usd=total_cost,
            overage_units=overage_units,
            overage_cost_usd=overage_cost,
        )

    def get_daily_summary(
        self, tenant_id: str, date: Optional[str] = None
    ) -> DailyUsageSummary:
        """Aggregate stats for a single UTC day (YYYY-MM-DD)."""
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            events = [
                e for e in self._store.get(tenant_id, [])
                if e.timestamp.startswith(date)
            ]

        unique_users = len({e.user_id for e in events})

        return DailyUsageSummary(
            tenant_id=tenant_id,
            date=date,
            total_events=len(events),
            simulations=sum(1 for e in events if e.event_type == UsageEventType.SIMULATION_RUN),
            api_calls=sum(1 for e in events if e.event_type == UsageEventType.API_CALL),
            compliance_scans=sum(1 for e in events if e.event_type == UsageEventType.COMPLIANCE_SCAN),
            blockchain_publishes=sum(1 for e in events if e.event_type == UsageEventType.BLOCKCHAIN_PUBLISH),
            esg_badges_minted=sum(1 for e in events if e.event_type == UsageEventType.ESG_BADGE_MINT),
            reports_generated=sum(1 for e in events if e.event_type == UsageEventType.REPORT_GENERATE),
            total_units=sum(e.units for e in events),
            total_cost_usd=sum(e.cost_usd for e in events),
            unique_users=unique_users,
        )

    def get_all_tenant_ids(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def export_jsonl(self, tenant_id: str) -> str:
        """Export all events for a tenant as JSONL for BI ingestion."""
        events = self.get_events(tenant_id, limit=100_000)
        lines = []
        for e in events:
            lines.append(json.dumps({
                "event_id":   e.event_id,
                "event_type": e.event_type.value,
                "tenant_id":  e.tenant_id,
                "user_id":    e.user_id,
                "timestamp":  e.timestamp,
                "endpoint":   e.endpoint,
                "resource":   e.resource,
                "units":      e.units,
                "cost_usd":   e.cost_usd,
                "metadata":   e.metadata,
            }))
        return "\n".join(lines)

    def export_bi_dict(self, tenant_id: str) -> Dict[str, Any]:
        """
        Structured dict ready for Tableau / Power BI / Looker ingestion.
        """
        events  = self.get_events(tenant_id, limit=100_000)
        snap    = self.get_snapshot(tenant_id)
        plan    = PLAN_QUOTAS.get(snap.plan, PLAN_QUOTAS["starter"])

        # Group by day
        by_day: Dict[str, DailyUsageSummary] = {}
        all_dates = sorted({e.timestamp[:10] for e in events})
        for d in all_dates:
            by_day[d] = self.get_daily_summary(tenant_id, d)

        # Event type distribution
        type_dist: Dict[str, int] = defaultdict(int)
        for e in events:
            type_dist[e.event_type.value] += 1

        # Top users
        user_costs: Dict[str, float] = defaultdict(float)
        for e in events:
            user_costs[e.user_id] += e.cost_usd
        top_users = sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "tenant_id":       tenant_id,
            "plan":            snap.plan,
            "plan_details":    plan,
            "snapshot":        snap.__dict__,
            "event_type_dist": dict(type_dist),
            "top_users":       [{"user_id": u, "cost_usd": c} for u, c in top_users],
            "daily_series":    [v.__dict__ for v in by_day.values()],
            "total_events":    len(events),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            tenant_count = len(self._store)
            total_events = sum(len(v) for v in self._store.values())
        return {
            "status":        "ok",
            "tenants":       tenant_count,
            "total_events":  total_events,
            "mixpanel":      self._mixpanel._available,
            "ga4":           self._ga4._available,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(
            mixpanel_token=os.getenv("MIXPANEL_TOKEN"),
            ga_measurement_id=os.getenv("GA_MEASUREMENT_ID"),
            ga_api_secret=os.getenv("GA_API_SECRET"),
        )
    return _tracker
