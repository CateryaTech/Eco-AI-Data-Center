"""
monetization/billing.py
========================
Stripe-powered subscription + pay-per-use billing for Eco AI Data Center.

Supports:
  - Multi-tenant subscription management (Starter / Growth / Enterprise / Government)
  - Pay-per-use metered billing via Stripe Usage Records API
  - Webhook handler for Stripe events (payment success, invoice, cancellation)
  - Invoice generation (local + Stripe) with provenance hash
  - Graceful fallback when STRIPE_SECRET_KEY not set (local billing simulation)

Architecture
------------
  BillingService
    ├── SubscriptionManager  — create/upgrade/cancel plans
    ├── MeterRecorder        — push usage records to Stripe Meter API
    ├── InvoiceBuilder       — build itemized invoices with CATERYA provenance
    └── WebhookHandler       — process Stripe webhook events

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from monetization.analytics import (
    UsageTracker,
    UsageEventType,
    PLAN_QUOTAS,
    TenantUsageSnapshot,
    get_tracker,
)

logger = logging.getLogger("eco_ai.monetization.billing")

# ---------------------------------------------------------------------------
# Optional Stripe SDK
# ---------------------------------------------------------------------------

try:
    import stripe as _stripe
    _STRIPE_OK = True
except ImportError:
    _stripe = None          # type: ignore
    _STRIPE_OK = False

# ---------------------------------------------------------------------------
# Stripe product/price IDs (set via env or replace with your own)
# ---------------------------------------------------------------------------

STRIPE_PRICE_IDS: Dict[str, str] = {
    "starter":    os.getenv("STRIPE_PRICE_STARTER",    "price_starter_free"),
    "growth":     os.getenv("STRIPE_PRICE_GROWTH",     "price_growth_299"),
    "enterprise": os.getenv("STRIPE_PRICE_ENTERPRISE", "price_enterprise_999"),
    "government": os.getenv("STRIPE_PRICE_GOVERNMENT", "price_govt_1499"),
}

# Stripe Meter event names (configure in Stripe dashboard)
STRIPE_METER_EVENTS: Dict[UsageEventType, str] = {
    UsageEventType.SIMULATION_RUN:     "simulation_run",
    UsageEventType.API_CALL:           "api_call",
    UsageEventType.COMPLIANCE_SCAN:    "compliance_scan",
    UsageEventType.ESG_BADGE_MINT:     "esg_badge_mint",
    UsageEventType.REPORT_GENERATE:    "report_generate",
    UsageEventType.BLOCKCHAIN_PUBLISH: "blockchain_publish",
    UsageEventType.QUANTUM_THERMAL:    "quantum_thermal",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BillingStatus(str, Enum):
    ACTIVE     = "active"
    TRIALING   = "trialing"
    PAST_DUE   = "past_due"
    CANCELLED  = "cancelled"
    PAUSED     = "paused"


@dataclass
class Subscription:
    tenant_id:          str
    plan:               str
    status:             BillingStatus   = BillingStatus.ACTIVE
    subscription_id:    str             = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:16]}")
    customer_id:        str             = field(default_factory=lambda: f"cus_{uuid.uuid4().hex[:16]}")
    trial_end:          Optional[str]   = None
    current_period_start: str          = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_period_end:   str          = field(default_factory=lambda: (
        datetime.now(timezone.utc) + timedelta(days=30)
    ).isoformat())
    stripe_linked:      bool            = False
    metadata:           Dict            = field(default_factory=dict)
    created_at:         str             = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class InvoiceLineItem:
    description: str
    quantity:    float
    unit_cost:   float
    total:       float
    event_type:  str = ""


@dataclass
class Invoice:
    invoice_id:       str
    tenant_id:        str
    period_start:     str
    period_end:       str
    plan:             str
    line_items:       List[InvoiceLineItem] = field(default_factory=list)
    subscription_fee: float = 0.0
    usage_total:      float = 0.0
    overage_total:    float = 0.0
    tax_rate:         float = 0.11         # 11% PPN Indonesia
    tax_amount:       float = 0.0
    grand_total:      float = 0.0
    currency:         str   = "USD"
    status:           str   = "draft"
    stripe_invoice_id: Optional[str] = None
    provenance_hash:  str   = ""
    issued_at:        str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    due_date:         str   = field(default_factory=lambda: (
        datetime.now(timezone.utc) + timedelta(days=14)
    ).isoformat())


@dataclass
class MeterRecord:
    tenant_id:    str
    event_type:   UsageEventType
    quantity:     float
    recorded_at:  str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stripe_event_id: Optional[str] = None
    meter_id:     str  = field(default_factory=lambda: f"meter_{uuid.uuid4().hex[:12]}")


# ---------------------------------------------------------------------------
# SubscriptionManager
# ---------------------------------------------------------------------------

class SubscriptionManager:
    """
    Create, retrieve, upgrade, and cancel tenant subscriptions.
    Uses Stripe when configured, local store otherwise.
    """

    def __init__(self, stripe_key: Optional[str] = None):
        self._key  = stripe_key or os.getenv("STRIPE_SECRET_KEY", "")
        self._live = bool(self._key and _STRIPE_OK)
        self._subs: Dict[str, Subscription] = {}  # tenant_id → Subscription
        self._lock = threading.Lock()

        if self._live:
            _stripe.api_key = self._key
            logger.info("[Billing] Stripe configured — live mode active.")
        else:
            logger.info("[Billing] Stripe not configured — local simulation mode.")

    def create_subscription(
        self,
        tenant_id: str,
        plan: str,
        customer_email: str = "",
        payment_method_id: str = "",
        trial_days: int = 14,
    ) -> Subscription:
        """Create a new subscription for a tenant."""
        if plan not in PLAN_QUOTAS:
            raise ValueError(f"Invalid plan: {plan}")

        if self._live and payment_method_id:
            return self._stripe_create(tenant_id, plan, customer_email,
                                       payment_method_id, trial_days)
        return self._local_create(tenant_id, plan, trial_days)

    def _local_create(self, tenant_id: str, plan: str, trial_days: int = 14) -> Subscription:
        now     = datetime.now(timezone.utc)
        sub = Subscription(
            tenant_id=tenant_id,
            plan=plan,
            status=BillingStatus.TRIALING if trial_days > 0 else BillingStatus.ACTIVE,
            trial_end=(now + timedelta(days=trial_days)).isoformat() if trial_days > 0 else None,
            current_period_end=(now + timedelta(days=30)).isoformat(),
            stripe_linked=False,
        )
        with self._lock:
            self._subs[tenant_id] = sub
        logger.info("[Billing] Local subscription created: %s / %s", tenant_id, plan)
        return sub

    def _stripe_create(
        self, tenant_id: str, plan: str,
        email: str, payment_method_id: str, trial_days: int
    ) -> Subscription:
        try:
            customer = _stripe.Customer.create(
                email=email,
                payment_method=payment_method_id,
                invoice_settings={"default_payment_method": payment_method_id},
                metadata={"tenant_id": tenant_id},
            )
            price_id = STRIPE_PRICE_IDS[plan]
            params: Dict[str, Any] = {
                "customer": customer["id"],
                "items": [{"price": price_id}],
                "metadata": {"tenant_id": tenant_id, "plan": plan},
            }
            if trial_days > 0:
                params["trial_period_days"] = trial_days

            stripe_sub = _stripe.Subscription.create(**params)
            sub = Subscription(
                tenant_id=tenant_id,
                plan=plan,
                subscription_id=stripe_sub["id"],
                customer_id=customer["id"],
                status=BillingStatus(stripe_sub["status"]),
                current_period_end=datetime.fromtimestamp(
                    stripe_sub["current_period_end"], tz=timezone.utc
                ).isoformat(),
                stripe_linked=True,
            )
            with self._lock:
                self._subs[tenant_id] = sub
            return sub
        except Exception as exc:
            logger.warning("[Billing] Stripe create failed (%s) — falling back.", exc)
            return self._local_create(tenant_id, plan, trial_days)

    def get_subscription(self, tenant_id: str) -> Optional[Subscription]:
        with self._lock:
            return self._subs.get(tenant_id)

    def upgrade_plan(self, tenant_id: str, new_plan: str) -> Subscription:
        """Upgrade or downgrade a tenant's plan."""
        sub = self.get_subscription(tenant_id)
        if not sub:
            return self.create_subscription(tenant_id, new_plan)

        old_plan   = sub.plan
        sub.plan   = new_plan
        sub.status = BillingStatus.ACTIVE

        if self._live and sub.stripe_linked:
            try:
                stripe_sub = _stripe.Subscription.retrieve(sub.subscription_id)
                item_id    = stripe_sub["items"]["data"][0]["id"]
                _stripe.Subscription.modify(
                    sub.subscription_id,
                    items=[{"id": item_id, "price": STRIPE_PRICE_IDS[new_plan]}],
                    proration_behavior="create_prorations",
                )
            except Exception as exc:
                logger.warning("[Billing] Stripe upgrade failed: %s", exc)

        logger.info("[Billing] %s upgraded %s → %s", tenant_id, old_plan, new_plan)
        return sub

    def cancel_subscription(self, tenant_id: str) -> bool:
        sub = self.get_subscription(tenant_id)
        if not sub:
            return False
        sub.status = BillingStatus.CANCELLED
        if self._live and sub.stripe_linked:
            try:
                _stripe.Subscription.delete(sub.subscription_id)
            except Exception as exc:
                logger.warning("[Billing] Stripe cancel failed: %s", exc)
        return True

    def all_subscriptions(self) -> List[Subscription]:
        with self._lock:
            return list(self._subs.values())


# ---------------------------------------------------------------------------
# MeterRecorder
# ---------------------------------------------------------------------------

class MeterRecorder:
    """
    Records usage units to Stripe Meter API (or local store).
    Called after every API event to enable pay-per-use billing.
    """

    def __init__(self, stripe_key: Optional[str] = None):
        self._key  = stripe_key or os.getenv("STRIPE_SECRET_KEY", "")
        self._live = bool(self._key and _STRIPE_OK)
        self._log: List[MeterRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        tenant_id: str,
        customer_id: str,
        event_type: UsageEventType,
        quantity: float = 1.0,
    ) -> MeterRecord:
        """
        Record billable usage. Non-blocking — Stripe call runs in background thread.
        """
        meter_name = STRIPE_METER_EVENTS.get(event_type)
        mr = MeterRecord(tenant_id=tenant_id, event_type=event_type, quantity=quantity)

        with self._lock:
            self._log.append(mr)

        if self._live and meter_name and customer_id:
            threading.Thread(
                target=self._stripe_record,
                args=(mr, meter_name, customer_id),
                daemon=True,
            ).start()

        return mr

    def _stripe_record(self, mr: MeterRecord, meter_name: str, customer_id: str) -> None:
        try:
            event = _stripe.billing.MeterEvent.create(
                event_name=meter_name,
                payload={
                    "stripe_customer_id": customer_id,
                    "value": str(int(mr.quantity)),
                },
            )
            mr.stripe_event_id = event.get("identifier", "")
            logger.debug("[Billing] Meter recorded: %s × %.1f → %s", meter_name, mr.quantity, customer_id)
        except Exception as exc:
            logger.warning("[Billing] Stripe meter failed: %s", exc)

    def get_log(self, tenant_id: Optional[str] = None) -> List[MeterRecord]:
        with self._lock:
            if tenant_id:
                return [r for r in self._log if r.tenant_id == tenant_id]
            return list(self._log)


# ---------------------------------------------------------------------------
# InvoiceBuilder
# ---------------------------------------------------------------------------

class InvoiceBuilder:
    """
    Build itemized monthly invoices from usage analytics data.
    Includes CATERYA ProvenanceChain hash for audit-grade traceability.
    """

    def __init__(self, tracker: Optional[UsageTracker] = None):
        self._tracker = tracker or get_tracker()
        self._invoices: Dict[str, Invoice] = {}

    def build_monthly_invoice(
        self,
        tenant_id: str,
        year: int,
        month: int,
        plan: str = "starter",
        currency: str = "USD",
    ) -> Invoice:
        """Generate an itemized invoice for a calendar month."""
        from calendar import monthrange
        _, last_day = monthrange(year, month)
        period_start = f"{year:04d}-{month:02d}-01"
        period_end   = f"{year:04d}-{month:02d}-{last_day:02d}"

        snapshot = self._tracker.get_snapshot(tenant_id)
        plan_cfg  = PLAN_QUOTAS.get(plan, PLAN_QUOTAS["starter"])

        line_items: List[InvoiceLineItem] = []

        # Base subscription fee
        sub_fee = float(plan_cfg.get("price_usd_month", 0))

        # Usage line items
        usage_map = {
            "Simulation Runs":      (UsageEventType.SIMULATION_RUN,     snapshot.simulations_used),
            "API Calls":            (UsageEventType.API_CALL,           snapshot.api_calls_used),
            "ESG Badge Mints":      (UsageEventType.ESG_BADGE_MINT,     snapshot.badges_minted),
            "Reports Generated":    (UsageEventType.REPORT_GENERATE,    snapshot.reports_exported),
        }

        usage_total = 0.0
        for desc, (evt, qty) in usage_map.items():
            if qty > 0:
                from monetization.analytics import EVENT_UNIT_COST
                unit_cost = EVENT_UNIT_COST.get(evt, 0.0)
                total     = unit_cost * qty
                usage_total += total
                line_items.append(InvoiceLineItem(
                    description=desc,
                    quantity=float(qty),
                    unit_cost=unit_cost,
                    total=total,
                    event_type=evt.value,
                ))

        overage = snapshot.overage_cost_usd
        subtotal = sub_fee + usage_total + overage
        tax_amt  = round(subtotal * 0.11, 4)  # 11% PPN
        grand    = round(subtotal + tax_amt, 4)

        # Provenance hash for invoice integrity
        prov_data = f"{tenant_id}|{period_start}|{period_end}|{grand}"
        prov_hash = hashlib.sha256(prov_data.encode()).hexdigest()

        invoice = Invoice(
            invoice_id=f"inv-{uuid.uuid4().hex[:10]}",
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            plan=plan,
            line_items=line_items,
            subscription_fee=sub_fee,
            usage_total=usage_total,
            overage_total=overage,
            tax_amount=tax_amt,
            grand_total=grand,
            currency=currency,
            status="issued",
            provenance_hash=prov_hash,
        )

        self._invoices[invoice.invoice_id] = invoice
        return invoice

    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        return self._invoices.get(invoice_id)

    def to_html(self, invoice: Invoice) -> str:
        """Render invoice as HTML — ready to email or display in dashboard."""
        plan_cfg = PLAN_QUOTAS.get(invoice.plan, {})
        rows = ""
        for li in invoice.line_items:
            rows += f"""
            <tr>
              <td>{li.description}</td>
              <td style="text-align:right">{li.quantity:,.0f}</td>
              <td style="text-align:right">${li.unit_cost:.4f}</td>
              <td style="text-align:right"><strong>${li.total:.4f}</strong></td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Invoice {invoice.invoice_id} — CateryaTech</title>
<style>
  body {{ font-family: Inter, sans-serif; max-width: 860px; margin: 40px auto; color: #1a1a2e; }}
  h1 {{ color: #6c63ff; }} h3 {{ color: #444; }}
  table {{ width:100%; border-collapse:collapse; margin: 16px 0; }}
  th {{ background:#6c63ff; color:white; padding:10px; text-align:left; }}
  td {{ padding:9px 10px; border-bottom:1px solid #eee; }}
  .total-row {{ background:#f5f3ff; font-weight:bold; }}
  .badge {{ background:#22c55e; color:white; padding:3px 10px; border-radius:20px; font-size:12px; }}
  .hash {{ font-family:monospace; font-size:11px; color:#888; word-break:break-all; }}
</style>
</head>
<body>
<img src="https://cateryatech.com/logo.png" alt="CateryaTech" height="36" onerror="this.style.display='none'">
<h1>Tax Invoice</h1>
<table style="margin-bottom:24px">
  <tr><td><strong>Invoice ID</strong></td><td>{invoice.invoice_id}</td>
      <td><strong>Status</strong></td><td><span class="badge">{invoice.status.upper()}</span></td></tr>
  <tr><td><strong>Tenant</strong></td><td>{invoice.tenant_id}</td>
      <td><strong>Plan</strong></td><td>{invoice.plan.title()} — {plan_cfg.get('description','')}</td></tr>
  <tr><td><strong>Period</strong></td><td colspan="3">{invoice.period_start} → {invoice.period_end}</td></tr>
  <tr><td><strong>Issued</strong></td><td>{invoice.issued_at[:10]}</td>
      <td><strong>Due</strong></td><td>{invoice.due_date[:10]}</td></tr>
</table>

<h3>Line Items</h3>
<table>
  <tr><th>Description</th><th>Qty</th><th>Unit Rate</th><th>Amount</th></tr>
  <tr><td>Subscription — {invoice.plan.title()}</td>
      <td style="text-align:right">1</td>
      <td style="text-align:right">${invoice.subscription_fee:,.2f}</td>
      <td style="text-align:right"><strong>${invoice.subscription_fee:,.2f}</strong></td></tr>
  {rows}
  {"<tr><td>Overage charges</td><td>—</td><td>—</td><td style='text-align:right'><strong>$"+f"{invoice.overage_total:.4f}</strong></td></tr>" if invoice.overage_total > 0 else ""}
</table>

<table style="max-width:360px;margin-left:auto">
  <tr><td>Subtotal</td><td style="text-align:right">${(invoice.subscription_fee+invoice.usage_total+invoice.overage_total):.4f}</td></tr>
  <tr><td>PPN 11%</td><td style="text-align:right">${invoice.tax_amount:.4f}</td></tr>
  <tr class="total-row"><td>Total ({invoice.currency})</td><td style="text-align:right">${invoice.grand_total:.4f}</td></tr>
</table>

<p style="margin-top:32px;font-size:12px">
  <strong>Payment:</strong> Bank transfer to BCA 123-456-7890 a/n CateryaTech Indonesia
  <br>or pay online: <a href="https://eco-ai.cateryatech.com/billing/{invoice.invoice_id}">billing link</a>
</p>
<p class="hash">Provenance hash: {invoice.provenance_hash}<br>
Issued by CateryaTech &bull; cateryatech@proton.me &bull; NPWP 00.000.000.0-000.000</p>
</body></html>"""


# ---------------------------------------------------------------------------
# WebhookHandler
# ---------------------------------------------------------------------------

class WebhookHandler:
    """
    Process Stripe webhook events.
    Verifies signature with STRIPE_WEBHOOK_SECRET env var.
    """

    def __init__(
        self,
        subscription_manager: Optional[SubscriptionManager] = None,
        tracker: Optional[UsageTracker] = None,
        webhook_secret: Optional[str] = None,
    ):
        self._subs    = subscription_manager or SubscriptionManager()
        self._tracker = tracker or get_tracker()
        self._secret  = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET", "")
        self._events: List[Dict] = []

    def handle(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """
        Verify and process a Stripe webhook delivery.
        Returns {"status": "ok", "event_type": "..."} or {"status": "error", "detail": "..."}.
        """
        # Verify signature
        if self._secret and _STRIPE_OK:
            try:
                event = _stripe.Webhook.construct_event(payload, sig_header, self._secret)
            except Exception as exc:
                return {"status": "error", "detail": str(exc)}
        else:
            # Dev mode — parse without verification
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                return {"status": "error", "detail": str(exc)}

        self._events.append(event)
        event_type = event.get("type", "")
        data       = event.get("data", {}).get("object", {})

        if event_type == "customer.subscription.updated":
            self._on_subscription_updated(data)
        elif event_type == "customer.subscription.deleted":
            self._on_subscription_cancelled(data)
        elif event_type == "invoice.payment_succeeded":
            self._on_payment_succeeded(data)
        elif event_type == "invoice.payment_failed":
            self._on_payment_failed(data)

        return {"status": "ok", "event_type": event_type}

    def _on_subscription_updated(self, data: Dict) -> None:
        tenant_id = data.get("metadata", {}).get("tenant_id", "")
        plan      = data.get("metadata", {}).get("plan", "")
        if tenant_id and plan:
            self._subs.upgrade_plan(tenant_id, plan)
            self._tracker.track(UsageEventType.SUBSCRIPTION_START, tenant_id, "system",
                                metadata={"plan": plan, "source": "stripe_webhook"})

    def _on_subscription_cancelled(self, data: Dict) -> None:
        tenant_id = data.get("metadata", {}).get("tenant_id", "")
        if tenant_id:
            self._subs.cancel_subscription(tenant_id)
            self._tracker.track(UsageEventType.SUBSCRIPTION_END, tenant_id, "system")

    def _on_payment_succeeded(self, data: Dict) -> None:
        logger.info("[Billing] Payment succeeded: invoice=%s amount=$%.2f",
                    data.get("id"), data.get("amount_paid", 0) / 100)

    def _on_payment_failed(self, data: Dict) -> None:
        logger.warning("[Billing] Payment FAILED: invoice=%s", data.get("id"))

    def recent_events(self, n: int = 20) -> List[Dict]:
        return self._events[-n:]


# ---------------------------------------------------------------------------
# BillingService (facade)
# ---------------------------------------------------------------------------

class BillingService:
    """
    Single entry point for all billing operations.

    Usage
    -----
    billing = BillingService()
    billing.onboard_tenant("telkom", "enterprise", "tech@telkom.co.id")
    billing.record_usage("telkom", "analyst01", "cus_xxx", UsageEventType.SIMULATION_RUN)
    invoice = billing.generate_invoice("telkom", 2026, 3)
    html    = billing.invoice_html(invoice.invoice_id)
    """

    def __init__(self, stripe_key: Optional[str] = None):
        self._tracker  = get_tracker()
        self._subs     = SubscriptionManager(stripe_key)
        self._meter    = MeterRecorder(stripe_key)
        self._invoicer = InvoiceBuilder(self._tracker)
        self._webhooks = WebhookHandler(self._subs, self._tracker)

    def onboard_tenant(
        self,
        tenant_id: str,
        plan: str = "starter",
        email: str = "",
        payment_method_id: str = "",
        trial_days: int = 14,
    ) -> Subscription:
        """Full onboarding: create subscription + set analytics plan."""
        sub = self._subs.create_subscription(
            tenant_id, plan, email, payment_method_id, trial_days
        )
        self._tracker.set_plan(tenant_id, plan)
        self._tracker.track(
            UsageEventType.SUBSCRIPTION_START, tenant_id, "system",
            metadata={"plan": plan, "email": email, "trial_days": trial_days}
        )
        logger.info("[Billing] Tenant onboarded: %s / %s", tenant_id, plan)
        return sub

    def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        customer_id: str,
        event_type: UsageEventType,
        units: float = 1.0,
        endpoint: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a billable event in analytics + Stripe meter."""
        self._tracker.track(
            event_type, tenant_id, user_id,
            endpoint=endpoint, units=units, metadata=metadata or {}
        )
        self._meter.record(tenant_id, customer_id, event_type, units)

    def get_snapshot(self, tenant_id: str) -> TenantUsageSnapshot:
        return self._tracker.get_snapshot(tenant_id)

    def generate_invoice(self, tenant_id: str, year: int, month: int) -> Invoice:
        sub  = self._subs.get_subscription(tenant_id)
        plan = sub.plan if sub else "starter"
        return self._invoicer.build_monthly_invoice(tenant_id, year, month, plan)

    def invoice_html(self, invoice_id: str) -> str:
        inv = self._invoicer.get_invoice(invoice_id)
        if not inv:
            return "<p>Invoice not found.</p>"
        return self._invoicer.to_html(inv)

    def handle_webhook(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        return self._webhooks.handle(payload, sig_header)

    def upgrade_tenant(self, tenant_id: str, new_plan: str) -> Subscription:
        self._tracker.set_plan(tenant_id, new_plan)
        return self._subs.upgrade_plan(tenant_id, new_plan)

    def cancel_tenant(self, tenant_id: str) -> bool:
        return self._subs.cancel_subscription(tenant_id)

    def bi_export(self, tenant_id: str) -> Dict:
        return self._tracker.export_bi_dict(tenant_id)

    def health(self) -> Dict:
        return {
            **self._tracker.health(),
            "stripe_available": _STRIPE_OK and bool(os.getenv("STRIPE_SECRET_KEY")),
            "subscriptions":    len(self._subs.all_subscriptions()),
        }
