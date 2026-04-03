"""
tests/test_monetization_e2e.py
================================
End-to-end test suite: analytics → billing → insights → BI export → ISO 42001.

This is the enterprise readiness test — covers the complete journey a customer
like Telkom or Pertamina would take from onboarding to invoice to ESG report.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import json
import os
import sys
import unittest
import logging
import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monetization.analytics import (
    UsageTracker,
    UsageEventType,
    PLAN_QUOTAS,
    EVENT_UNIT_COST,
    DailyUsageSummary,
    TenantUsageSnapshot,
    MixpanelForwarder,
    GA4Forwarder,
)
from monetization.billing import (
    BillingService,
    SubscriptionManager,
    MeterRecorder,
    InvoiceBuilder,
    WebhookHandler,
    BillingStatus,
    Subscription,
    Invoice,
    InvoiceLineItem,
)
from reports.insights import (
    InsightsEngine,
    ISO42001Assessor,
    BIExporter,
    ESGReportData,
    Insight,
    InsightCategory,
    InsightPriority,
    ISO42001Assessment,
    RENEWABLE_OPTIONS_ID,
    GRID_CARBON_INTENSITY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker() -> UsageTracker:
    return UsageTracker()


def _populate_tracker(tracker: UsageTracker, tenant_id: str, n_events: int = 20) -> None:
    events = [
        UsageEventType.SIMULATION_RUN,
        UsageEventType.API_CALL,
        UsageEventType.COS_EVALUATION,
        UsageEventType.COMPLIANCE_SCAN,
        UsageEventType.ESG_BADGE_MINT,
        UsageEventType.REPORT_GENERATE,
        UsageEventType.BLOCKCHAIN_PUBLISH,
    ]
    for i in range(n_events):
        tracker.track(
            events[i % len(events)],
            tenant_id,
            f"user_{i % 3:02d}",
            endpoint=f"/api/v1/endpoint_{i % 5}",
            metadata={"test_run": True, "iteration": i},
        )


def _make_esg_data(tenant_id: str = "telkom") -> ESGReportData:
    engine = InsightsEngine()
    assessor = ISO42001Assessor()
    insights = engine.generate(
        cos_score=0.79,
        pue=1.82,
        wue=1.65,
        carbon_intensity_kg_kwh=0.748,
        renewable_pct=0.0,
        tenant_region="jawa_bali",
        annual_kwh=50_000_000,
        compliance_scores={"GDPR": 0.56, "ISO27001": 0.79, "SRN_PPI": 0.55},
    )
    iso = assessor.assess(
        tenant_id=tenant_id,
        cos_score=0.94,
        swarm_approved=True,
        swarm_consensus=0.93,
        provenance_chain_events=12,
        audit_log_entries=87,
    )
    return ESGReportData(
        tenant_id=tenant_id,
        report_type="quarterly",
        period="Q1-2026",
        cos_composite=0.94,
        cos_swarm_approved=True,
        pue=1.82,
        wue=1.65,
        carbon_total_tonnes=37_400.0,
        carbon_intensity=0.748,
        renewable_pct=0.0,
        insights=insights,
        iso42001=iso,
        compliance_scores={"GDPR": 0.56, "ISO27001": 0.79, "SRN_PPI": 0.55},
        provenance_hashes=["abc123def456", "def789ghi012"],
        badge_ids=["badge-xyz-001"],
    )


# ===========================================================================
# [1] USAGE ANALYTICS
# ===========================================================================

class TestUsageAnalytics(unittest.TestCase):

    def setUp(self):
        self.tracker = _make_tracker()

    def test_track_returns_usage_event(self):
        from monetization.analytics import UsageEvent
        e = self.tracker.track(UsageEventType.SIMULATION_RUN, "bni", "analyst01")
        self.assertIsNotNone(e)
        self.assertEqual(e.tenant_id, "bni")
        self.assertEqual(e.user_id, "analyst01")
        self.assertEqual(e.event_type, UsageEventType.SIMULATION_RUN)

    def test_event_cost_computed(self):
        e = self.tracker.track(UsageEventType.SIMULATION_RUN, "bni", "user01")
        expected = EVENT_UNIT_COST[UsageEventType.SIMULATION_RUN]
        self.assertAlmostEqual(e.cost_usd, expected, places=6)

    def test_free_events_have_zero_cost(self):
        e = self.tracker.track(UsageEventType.LOGIN, "bni", "user01")
        self.assertEqual(e.cost_usd, 0.0)

    def test_multiple_tenants_isolated(self):
        self.tracker.track(UsageEventType.SIMULATION_RUN, "telkom", "u01")
        self.tracker.track(UsageEventType.SIMULATION_RUN, "pertamina", "u02")
        snap_t = self.tracker.get_snapshot("telkom")
        snap_p = self.tracker.get_snapshot("pertamina")
        self.assertEqual(snap_t.simulations_used, 1)
        self.assertEqual(snap_p.simulations_used, 1)
        self.assertEqual(snap_t.tenant_id, "telkom")

    def test_set_and_get_plan(self):
        self.tracker.set_plan("telkom", "enterprise")
        self.assertEqual(self.tracker.get_plan("telkom"), "enterprise")

    def test_invalid_plan_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.set_plan("telkom", "super_mega_plan")

    def test_snapshot_returns_correct_type(self):
        snap = self.tracker.get_snapshot("newco")
        self.assertIsInstance(snap, TenantUsageSnapshot)

    def test_snapshot_plan_reflects_set_plan(self):
        self.tracker.set_plan("bni", "growth")
        snap = self.tracker.get_snapshot("bni")
        self.assertEqual(snap.plan, "growth")

    def test_snapshot_counts_each_event_type(self):
        _populate_tracker(self.tracker, "telkom", 21)
        snap = self.tracker.get_snapshot("telkom")
        self.assertGreater(snap.simulations_used, 0)
        self.assertGreater(snap.api_calls_used, 0)
        self.assertGreater(snap.badges_minted, 0)

    def test_daily_summary_structure(self):
        _populate_tracker(self.tracker, "pertamina", 10)
        daily = self.tracker.get_daily_summary("pertamina")
        self.assertIsInstance(daily, DailyUsageSummary)
        self.assertGreater(daily.total_events, 0)
        self.assertGreater(daily.unique_users, 0)

    def test_daily_summary_unique_users_correct(self):
        self.tracker.track(UsageEventType.API_CALL, "t1", "userA")
        self.tracker.track(UsageEventType.API_CALL, "t1", "userA")
        self.tracker.track(UsageEventType.API_CALL, "t1", "userB")
        daily = self.tracker.get_daily_summary("t1")
        self.assertEqual(daily.unique_users, 2)

    def test_get_events_filter_by_type(self):
        self.tracker.track(UsageEventType.SIMULATION_RUN, "bni", "u01")
        self.tracker.track(UsageEventType.API_CALL, "bni", "u01")
        sims = self.tracker.get_events("bni", event_type=UsageEventType.SIMULATION_RUN)
        self.assertEqual(len(sims), 1)
        self.assertEqual(sims[0].event_type, UsageEventType.SIMULATION_RUN)

    def test_export_jsonl_valid_json_lines(self):
        _populate_tracker(self.tracker, "pln", 5)
        jsonl = self.tracker.export_jsonl("pln")
        for line in jsonl.strip().split("\n"):
            obj = json.loads(line)
            self.assertIn("event_id", obj)
            self.assertIn("event_type", obj)

    def test_export_bi_dict_structure(self):
        _populate_tracker(self.tracker, "bri", 15)
        bi = self.tracker.export_bi_dict("bri")
        for key in ["tenant_id", "plan", "snapshot", "event_type_dist", "daily_series", "top_users"]:
            self.assertIn(key, bi)

    def test_bi_dict_daily_series_is_list(self):
        _populate_tracker(self.tracker, "mandiri", 5)
        bi = self.tracker.export_bi_dict("mandiri")
        self.assertIsInstance(bi["daily_series"], list)
        self.assertGreater(len(bi["daily_series"]), 0)

    def test_health_returns_status(self):
        h = self.tracker.health()
        self.assertEqual(h["status"], "ok")
        self.assertIn("tenants", h)
        self.assertIn("total_events", h)

    def test_all_tenant_ids_returned(self):
        self.tracker.track(UsageEventType.LOGIN, "co_a", "u01")
        self.tracker.track(UsageEventType.LOGIN, "co_b", "u01")
        ids = self.tracker.get_all_tenant_ids()
        self.assertIn("co_a", ids)
        self.assertIn("co_b", ids)

    def test_plan_quotas_contain_required_fields(self):
        for plan, cfg in PLAN_QUOTAS.items():
            for key in ["price_usd_month", "simulations", "api_calls"]:
                self.assertIn(key, cfg, f"Plan '{plan}' missing '{key}'")

    def test_overage_detected_when_quota_exceeded(self):
        tracker = _make_tracker()
        tracker.set_plan("startup", "starter")
        # starter = 50 sims included
        for _ in range(60):
            tracker.track(UsageEventType.SIMULATION_RUN, "startup", "u01")
        snap = tracker.get_snapshot("startup")
        self.assertEqual(snap.simulations_used, 60)
        self.assertGreater(snap.overage_units, 0)
        self.assertGreater(snap.overage_cost_usd, 0)


# ===========================================================================
# [2] BILLING SERVICE
# ===========================================================================

class TestBillingService(unittest.TestCase):

    def setUp(self):
        self.billing = BillingService()

    def test_onboard_tenant_returns_subscription(self):
        sub = self.billing.onboard_tenant("telkom", "enterprise", "ict@telkom.co.id")
        self.assertIsInstance(sub, Subscription)
        self.assertEqual(sub.tenant_id, "telkom")
        self.assertEqual(sub.plan, "enterprise")

    def test_onboarded_tenant_has_trial_status(self):
        sub = self.billing.onboard_tenant("pln", "growth", trial_days=14)
        self.assertEqual(sub.status, BillingStatus.TRIALING)
        self.assertIsNotNone(sub.trial_end)

    def test_subscription_no_trial(self):
        sub = self.billing.onboard_tenant("bni2", "enterprise", trial_days=0)
        self.assertEqual(sub.status, BillingStatus.ACTIVE)

    def test_generate_invoice_returns_invoice(self):
        self.billing.onboard_tenant("pertamina", "enterprise")
        for _ in range(5):
            self.billing.record_usage("pertamina", "u01", "cus_xxx", UsageEventType.SIMULATION_RUN)
        inv = self.billing.generate_invoice("pertamina", 2026, 3)
        self.assertIsInstance(inv, Invoice)
        self.assertEqual(inv.tenant_id, "pertamina")

    def test_invoice_has_grand_total(self):
        self.billing.onboard_tenant("pdam", "growth")
        self.billing.record_usage("pdam", "u01", "cus_1", UsageEventType.ESG_BADGE_MINT)
        inv = self.billing.generate_invoice("pdam", 2026, 3)
        self.assertGreater(inv.grand_total, 0)

    def test_invoice_includes_tax(self):
        self.billing.onboard_tenant("bca", "enterprise")
        self.billing.record_usage("bca", "u01", "cus_1", UsageEventType.SIMULATION_RUN, units=10)
        inv = self.billing.generate_invoice("bca", 2026, 3)
        self.assertGreater(inv.tax_amount, 0)

    def test_invoice_has_provenance_hash(self):
        self.billing.onboard_tenant("bri", "growth")
        inv = self.billing.generate_invoice("bri", 2026, 3)
        self.assertIsNotNone(inv.provenance_hash)
        self.assertIsInstance(inv.provenance_hash, str)
        self.assertGreater(len(inv.provenance_hash), 32)

    def test_invoice_html_contains_invoice_id(self):
        self.billing.onboard_tenant("mandiri", "enterprise")
        inv = self.billing.generate_invoice("mandiri", 2026, 3)
        html = self.billing.invoice_html(inv.invoice_id)
        self.assertIn(inv.invoice_id, html)

    def test_invoice_html_is_valid_html(self):
        self.billing.onboard_tenant("cimb", "growth")
        inv = self.billing.generate_invoice("cimb", 2026, 3)
        html = self.billing.invoice_html(inv.invoice_id)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)

    def test_upgrade_tenant_plan(self):
        self.billing.onboard_tenant("bnii", "starter")
        sub = self.billing.upgrade_tenant("bnii", "enterprise")
        self.assertEqual(sub.plan, "enterprise")

    def test_cancel_tenant_returns_true(self):
        self.billing.onboard_tenant("xyz_corp", "growth")
        result = self.billing.cancel_tenant("xyz_corp")
        self.assertTrue(result)

    def test_cancel_nonexistent_tenant_returns_false(self):
        result = self.billing.cancel_tenant("ghost_corp_that_never_existed")
        self.assertFalse(result)

    def test_get_snapshot_returns_correct_plan(self):
        self.billing.onboard_tenant("gojek", "government")
        snap = self.billing.get_snapshot("gojek")
        self.assertEqual(snap.plan, "government")

    def test_bi_export_contains_daily_series(self):
        self.billing.onboard_tenant("grab", "enterprise")
        _populate_tracker(self.billing._tracker, "grab", 10)
        bi = self.billing.bi_export("grab")
        self.assertIn("daily_series", bi)

    def test_billing_health_has_stripe_flag(self):
        h = self.billing.health()
        self.assertIn("stripe_available", h)
        self.assertIn("subscriptions", h)

    def test_webhook_handler_handles_valid_json(self):
        payload = json.dumps({
            "type": "customer.subscription.updated",
            "data": {"object": {
                "id": "sub_xxx",
                "status": "active",
                "metadata": {"tenant_id": "telkom", "plan": "enterprise"}
            }}
        }).encode()
        result = self.billing.handle_webhook(payload, "")
        self.assertEqual(result["status"], "ok")

    def test_webhook_rejects_invalid_json(self):
        result = self.billing.handle_webhook(b"not json at all {", "sig_xxx")
        self.assertEqual(result["status"], "error")


# ===========================================================================
# [3] INSIGHTS ENGINE
# ===========================================================================

class TestInsightsEngine(unittest.TestCase):

    def setUp(self):
        self.engine = InsightsEngine()

    def _generate(self, **kwargs) -> list:
        defaults = dict(
            cos_score=0.79, pue=1.82, wue=1.65,
            carbon_intensity_kg_kwh=0.748, renewable_pct=0.0,
            tenant_region="jawa_bali", annual_kwh=50_000_000,
        )
        defaults.update(kwargs)
        return self.engine.generate(**defaults)

    def test_generate_returns_list(self):
        insights = self._generate()
        self.assertIsInstance(insights, list)

    def test_generate_returns_insights_when_issues_present(self):
        insights = self._generate(pue=1.9, renewable_pct=0.0, cos_score=0.6)
        self.assertGreater(len(insights), 0)

    def test_insights_are_insight_instances(self):
        insights = self._generate()
        for i in insights:
            self.assertIsInstance(i, Insight)

    def test_insights_sorted_critical_first(self):
        insights = self._generate(pue=2.0, renewable_pct=0.0, cos_score=0.5)
        priorities = [i.priority.value for i in insights]
        order_map  = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for j in range(len(priorities) - 1):
            self.assertLessEqual(
                order_map[priorities[j]], order_map[priorities[j + 1]]
            )

    def test_no_insights_for_perfect_datacenter(self):
        """A near-perfect datacenter should produce zero critical insights."""
        insights = self._generate(
            cos_score=0.95, pue=1.2, wue=1.3,
            carbon_intensity_kg_kwh=0.03, renewable_pct=1.0,
            compliance_scores={"GDPR": 0.92, "ISO27001": 0.90},
        )
        critical = [i for i in insights if i.priority == InsightPriority.CRITICAL]
        self.assertEqual(len(critical), 0)

    def test_renewable_insight_when_zero_renewable(self):
        insights = self._generate(renewable_pct=0.0)
        categories = [i.category for i in insights]
        self.assertIn(InsightCategory.RENEWABLE_ENERGY, categories)

    def test_no_renewable_insight_when_full_renewable(self):
        insights = self._generate(renewable_pct=1.0)
        categories = [i.category for i in insights]
        self.assertNotIn(InsightCategory.RENEWABLE_ENERGY, categories)

    def test_cooling_insight_for_high_pue(self):
        insights = self._generate(pue=2.1)
        categories = [i.category for i in insights]
        self.assertIn(InsightCategory.COOLING_EFFICIENCY, categories)

    def test_no_cooling_insight_for_good_pue(self):
        insights = self._generate(pue=1.15)
        cooling = [i for i in insights if i.category == InsightCategory.COOLING_EFFICIENCY]
        self.assertEqual(len(cooling), 0)

    def test_ai_governance_insight_for_low_cos(self):
        insights = self._generate(cos_score=0.55)
        categories = [i.category for i in insights]
        self.assertIn(InsightCategory.AI_GOVERNANCE, categories)

    def test_compliance_insight_when_score_low(self):
        insights = self._generate(compliance_scores={"GDPR": 0.30, "ISO27001": 0.40})
        categories = [i.category for i in insights]
        self.assertIn(InsightCategory.COMPLIANCE_GAP, categories)

    def test_insight_estimated_saving_non_negative(self):
        insights = self._generate()
        for i in insights:
            self.assertGreaterEqual(i.estimated_saving_usd_year, 0.0)

    def test_insight_has_action_text(self):
        insights = self._generate()
        for i in insights:
            self.assertIsInstance(i.action, str)
            self.assertGreater(len(i.action), 20)

    def test_insight_has_tags(self):
        insights = self._generate(renewable_pct=0.0)
        renewable_insights = [i for i in insights if i.category == InsightCategory.RENEWABLE_ENERGY]
        for i in renewable_insights:
            self.assertIsInstance(i.tags, list)
            self.assertGreater(len(i.tags), 0)

    def test_renewable_options_database_not_empty(self):
        self.assertGreater(len(RENEWABLE_OPTIONS_ID), 0)

    def test_grid_carbon_intensity_has_jawa_bali(self):
        self.assertIn("jawa_bali", GRID_CARBON_INTENSITY)
        self.assertGreater(GRID_CARBON_INTENSITY["jawa_bali"], 0)


# ===========================================================================
# [4] ISO 42001 ASSESSMENT
# ===========================================================================

class TestISO42001Assessor(unittest.TestCase):

    def setUp(self):
        self.assessor = ISO42001Assessor()

    def _assess(self, **kwargs) -> ISO42001Assessment:
        defaults = dict(
            tenant_id="telkom",
            cos_score=0.94,
            swarm_approved=True,
            swarm_consensus=0.93,
            provenance_chain_events=12,
            audit_log_entries=87,
            rbac_enabled=True,
            encryption_enabled=True,
            compliance_score_iso27001=0.79,
            has_incident_response=True,
        )
        defaults.update(kwargs)
        return self.assessor.assess(**defaults)

    def test_returns_iso42001_assessment(self):
        result = self._assess()
        self.assertIsInstance(result, ISO42001Assessment)

    def test_high_cos_high_swarm_gives_high_score(self):
        result = self._assess(cos_score=0.95, swarm_approved=True, swarm_consensus=0.95)
        self.assertGreater(result.overall_score, 0.70)

    def test_certification_ready_when_high_score(self):
        result = self._assess()
        self.assertTrue(result.certification_ready)

    def test_not_certified_when_low_cos(self):
        result = self._assess(
            cos_score=0.40, swarm_approved=False, swarm_consensus=0.50,
            provenance_chain_events=0, audit_log_entries=0,
            rbac_enabled=False, encryption_enabled=False,
            compliance_score_iso27001=0.30, has_incident_response=False
        )
        self.assertFalse(result.certification_ready)

    def test_gaps_list_when_issues(self):
        result = self._assess(
            cos_score=0.55, swarm_approved=False, provenance_chain_events=0,
            audit_log_entries=0, encryption_enabled=False
        )
        self.assertIsInstance(result.gaps, list)
        # Should have at least one gap
        self.assertGreater(len(result.gaps), 0)

    def test_recommendations_match_gaps(self):
        result = self._assess(
            cos_score=0.55, swarm_approved=False, provenance_chain_events=0
        )
        self.assertEqual(len(result.gaps), len(result.recommendations))

    def test_all_clauses_in_0_1_range(self):
        result = self._assess()
        for clause_score in [
            result.clause_4_context, result.clause_5_leadership,
            result.clause_6_planning, result.clause_7_support,
            result.clause_8_operation, result.clause_9_performance,
            result.clause_10_improvement,
        ]:
            self.assertGreaterEqual(clause_score, 0.0)
            self.assertLessEqual(clause_score, 1.0)

    def test_overall_score_in_0_1_range(self):
        result = self._assess()
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_caterya_cos_evidence_stored(self):
        result = self._assess(cos_score=0.8765)
        self.assertAlmostEqual(result.caterya_cos_evidence, 0.8765, places=4)

    def test_tenant_id_preserved(self):
        result = self._assess(tenant_id="pertamina")
        self.assertEqual(result.tenant_id, "pertamina")


# ===========================================================================
# [5] BI EXPORTER
# ===========================================================================

class TestBIExporter(unittest.TestCase):

    def setUp(self):
        self.exporter = BIExporter()
        self.data = _make_esg_data("telkom")

    def test_to_csv_returns_string(self):
        csv_out = self.exporter.to_csv(self.data)
        self.assertIsInstance(csv_out, str)
        self.assertGreater(len(csv_out), 0)

    def test_to_csv_has_header_row(self):
        csv_out = self.exporter.to_csv(self.data)
        lines = csv_out.strip().split("\n")
        self.assertGreater(len(lines), 1)
        header = lines[0]
        self.assertIn("tenant_id", header)
        self.assertIn("cos_composite", header)

    def test_to_csv_has_data_row(self):
        csv_out = self.exporter.to_csv(self.data)
        lines = csv_out.strip().split("\n")
        self.assertEqual(len(lines), 2)  # header + 1 data row

    def test_to_insights_csv_one_row_per_insight(self):
        csv_out = self.exporter.to_insights_csv(self.data)
        lines = [l for l in csv_out.strip().split("\n") if l]
        # header + N insight rows
        self.assertEqual(len(lines), len(self.data.insights) + 1)

    def test_to_json_schema_has_required_sections(self):
        jschema = self.exporter.to_json_schema(self.data)
        for key in ["schema_version", "tenant_id", "ai_governance", "energy",
                    "emissions", "compliance", "insights", "provenance"]:
            self.assertIn(key, jschema)

    def test_json_schema_cos_matches(self):
        jschema = self.exporter.to_json_schema(self.data)
        self.assertAlmostEqual(
            jschema["ai_governance"]["cos_composite"],
            self.data.cos_composite, places=4
        )

    def test_json_schema_energy_pue_rating(self):
        jschema = self.exporter.to_json_schema(self.data)
        self.assertIn(jschema["energy"]["pue_rating"], ["excellent", "good", "average", "poor"])

    def test_json_schema_compliance_has_iso42001(self):
        jschema = self.exporter.to_json_schema(self.data)
        self.assertIn("iso_42001", jschema["compliance"])
        iso_data = jschema["compliance"]["iso_42001"]
        self.assertIsNotNone(iso_data)
        self.assertIn("overall_score", iso_data)
        self.assertIn("clause_scores", iso_data)

    def test_json_schema_insights_list(self):
        jschema = self.exporter.to_json_schema(self.data)
        self.assertIsInstance(jschema["insights"], list)

    def test_json_schema_emissions_scope2(self):
        jschema = self.exporter.to_json_schema(self.data)
        self.assertIn("scope_2_tonnes_co2e", jschema["emissions"])

    def test_html_report_is_html(self):
        html = self.exporter.to_html_report(self.data)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)

    def test_html_report_contains_tenant_id(self):
        html = self.exporter.to_html_report(self.data)
        self.assertIn("telkom", html)

    def test_html_report_contains_kpis(self):
        html = self.exporter.to_html_report(self.data)
        # Should show PUE
        self.assertIn(f"{self.data.pue:.2f}", html)

    def test_html_report_contains_insights(self):
        html = self.exporter.to_html_report(self.data)
        # At least one insight title should appear
        if self.data.insights:
            snippet = self.data.insights[0].title[:30]
            self.assertIn(snippet, html)

    def test_html_report_is_printable(self):
        html = self.exporter.to_html_report(self.data)
        self.assertIn("@media print", html)

    def test_json_schema_serialisable(self):
        jschema = self.exporter.to_json_schema(self.data)
        serialised = json.dumps(jschema)
        self.assertGreater(len(serialised), 100)


# ===========================================================================
# [6] END-TO-END INTEGRATION TESTS
# ===========================================================================

class TestE2EMonetisationPipeline(unittest.TestCase):
    """
    Full Telkom / Pertamina pilot journey:
    onboard → simulate → track usage → invoice → ESG report → ISO 42001
    """

    def setUp(self):
        self.billing = BillingService()
        self.engine  = InsightsEngine()
        self.assessor = ISO42001Assessor()
        self.exporter = BIExporter()

    def test_e2e_telkom_pilot_journey(self):
        # 1. Onboard Telkom
        sub = self.billing.onboard_tenant(
            "telkom_pilot", "enterprise",
            email="datacenter@telkom.co.id",
            trial_days=14,
        )
        self.assertEqual(sub.plan, "enterprise")

        # 2. Record usage (simulating 1 month of activity)
        for _ in range(10):
            self.billing.record_usage(
                "telkom_pilot", "analyst01", sub.customer_id,
                UsageEventType.SIMULATION_RUN
            )
        for _ in range(50):
            self.billing.record_usage(
                "telkom_pilot", "analyst01", sub.customer_id,
                UsageEventType.API_CALL
            )
        self.billing.record_usage("telkom_pilot", "compliance01", sub.customer_id,
                                  UsageEventType.COMPLIANCE_SCAN)
        self.billing.record_usage("telkom_pilot", "analyst01", sub.customer_id,
                                  UsageEventType.ESG_BADGE_MINT)

        # 3. Check snapshot
        snap = self.billing.get_snapshot("telkom_pilot")
        self.assertEqual(snap.simulations_used, 10)
        self.assertEqual(snap.api_calls_used, 50)
        self.assertEqual(snap.badges_minted, 1)

        # 4. Generate invoice
        inv = self.billing.generate_invoice("telkom_pilot", 2026, 3)
        self.assertGreater(inv.grand_total, 0)
        self.assertGreater(inv.subscription_fee, 0)
        self.assertIsNotNone(inv.provenance_hash)

        # 5. Generate ESG insights
        insights = self.engine.generate(
            cos_score=0.86, pue=1.72, wue=1.58,
            carbon_intensity_kg_kwh=0.748, renewable_pct=0.05,
            tenant_region="jawa_bali", annual_kwh=120_000_000,
            compliance_scores={"ISO27001": 0.79, "SRN_PPI": 0.67},
        )
        self.assertGreater(len(insights), 0)

        # 6. ISO 42001 assessment
        iso = self.assessor.assess(
            tenant_id="telkom_pilot",
            cos_score=0.86, swarm_approved=True, swarm_consensus=0.88,
            provenance_chain_events=10, audit_log_entries=50,
            compliance_score_iso27001=0.79,
        )
        self.assertGreaterEqual(iso.overall_score, 0.0)

        # 7. BI export
        esg_data = ESGReportData(
            tenant_id="telkom_pilot", report_type="quarterly", period="Q1-2026",
            cos_composite=0.86, cos_swarm_approved=True,
            pue=1.72, wue=1.58, carbon_total_tonnes=89_760.0, carbon_intensity=0.748,
            renewable_pct=0.05, insights=insights, iso42001=iso,
            compliance_scores={"ISO27001": 0.79, "SRN_PPI": 0.67},
            provenance_hashes=[inv.provenance_hash],
            badge_ids=["badge-telkom-q1"],
        )
        json_schema = self.exporter.to_json_schema(esg_data)
        html_report = self.exporter.to_html_report(esg_data)
        csv_summary = self.exporter.to_csv(esg_data)

        self.assertEqual(json_schema["tenant_id"], "telkom_pilot")
        self.assertIn("telkom_pilot", html_report)
        self.assertGreater(len(csv_summary), 0)

    def test_e2e_pertamina_government_plan(self):
        """Government plan: unlimited quota, no overage."""
        sub = self.billing.onboard_tenant(
            "pertamina_pilot", "government",
            email="esg@pertamina.com",
        )
        self.assertEqual(sub.plan, "government")

        # Record heavy usage — should not generate overage on gov plan
        for _ in range(1000):
            self.billing.record_usage(
                "pertamina_pilot", "system", sub.customer_id,
                UsageEventType.SIMULATION_RUN
            )
        snap = self.billing.get_snapshot("pertamina_pilot")
        self.assertEqual(snap.overage_units, 0.0)  # government = unlimited
        self.assertEqual(snap.overage_cost_usd, 0.0)

    def test_e2e_multi_tenant_isolation(self):
        """Two tenants don't see each other's data."""
        self.billing.onboard_tenant("bni_prod", "enterprise")
        self.billing.onboard_tenant("bca_prod", "growth")

        for _ in range(5):
            self.billing.record_usage("bni_prod", "u01", "cus_bni",
                                      UsageEventType.SIMULATION_RUN)
        for _ in range(3):
            self.billing.record_usage("bca_prod", "u01", "cus_bca",
                                      UsageEventType.API_CALL)

        snap_bni = self.billing.get_snapshot("bni_prod")
        snap_bca = self.billing.get_snapshot("bca_prod")

        self.assertEqual(snap_bni.simulations_used, 5)
        self.assertEqual(snap_bca.simulations_used, 0)
        self.assertEqual(snap_bni.api_calls_used, 0)
        self.assertEqual(snap_bca.api_calls_used, 3)

    def test_e2e_upgrade_and_reinvoice(self):
        """Upgrade from Starter to Enterprise mid-month."""
        self.billing.onboard_tenant("startup_x", "starter")
        self.billing.record_usage("startup_x", "u01", "cus_1",
                                  UsageEventType.SIMULATION_RUN, units=5)

        # Upgrade
        sub = self.billing.upgrade_tenant("startup_x", "enterprise")
        self.assertEqual(sub.plan, "enterprise")

        # Invoice should reflect enterprise pricing
        inv = self.billing.generate_invoice("startup_x", 2026, 3)
        self.assertEqual(inv.plan, "enterprise")

    def test_e2e_quarterly_reporting_all_formats(self):
        """Generate Q1 report in all 4 output formats."""
        data = _make_esg_data("quarterly_test")

        csv_main     = self.exporter.to_csv(data)
        csv_insights = self.exporter.to_insights_csv(data)
        json_schema  = self.exporter.to_json_schema(data)
        html_report  = self.exporter.to_html_report(data)

        self.assertTrue(csv_main.startswith("tenant_id"))
        self.assertTrue(csv_insights.startswith("tenant_id"))
        self.assertEqual(json_schema["schema_version"], "eco-ai-esg-v2")
        self.assertIn("ESG Sustainability Report", html_report)


# ===========================================================================
# [7] CATERYA FRAMEWORK INTEGRATION
# ===========================================================================

class TestCATERYAIntegrationWithMonetisation(unittest.TestCase):
    """
    Verify CATERYA COS + EthicsSwarm + ProvenanceChain integrate correctly
    with the monetisation and reporting pipeline.
    """

    def setUp(self):
        from caterya_framework.evaluator import CATERYAEvaluator
        from caterya_framework.swarm import EthicsSwarm
        from caterya_framework.provenance import ProvenanceChain
        self.evaluator = CATERYAEvaluator()
        self.swarm     = EthicsSwarm(n_agents=5, threshold=0.7)
        self.prov      = ProvenanceChain(model_id="e2e-test")

        rng = np.random.default_rng(42)
        self.df = pd.DataFrame({
            "power_kw":                    rng.normal(1200, 100, 80).clip(800, 2000),
            "it_power_kw":                 rng.normal(700,   60, 80).clip(400, 1200),
            "water_usage_litres":          rng.normal(500,   40, 80).clip(200, 900),
            "carbon_intensity_kg_per_kwh": rng.normal(0.28, 0.07, 80).clip(0.05, 0.6),
        })

    def test_caterya_cos_used_in_iso42001(self):
        cos = self.evaluator.score_only(self.df)
        swarm_result = self.swarm.deliberate(cos)
        assessor = ISO42001Assessor()
        iso = assessor.assess(
            tenant_id="caterya_test",
            cos_score=cos.composite,
            swarm_approved=swarm_result["approved"],
            swarm_consensus=swarm_result["consensus_score"],
            provenance_chain_events=5,
            audit_log_entries=30,
        )
        self.assertAlmostEqual(iso.caterya_cos_evidence, cos.composite, places=4)
        self.assertIsInstance(iso.certification_ready, (bool, np.bool_))

    def test_caterya_provenance_hash_in_esg_report(self):
        cos = self.evaluator.score_only(self.df)
        h = self.prov.record("cos_evaluation", {"cos": cos.composite}, actor="analyst")
        self.assertTrue(self.prov.verify())

        engine = InsightsEngine()
        insights = engine.generate(
            cos_score=cos.composite, pue=1.69, wue=1.43,
            carbon_intensity_kg_kwh=0.748, renewable_pct=0.10,
            annual_kwh=50_000_000,
        )
        data = ESGReportData(
            tenant_id="caterya_test", report_type="quarterly", period="Q1-2026",
            cos_composite=cos.composite, cos_swarm_approved=True,
            pue=1.69, wue=1.43, carbon_total_tonnes=37_400.0, carbon_intensity=0.748,
            renewable_pct=0.10, insights=insights,
            provenance_hashes=[h],
        )
        exp = BIExporter()
        jschema = exp.to_json_schema(data)
        self.assertIn(h, jschema["provenance"]["hashes"])

    def test_caterya_cos_to_billing_snapshot(self):
        cos = self.evaluator.score_only(self.df)
        tracker = UsageTracker()
        tracker.track(UsageEventType.COS_EVALUATION, "caterya_test", "system",
                      metadata={"cos_composite": cos.composite})
        daily = tracker.get_daily_summary("caterya_test")
        self.assertGreater(daily.total_events, 0)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestUsageAnalytics,
        TestBillingService,
        TestInsightsEngine,
        TestISO42001Assessor,
        TestBIExporter,
        TestE2EMonetisationPipeline,
        TestCATERYAIntegrationWithMonetisation,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print(f"\n{'━'*60}")
    print(f" Results: {passed}/{total} passed | {failed} failed")
    if failed == 0:
        print(" ALL MONETISATION + E2E TESTS PASSED ✅")
        print(" Enterprise-ready: Telkom · Pertamina · BNI · PLN")
    else:
        print(f" {failed} test(s) failed — review output above")
    print(f"{'━'*60}")

    import sys
    sys.exit(0 if failed == 0 else 1)
