"""
reports/insights.py
====================
AI-driven sustainability insights engine.

Generates actionable recommendations from COS scores, PUE/WUE trends,
and compliance data — including renewable energy switching advice,
carbon reduction roadmaps, and ISO 42001 AI certification assessment.

Also provides:
  - BI-ready data exports (Tableau / Power BI / Looker)
  - Custom ESG report templates (quarterly, annual, regulatory)
  - ISO 42001 AI Management System certification checklist via CATERYA

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("eco_ai.reports.insights")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class InsightCategory(str, Enum):
    RENEWABLE_ENERGY   = "renewable_energy"
    COOLING_EFFICIENCY = "cooling_efficiency"
    CARBON_REDUCTION   = "carbon_reduction"
    WATER_CONSERVATION = "water_conservation"
    AI_GOVERNANCE      = "ai_governance"
    COMPLIANCE_GAP     = "compliance_gap"
    COST_SAVINGS       = "cost_savings"

class InsightPriority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

@dataclass
class Insight:
    category:     InsightCategory
    priority:     InsightPriority
    title:        str
    description:  str
    impact:       str
    action:       str
    estimated_saving_usd_year: float = 0.0
    estimated_carbon_reduction_pct: float = 0.0
    cos_improvement_delta:    float = 0.0
    data_sources:  List[str] = field(default_factory=list)
    tags:          List[str] = field(default_factory=list)


@dataclass
class ISO42001Assessment:
    """ISO/IEC 42001:2023 AI Management System certification readiness."""
    tenant_id:     str
    assessed_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Clause scores (0.0–1.0)
    clause_4_context:       float = 0.0   # Context of the organisation
    clause_5_leadership:    float = 0.0   # Leadership
    clause_6_planning:      float = 0.0   # Planning
    clause_7_support:       float = 0.0   # Support
    clause_8_operation:     float = 0.0   # Operation
    clause_9_performance:   float = 0.0   # Performance evaluation
    clause_10_improvement:  float = 0.0   # Improvement
    overall_score:          float = 0.0
    certification_ready:    bool  = False
    gaps:                   List[str] = field(default_factory=list)
    recommendations:        List[str] = field(default_factory=list)
    caterya_cos_evidence:   float = 0.0   # COS score used as evidence


@dataclass
class ESGReportData:
    tenant_id:          str
    report_type:        str     # quarterly / annual / regulatory / pilot
    period:             str
    cos_composite:      float
    cos_swarm_approved: bool
    pue:                float
    wue:                float
    carbon_total_tonnes: float
    carbon_intensity:   float
    renewable_pct:      float
    insights:           List[Insight] = field(default_factory=list)
    iso42001:           Optional[ISO42001Assessment] = None
    compliance_scores:  Dict[str, float] = field(default_factory=dict)
    provenance_hashes:  List[str] = field(default_factory=list)
    badge_ids:          List[str] = field(default_factory=list)
    generated_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Renewable energy database (Indonesian grid + major providers)
# ---------------------------------------------------------------------------

RENEWABLE_OPTIONS_ID = [
    {
        "provider": "PLN EBT (Energi Baru Terbarukan)",
        "source": "hydro + geothermal",
        "carbon_intensity_kg_kwh": 0.042,
        "availability": "Jawa-Bali, Sumatra",
        "contract_type": "PPA (Power Purchase Agreement)",
        "estimated_cost_idr_kwh": 1_050,
        "url": "https://www.pln.co.id/ebt",
    },
    {
        "provider": "PLN Surya (Solar)",
        "source": "solar PV",
        "carbon_intensity_kg_kwh": 0.028,
        "availability": "Nasional",
        "contract_type": "PLTS Atap / SPKLU",
        "estimated_cost_idr_kwh": 1_200,
        "url": "https://www.pln.co.id/plts",
    },
    {
        "provider": "Pertamina Geothermal",
        "source": "geothermal",
        "carbon_intensity_kg_kwh": 0.038,
        "availability": "Sumatra, Jawa, Sulawesi",
        "contract_type": "Direct offtake / IPP",
        "estimated_cost_idr_kwh": 980,
        "url": "https://www.pertaminageothermal.com",
    },
    {
        "provider": "UPC Renewables Indonesia",
        "source": "wind + solar",
        "carbon_intensity_kg_kwh": 0.022,
        "availability": "Jawa, Kalimantan",
        "contract_type": "VPPA / Bundled REC",
        "estimated_cost_idr_kwh": 1_150,
        "url": "https://upcrenewables.com",
    },
]

# Global average grid carbon intensity by region
GRID_CARBON_INTENSITY = {
    "jawa_bali":  0.748,   # kg CO₂/kWh — Indonesian grid
    "sumatra":    0.691,
    "kalimantan": 0.812,
    "default":    0.748,
}


# ---------------------------------------------------------------------------
# InsightsEngine
# ---------------------------------------------------------------------------

class InsightsEngine:
    """
    AI-driven recommendations based on datacenter metrics.

    Usage
    -----
    engine = InsightsEngine()
    insights = engine.generate(
        cos_score=0.79,
        pue=1.82,
        wue=1.65,
        carbon_intensity_kg_kwh=0.748,
        server_utilization_pct=61.0,
        renewable_pct=0.0,
        tenant_region="jawa_bali",
        annual_kwh=50_000_000,
    )
    for insight in insights:
        print(f"[{insight.priority.value}] {insight.title}")
    """

    def generate(
        self,
        cos_score: float,
        pue: float,
        wue: float,
        carbon_intensity_kg_kwh: float,
        server_utilization_pct: float = 70.0,
        renewable_pct: float = 0.0,
        tenant_region: str = "default",
        annual_kwh: float = 10_000_000,
        compliance_scores: Optional[Dict[str, float]] = None,
    ) -> List[Insight]:
        insights: List[Insight] = []

        insights += self._renewable_insights(
            renewable_pct, carbon_intensity_kg_kwh, tenant_region, annual_kwh
        )
        insights += self._cooling_insights(pue, annual_kwh)
        insights += self._carbon_insights(
            carbon_intensity_kg_kwh, annual_kwh, renewable_pct, tenant_region
        )
        insights += self._water_insights(wue)
        insights += self._cos_insights(cos_score)
        if compliance_scores:
            insights += self._compliance_insights(compliance_scores)

        # Sort: critical first, then by estimated saving
        insights.sort(
            key=lambda x: (
                ["critical", "high", "medium", "low"].index(x.priority.value),
                -x.estimated_saving_usd_year,
            )
        )
        return insights

    # ── Renewable energy ─────────────────────────────────────────────────

    def _renewable_insights(
        self,
        renewable_pct: float,
        current_intensity: float,
        region: str,
        annual_kwh: float,
    ) -> List[Insight]:
        out = []
        grid_intensity = GRID_CARBON_INTENSITY.get(region, GRID_CARBON_INTENSITY["default"])

        if renewable_pct < 0.30:
            # Find cheapest renewable option for this region
            best = min(RENEWABLE_OPTIONS_ID, key=lambda r: r["carbon_intensity_kg_kwh"])
            carbon_saved_tonne = (current_intensity - best["carbon_intensity_kg_kwh"]) * annual_kwh / 1000
            # Rough cost saving: carbon credits at $12/tonne (REDD+ Indonesia)
            credit_saving = carbon_saved_tonne * 12
            out.append(Insight(
                category=InsightCategory.RENEWABLE_ENERGY,
                priority=InsightPriority.CRITICAL if renewable_pct == 0 else InsightPriority.HIGH,
                title="Switch to Renewable Energy — Immediate Carbon Win",
                description=(
                    f"Your current grid mix has a carbon intensity of "
                    f"{current_intensity:.3f} kg CO₂/kWh. Switching to "
                    f"{best['provider']} ({best['source']}) would reduce this to "
                    f"{best['carbon_intensity_kg_kwh']:.3f} kg CO₂/kWh — a "
                    f"{((current_intensity - best['carbon_intensity_kg_kwh']) / current_intensity * 100):.1f}% reduction."
                ),
                impact=(
                    f"~{carbon_saved_tonne:,.0f} tonnes CO₂ avoided per year "
                    f"at {annual_kwh/1e6:.1f} GWh annual consumption."
                ),
                action=(
                    f"Contact {best['provider']} for a PPA pilot. "
                    f"Start with 20% renewable procurement in Q2, target 100% by year-end. "
                    f"Register with SRN PPI for carbon credit issuance."
                ),
                estimated_saving_usd_year=round(credit_saving, 2),
                estimated_carbon_reduction_pct=round(
                    (current_intensity - best["carbon_intensity_kg_kwh"]) / current_intensity * 100, 1
                ),
                data_sources=["grid_data", "srn_ppi", "pertamina_geothermal"],
                tags=["renewable", "carbon", "ppa", "srn_ppi"],
            ))

        if 0.0 < renewable_pct < 1.0:
            target_saving = (1.0 - renewable_pct) * annual_kwh * 0.7 * 12 / 1000
            out.append(Insight(
                category=InsightCategory.RENEWABLE_ENERGY,
                priority=InsightPriority.MEDIUM,
                title=f"Increase Renewable Mix from {renewable_pct:.0%} to 100%",
                description=(
                    f"You're already at {renewable_pct:.0%} renewable — great start. "
                    "Full RE100 commitment unlocks better ESG ratings and access to "
                    "green bond financing (OJK Green Taxonomy 2024)."
                ),
                impact="Full decarbonisation, RE100 compliance, improved investor ESG scoring.",
                action="Negotiate expanded PPA or add on-site solar + battery storage.",
                estimated_saving_usd_year=round(target_saving, 2),
                data_sources=["renewable_tracker"],
                tags=["re100", "green_bond", "ojk"],
            ))

        return out

    # ── Cooling efficiency ────────────────────────────────────────────────

    def _cooling_insights(self, pue: float, annual_kwh: float) -> List[Insight]:
        out = []
        if pue > 1.8:
            waste_kwh    = (pue - 1.2) / pue * annual_kwh
            saving_usd   = waste_kwh * 0.08  # $0.08/kWh average
            out.append(Insight(
                category=InsightCategory.COOLING_EFFICIENCY,
                priority=InsightPriority.CRITICAL,
                title=f"PUE {pue:.2f} — Critical Cooling Inefficiency",
                description=(
                    f"Industry best practice is PUE ≤ 1.2 for tropical datacenters "
                    f"(ASHRAE TC 9.9). Your PUE of {pue:.2f} means "
                    f"{((pue - 1) / pue * 100):.0f}% of all power goes to cooling overhead."
                ),
                impact=f"${saving_usd:,.0f}/year savings if PUE reduced to 1.2.",
                action=(
                    "1. Raise CRAC setpoints to 27°C (ASHRAE A1 class). "
                    "2. Implement hot-aisle/cold-aisle containment. "
                    "3. Enable economizer mode during Jakarta night hours (23:00–05:00). "
                    "4. Consider liquid cooling for GPU racks."
                ),
                estimated_saving_usd_year=round(saving_usd, 2),
                tags=["pue", "cooling", "ashrae", "capex"],
            ))
        elif pue > 1.5:
            waste_kwh  = (pue - 1.2) / pue * annual_kwh * 0.3
            saving_usd = waste_kwh * 0.08
            out.append(Insight(
                category=InsightCategory.COOLING_EFFICIENCY,
                priority=InsightPriority.HIGH,
                title=f"PUE {pue:.2f} — Optimise Cooling for 15–25% Reduction",
                description=(
                    f"PUE {pue:.2f} is average for Southeast Asian datacenters "
                    "but there's room to improve. Small operational changes can get "
                    "you to 1.3–1.4 within 3 months with zero capex."
                ),
                impact=f"Estimated ${saving_usd:,.0f}/year with operational improvements.",
                action=(
                    "Review airflow management, seal cable cutouts in raised floor, "
                    "set cooling setpoints dynamically based on external temperature."
                ),
                estimated_saving_usd_year=round(saving_usd, 2),
                tags=["pue", "cooling", "opex"],
            ))
        return out

    # ── Carbon reduction ──────────────────────────────────────────────────

    def _carbon_insights(
        self,
        carbon_intensity: float,
        annual_kwh: float,
        renewable_pct: float,
        region: str,
    ) -> List[Insight]:
        out = []
        annual_carbon_tonnes = carbon_intensity * annual_kwh / 1000
        target_intensity     = 0.10  # SRN PPI 2030 target

        if carbon_intensity > 0.4:
            reduction_pct = (carbon_intensity - target_intensity) / carbon_intensity * 100
            out.append(Insight(
                category=InsightCategory.CARBON_REDUCTION,
                priority=InsightPriority.HIGH,
                title="Carbon Intensity Exceeds SRN PPI 2030 Target",
                description=(
                    f"Current intensity: {carbon_intensity:.3f} kg CO₂/kWh. "
                    f"SRN PPI target: {target_intensity:.2f} kg CO₂/kWh by 2030. "
                    f"You need a {reduction_pct:.0f}% reduction over 5 years."
                ),
                impact=(
                    f"{annual_carbon_tonnes:,.0f} tonnes CO₂/year currently — "
                    f"subject to carbon tax at Rp 30,000/tonne (Perpres No. 98/2021)."
                ),
                action=(
                    "Develop a Science-Based Target (SBTi) aligned decarbonisation roadmap: "
                    "Year 1: 30% renewable. Year 2: PUE < 1.5. Year 3: 60% renewable. "
                    "Year 5: Net-zero with verified offsets."
                ),
                estimated_carbon_reduction_pct=round(reduction_pct, 1),
                data_sources=["srn_ppi", "perpres_98_2021"],
                tags=["carbon_tax", "srn_ppi", "sbti", "decarbonisation"],
            ))

        if renewable_pct == 0.0 and annual_carbon_tonnes > 5000:
            credit_cost = annual_carbon_tonnes * 12  # $12/tonne REDD+
            out.append(Insight(
                category=InsightCategory.CARBON_REDUCTION,
                priority=InsightPriority.MEDIUM,
                title="Register for Indonesia Carbon Market (IDX Carbon)",
                description=(
                    f"With {annual_carbon_tonnes:,.0f} tCO₂/year, your datacenter "
                    "qualifies for carbon credit issuance via IDX Carbon (launched 2023). "
                    "Reduction projects can earn tradeable credits under SRN PPI."
                ),
                impact=f"Potential ${credit_cost:,.0f}/year revenue from carbon credits.",
                action=(
                    "Register with KLHK (Ministry of Environment) via SRN-PPI portal. "
                    "Commission Measurement, Reporting & Verification (MRV) study. "
                    "List on IDX Carbon at https://idxcarbon.co.id"
                ),
                estimated_saving_usd_year=round(credit_cost * 0.3, 2),
                tags=["idx_carbon", "srn_ppi", "mrv", "carbon_credit"],
            ))

        return out

    # ── Water conservation ────────────────────────────────────────────────

    def _water_insights(self, wue: float) -> List[Insight]:
        out = []
        if wue > 1.8:
            out.append(Insight(
                category=InsightCategory.WATER_CONSERVATION,
                priority=InsightPriority.HIGH,
                title=f"WUE {wue:.2f} — Switch to Air-Side Economizers",
                description=(
                    f"Water Usage Effectiveness of {wue:.2f} is high. "
                    "Air-side economizers or closed-loop dry coolers can reduce "
                    "water consumption by 60–80% compared to evaporative cooling."
                ),
                impact="60–80% reduction in water consumption, lower BPJS/PDAM costs.",
                action=(
                    "Evaluate adiabatic cooling or indirect evaporative cooling systems. "
                    "In Jakarta, air-side economizers are viable 40% of the year (night hours)."
                ),
                tags=["wue", "water", "cooling"],
            ))
        return out

    # ── COS / AI governance ───────────────────────────────────────────────

    def _cos_insights(self, cos_score: float) -> List[Insight]:
        out = []
        if cos_score < 0.70:
            out.append(Insight(
                category=InsightCategory.AI_GOVERNANCE,
                priority=InsightPriority.CRITICAL,
                title=f"COS {cos_score:.4f} Below Threshold — AI Output Unsafe for Production",
                description=(
                    f"CATERYA Composite Optimisation Score of {cos_score:.4f} is below "
                    "the 0.70 threshold. AI recommendations should not be deployed without "
                    "human review. EthicsSwarm consensus not reached."
                ),
                impact="Risk of biased or unreliable optimization recommendations.",
                action=(
                    "Re-run evaluation with more data (≥100 rows). "
                    "Check for data quality issues in sensor feeds. "
                    "Review fairness and symmetry sub-scores."
                ),
                cos_improvement_delta=0.70 - cos_score,
                tags=["cos", "ai_governance", "ethics"],
            ))
        elif cos_score < 0.85:
            out.append(Insight(
                category=InsightCategory.AI_GOVERNANCE,
                priority=InsightPriority.MEDIUM,
                title=f"COS {cos_score:.4f} — Good but Can Reach Excellent",
                description=(
                    "Your CATERYA COS is above the safety threshold but below the "
                    "'Excellent' band (≥ 0.85). Improving data diversity and sensor "
                    "coverage will boost symmetry and fairness sub-scores."
                ),
                impact="Higher COS → stronger ESG badge rating → better B2B credibility.",
                action=(
                    "Add more sensor types (humidity, rack-level power, network traffic). "
                    "Increase historical data depth to 90 days. "
                    "Run the CATERYA fairness audit."
                ),
                cos_improvement_delta=0.85 - cos_score,
                tags=["cos", "data_quality", "ai_governance"],
            ))
        return out

    # ── Compliance gaps ───────────────────────────────────────────────────

    def _compliance_insights(self, scores: Dict[str, float]) -> List[Insight]:
        out = []
        for framework, score in scores.items():
            if score < 0.65:
                out.append(Insight(
                    category=InsightCategory.COMPLIANCE_GAP,
                    priority=InsightPriority.HIGH,
                    title=f"{framework} Score {score:.0%} — Remediation Required",
                    description=(
                        f"{framework} compliance at {score:.0%} is in the FAILED/PARTIAL range. "
                        "This may block enterprise deals requiring compliance certificates."
                    ),
                    impact=f"Lost deals from enterprises requiring {framework} certification.",
                    action=(
                        f"Run detailed {framework} gap analysis and create a remediation plan. "
                        "Engage a certified auditor for pre-assessment."
                    ),
                    tags=[framework.lower(), "compliance", "remediation"],
                ))
        return out


# ---------------------------------------------------------------------------
# ISO 42001 Assessor
# ---------------------------------------------------------------------------

class ISO42001Assessor:
    """
    ISO/IEC 42001:2023 AI Management System readiness assessment.
    Uses CATERYA evidence (COS, swarm, provenance) to auto-score clauses.
    """

    def assess(
        self,
        tenant_id: str,
        cos_score: float,
        swarm_approved: bool,
        swarm_consensus: float,
        provenance_chain_events: int,
        audit_log_entries: int,
        rbac_enabled: bool = True,
        encryption_enabled: bool = True,
        compliance_score_iso27001: float = 0.0,
        has_incident_response: bool = True,
    ) -> ISO42001Assessment:
        """Score all ISO 42001 clauses using CATERYA evidence."""

        # Clause 4: Context (governance, stakeholders, scope)
        c4 = min(1.0, 0.5 + (0.3 if rbac_enabled else 0) + (0.2 if audit_log_entries > 0 else 0))

        # Clause 5: Leadership (AI policy, roles, accountability)
        c5 = min(1.0, 0.4 + (0.3 if swarm_approved else 0.1) + (0.3 if cos_score >= 0.7 else 0.1))

        # Clause 6: Planning (risk treatment, objectives)
        c6 = min(1.0, cos_score * 0.6 + (0.3 if swarm_consensus >= 0.7 else 0.1) + 0.1)

        # Clause 7: Support (resources, awareness, documentation)
        c7 = min(1.0, (0.4 if provenance_chain_events > 0 else 0)
                     + (0.3 if encryption_enabled else 0)
                     + (0.3 if audit_log_entries >= 10 else 0.1))

        # Clause 8: Operation (AI system lifecycle, testing, deployment)
        c8 = min(1.0, cos_score * 0.5 + (0.3 if swarm_approved else 0) + 0.2)

        # Clause 9: Performance evaluation (monitoring, internal audit)
        c9 = min(1.0, (0.4 if provenance_chain_events >= 5 else 0.1)
                     + (0.3 if compliance_score_iso27001 >= 0.7 else 0.1)
                     + (0.3 if audit_log_entries >= 50 else 0.1))

        # Clause 10: Improvement (nonconformity, continual improvement)
        c10 = min(1.0, (0.5 if has_incident_response else 0.1)
                      + (0.3 if cos_score >= 0.85 else 0.1)
                      + 0.2)

        overall = round(
            (c4 * 0.10 + c5 * 0.15 + c6 * 0.15 + c7 * 0.15
             + c8 * 0.20 + c9 * 0.15 + c10 * 0.10), 4
        )

        gaps, recs = [], []
        if c4 < 0.6:
            gaps.append("Clause 4: AI governance scope not formally documented")
            recs.append("Define AI scope document and stakeholder register")
        if c5 < 0.6:
            gaps.append("Clause 5: AI policy or accountability framework missing")
            recs.append("Publish AI ethics policy signed by CTO/CEO")
        if c6 < 0.6:
            gaps.append("Clause 6: AI risk assessment not formalised")
            recs.append("Document AI risk register with CATERYA COS evidence")
        if c7 < 0.6:
            gaps.append("Clause 7: Provenance and documentation gaps")
            recs.append("Ensure ProvenanceChain records all AI decisions")
        if c8 < 0.6:
            gaps.append("Clause 8: AI lifecycle management process missing")
            recs.append("Implement CI/CD with CATERYA gate checks (COS ≥ 0.7)")
        if c9 < 0.6:
            gaps.append("Clause 9: Insufficient monitoring and audit evidence")
            recs.append("Schedule quarterly CATERYA swarm audits with exported reports")
        if c10 < 0.6:
            gaps.append("Clause 10: Improvement process not demonstrated")
            recs.append("Document one improvement cycle using simulation version control")

        return ISO42001Assessment(
            tenant_id=tenant_id,
            clause_4_context=round(c4, 4),
            clause_5_leadership=round(c5, 4),
            clause_6_planning=round(c6, 4),
            clause_7_support=round(c7, 4),
            clause_8_operation=round(c8, 4),
            clause_9_performance=round(c9, 4),
            clause_10_improvement=round(c10, 4),
            overall_score=overall,
            certification_ready=overall >= 0.75,
            gaps=gaps,
            recommendations=recs,
            caterya_cos_evidence=cos_score,
        )


# ---------------------------------------------------------------------------
# BI Exporter
# ---------------------------------------------------------------------------

class BIExporter:
    """
    Export ESG report data to formats consumable by Tableau, Power BI, Looker.
    """

    def to_csv(self, data: ESGReportData) -> str:
        """Flat CSV for Tableau / Excel direct import."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "tenant_id", "report_type", "period", "generated_at",
            "cos_composite", "cos_swarm_approved",
            "pue", "wue", "carbon_total_tonnes", "carbon_intensity",
            "renewable_pct",
            "gdpr_score", "iso27001_score", "srn_ppi_score",
            "iso42001_score", "iso42001_ready",
            "insights_count", "critical_insights",
        ])
        writer.writerow([
            data.tenant_id, data.report_type, data.period, data.generated_at,
            data.cos_composite, data.cos_swarm_approved,
            data.pue, data.wue, data.carbon_total_tonnes, data.carbon_intensity,
            data.renewable_pct,
            data.compliance_scores.get("GDPR", 0),
            data.compliance_scores.get("ISO27001", 0),
            data.compliance_scores.get("SRN_PPI", 0),
            data.iso42001.overall_score if data.iso42001 else 0,
            data.iso42001.certification_ready if data.iso42001 else False,
            len(data.insights),
            sum(1 for i in data.insights if i.priority == InsightPriority.CRITICAL),
        ])
        return buf.getvalue()

    def to_insights_csv(self, data: ESGReportData) -> str:
        """One row per insight — useful for tracking recommendations over time."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "tenant_id", "period", "category", "priority", "title",
            "estimated_saving_usd_year", "carbon_reduction_pct",
            "cos_delta", "tags",
        ])
        for ins in data.insights:
            writer.writerow([
                data.tenant_id, data.period,
                ins.category.value, ins.priority.value, ins.title,
                ins.estimated_saving_usd_year,
                ins.estimated_carbon_reduction_pct,
                ins.cos_improvement_delta,
                ",".join(ins.tags),
            ])
        return buf.getvalue()

    def to_json_schema(self, data: ESGReportData) -> Dict[str, Any]:
        """
        Structured JSON for Power BI / Looker / custom dashboards.
        Follows the GHG Protocol + GRI 305 schema naming conventions.
        """
        return {
            "schema_version": "eco-ai-esg-v2",
            "tenant_id":      data.tenant_id,
            "report_type":    data.report_type,
            "reporting_period": data.period,
            "generated_at":   data.generated_at,

            "ai_governance": {
                "cos_composite":       data.cos_composite,
                "cos_passed":          data.cos_composite >= 0.70,
                "swarm_approved":      data.cos_swarm_approved,
                "framework":           "CATERYA v2.0",
                "standard":            "ISO/IEC 42001:2023",
            },

            "energy": {
                "pue":                 data.pue,
                "wue":                 data.wue,
                "renewable_pct":       data.renewable_pct,
                "pue_benchmark_ashrae": 1.2,
                "pue_rating": (
                    "excellent" if data.pue <= 1.2 else
                    "good"      if data.pue <= 1.5 else
                    "average"   if data.pue <= 1.8 else "poor"
                ),
            },

            "emissions": {
                # GRI 305-1: Direct (Scope 1), 305-2: Indirect (Scope 2)
                "scope_2_tonnes_co2e": data.carbon_total_tonnes,
                "carbon_intensity_kg_kwh": data.carbon_intensity,
                "methodology": "Market-based (GHG Protocol)",
                "reporting_standard": "GRI 305, SRN PPI",
            },

            "compliance": {
                **{k: {"score": v, "status": (
                    "compliant" if v >= 0.85 else
                    "warning"   if v >= 0.65 else
                    "partial"   if v >= 0.40 else "failed"
                )} for k, v in data.compliance_scores.items()},
                "iso_42001": (
                    {
                        "overall_score": data.iso42001.overall_score,
                        "certification_ready": data.iso42001.certification_ready,
                        "clause_scores": {
                            "c4_context":    data.iso42001.clause_4_context,
                            "c5_leadership": data.iso42001.clause_5_leadership,
                            "c6_planning":   data.iso42001.clause_6_planning,
                            "c7_support":    data.iso42001.clause_7_support,
                            "c8_operation":  data.iso42001.clause_8_operation,
                            "c9_performance":data.iso42001.clause_9_performance,
                            "c10_improvement":data.iso42001.clause_10_improvement,
                        },
                    }
                    if data.iso42001 else None
                ),
            },

            "insights": [
                {
                    "category":    i.category.value,
                    "priority":    i.priority.value,
                    "title":       i.title,
                    "action":      i.action,
                    "saving_usd_year": i.estimated_saving_usd_year,
                    "carbon_reduction_pct": i.estimated_carbon_reduction_pct,
                    "tags":        i.tags,
                }
                for i in data.insights
            ],

            "provenance": {
                "hashes":    data.provenance_hashes,
                "badge_ids": data.badge_ids,
            },
        }

    def to_html_report(self, data: ESGReportData) -> str:
        """Full standalone HTML ESG report — printable / email-ready."""
        # Build insight cards
        priority_color = {
            "critical": "#ef4444", "high": "#f97316",
            "medium": "#eab308", "low": "#22c55e"
        }
        insight_html = ""
        for ins in data.insights[:8]:  # top 8
            col = priority_color.get(ins.priority.value, "#888")
            saving = f"${ins.estimated_saving_usd_year:,.0f}/yr" if ins.estimated_saving_usd_year else ""
            insight_html += f"""
            <div class="insight-card">
              <div class="insight-header" style="border-left:4px solid {col}">
                <span class="tag" style="background:{col}">{ins.priority.value.upper()}</span>
                <span class="tag-cat">{ins.category.value.replace('_',' ').title()}</span>
                <strong>{ins.title}</strong>
              </div>
              <p>{ins.description}</p>
              <p><em>📋 Action: {ins.action[:200]}{"..." if len(ins.action) > 200 else ""}</em></p>
              {f'<p style="color:#22c55e">💰 Estimated saving: <strong>{saving}</strong></p>' if saving else ""}
            </div>"""

        # Compliance bars
        comp_html = ""
        for fw, score in data.compliance_scores.items():
            bar_col = "#22c55e" if score >= 0.85 else "#eab308" if score >= 0.65 else "#ef4444"
            comp_html += f"""
            <div class="comp-row">
              <span class="comp-label">{fw}</span>
              <div class="bar-bg"><div class="bar-fill" style="width:{score*100:.0f}%;background:{bar_col}"></div></div>
              <span class="comp-score">{score:.0%}</span>
            </div>"""

        iso = data.iso42001
        iso_html = ""
        if iso:
            iso_html = f"""
            <div class="iso-box">
              <h3>ISO/IEC 42001:2023 AI Management System</h3>
              <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                <div class="cos-circle" style="background:{'#22c55e' if iso.certification_ready else '#f97316'}">
                  {iso.overall_score:.0%}
                </div>
                <div>
                  <strong>{'✅ Certification Ready' if iso.certification_ready else '⚠️  Gaps to Address'}</strong><br>
                  <small>Evidence: CATERYA COS {iso.caterya_cos_evidence:.4f}</small>
                </div>
              </div>
              {''.join(f"<p style='color:#ef4444;font-size:13px'>✗ {g}</p>" for g in iso.gaps[:4])}
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESG Report — {data.tenant_id} — {data.period}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Inter',sans-serif; background:#f8f9ff; color:#1a1a2e; }}
  .page {{ max-width:960px; margin:0 auto; padding:32px 24px; }}
  header {{ background:linear-gradient(135deg,#6c63ff,#22c55e); color:white; padding:32px; border-radius:12px; margin-bottom:24px; }}
  header h1 {{ font-size:28px; }} header p {{ opacity:.85; margin-top:6px; }}
  .badge-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; }}
  .badge {{ background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.4); padding:4px 12px; border-radius:20px; font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px; }}
  .card {{ background:white; border-radius:10px; padding:20px; box-shadow:0 1px 6px rgba(0,0,0,.07); }}
  .kpi {{ font-size:32px; font-weight:700; color:#6c63ff; }}
  .kpi-label {{ font-size:13px; color:#666; margin-top:4px; }}
  .section {{ background:white; border-radius:10px; padding:24px; margin-bottom:20px; box-shadow:0 1px 6px rgba(0,0,0,.07); }}
  .section h2 {{ font-size:18px; margin-bottom:16px; color:#6c63ff; }}
  .insight-card {{ border-radius:8px; padding:16px; margin-bottom:12px; background:#fafafa; border:1px solid #eee; }}
  .insight-header {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
  .tag {{ padding:2px 8px; border-radius:12px; color:white; font-size:11px; font-weight:600; }}
  .tag-cat {{ font-size:11px; color:#888; }}
  .cos-circle {{ width:72px; height:72px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-size:20px; font-weight:700; flex-shrink:0; }}
  .comp-row {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
  .comp-label {{ width:120px; font-size:14px; font-weight:600; }}
  .bar-bg {{ flex:1; height:12px; background:#eee; border-radius:6px; }}
  .bar-fill {{ height:100%; border-radius:6px; transition:width 0.3s; }}
  .comp-score {{ width:50px; text-align:right; font-weight:600; }}
  .iso-box {{ background:#f5f3ff; border-radius:8px; padding:20px; margin-top:16px; }}
  footer {{ text-align:center; font-size:12px; color:#888; margin-top:32px; padding-top:16px; border-top:1px solid #eee; }}
  @media print {{ body {{ background:white; }} .page {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>🌿 ESG Sustainability Report</h1>
    <p>{data.tenant_id} &bull; {data.period} &bull; Generated {data.generated_at[:10]}</p>
    <div class="badge-row">
      <span class="badge">🤖 CATERYA-Verified</span>
      <span class="badge">⛓ Blockchain-Anchored</span>
      <span class="badge">🔒 Zero-Trust Secured</span>
      {'<span class="badge">✅ ISO 42001 Ready</span>' if (data.iso42001 and data.iso42001.certification_ready) else ''}
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="kpi" style="color:{'#22c55e' if data.cos_composite >= 0.85 else '#f97316' if data.cos_composite >= 0.7 else '#ef4444'}">{data.cos_composite:.4f}</div>
      <div class="kpi-label">CATERYA COS Score</div>
    </div>
    <div class="card">
      <div class="kpi" style="color:{'#22c55e' if data.pue <= 1.5 else '#f97316' if data.pue <= 1.8 else '#ef4444'}">{data.pue:.2f}</div>
      <div class="kpi-label">Power Usage Effectiveness</div>
    </div>
    <div class="card">
      <div class="kpi" style="color:{'#22c55e' if data.wue <= 1.5 else '#f97316'}">{data.wue:.2f}</div>
      <div class="kpi-label">Water Usage Effectiveness</div>
    </div>
    <div class="card">
      <div class="kpi">{data.carbon_total_tonnes:,.0f}</div>
      <div class="kpi-label">Tonnes CO₂e (Scope 2)</div>
    </div>
    <div class="card">
      <div class="kpi" style="color:#22c55e">{data.renewable_pct:.0%}</div>
      <div class="kpi-label">Renewable Energy Mix</div>
    </div>
    <div class="card">
      <div class="kpi">{len(data.insights)}</div>
      <div class="kpi-label">AI Insights Generated</div>
    </div>
  </div>

  <div class="section">
    <h2>🤖 AI Governance (CATERYA)</h2>
    <div style="display:flex;align-items:center;gap:20px">
      <div class="cos-circle" style="background:{'#22c55e' if data.cos_composite >= 0.70 else '#ef4444'}">{data.cos_composite:.2f}</div>
      <div>
        <strong>COS: {'✅ APPROVED' if data.cos_composite >= 0.70 else '❌ BELOW THRESHOLD'}</strong><br>
        <small>EthicsSwarm: {'✅ Consensus reached' if data.cos_swarm_approved else '⚠️  No consensus'}</small><br>
        <small>Provenance hashes on-chain: {len(data.provenance_hashes)}</small>
      </div>
    </div>
    {iso_html}
  </div>

  <div class="section">
    <h2>📊 Compliance Scores</h2>
    {comp_html}
  </div>

  <div class="section">
    <h2>💡 AI-Driven Insights & Recommendations</h2>
    {insight_html or "<p style='color:#888'>No critical insights detected. System performing well.</p>"}
  </div>

  <footer>
    <p>Generated by Eco AI Data Center v2.0 — CateryaTech &bull; cateryatech@proton.me</p>
    <p style="font-family:monospace;font-size:10px">
      Provenance: {data.provenance_hashes[0][:32] if data.provenance_hashes else 'N/A'}...
    </p>
  </footer>
</div>
</body>
</html>"""
