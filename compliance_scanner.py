"""
compliance_scanner.py
=====================
Automated compliance scanning for:
  - GDPR (EU General Data Protection Regulation)
  - ISO 27001 (Information Security Management)
  - SRN PPI (Standar Regulasi Nasional Perlindungan Privasi Indonesia)
  - SOC 2 Type II (Service Organization Control)
  - NIST Cybersecurity Framework

Each framework has a set of control checks. Results are evaluated
through EthicsSwarm from CATERYA for consensus-based compliance scoring.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from swarm import EthicsSwarm
from scoring import COSScore
from provenance import ProvenanceChain

logger = logging.getLogger("eco_ai.compliance")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ComplianceStatus(str, Enum):
    PASSED    = "PASSED"
    FAILED    = "FAILED"
    WARNING   = "WARNING"
    NOT_APPLICABLE = "N/A"
    PARTIAL   = "PARTIAL"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


@dataclass
class ComplianceFinding:
    """A single compliance check result."""
    control_id:   str
    control_name: str
    framework:    str
    status:       ComplianceStatus
    severity:     Severity
    description:  str
    evidence:     str = ""
    remediation:  str = ""
    article_ref:  str = ""     # e.g. "GDPR Art. 32", "ISO 27001 A.10.1.1"


@dataclass
class ComplianceScanResult:
    """Full compliance scan result for one framework."""
    framework:       str
    version:         str
    scan_timestamp:  str
    overall_status:  ComplianceStatus
    compliance_score: float          # 0-1
    swarm_consensus: float           # EthicsSwarm consensus
    swarm_approved:  bool
    total_controls:  int
    passed:          int
    failed:          int
    warnings:        int
    findings:        List[ComplianceFinding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata:        dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.framework} {self.version} | {self.overall_status.value} | "
            f"Score: {self.compliance_score:.1%} | "
            f"Controls: {self.passed}/{self.total_controls} passed | "
            f"Swarm: {'✅' if self.swarm_approved else '⚠️'} {self.swarm_consensus:.4f}"
        )


# ---------------------------------------------------------------------------
# Framework control definitions
# ---------------------------------------------------------------------------

GDPR_CONTROLS = [
    {"id": "GDPR-5.1a",  "name": "Lawfulness of Processing",          "article": "Art. 5(1)(a)", "severity": Severity.CRITICAL},
    {"id": "GDPR-5.1b",  "name": "Purpose Limitation",                "article": "Art. 5(1)(b)", "severity": Severity.HIGH},
    {"id": "GDPR-5.1c",  "name": "Data Minimisation",                 "article": "Art. 5(1)(c)", "severity": Severity.HIGH},
    {"id": "GDPR-5.1e",  "name": "Storage Limitation",                "article": "Art. 5(1)(e)", "severity": Severity.MEDIUM},
    {"id": "GDPR-5.1f",  "name": "Integrity and Confidentiality",     "article": "Art. 5(1)(f)", "severity": Severity.CRITICAL},
    {"id": "GDPR-25",    "name": "Data Protection by Design/Default", "article": "Art. 25",      "severity": Severity.HIGH},
    {"id": "GDPR-32",    "name": "Security of Processing (Encryption)", "article": "Art. 32",   "severity": Severity.CRITICAL},
    {"id": "GDPR-33",    "name": "Breach Notification (72h)",         "article": "Art. 33",      "severity": Severity.HIGH},
    {"id": "GDPR-35",    "name": "Data Protection Impact Assessment", "article": "Art. 35",      "severity": Severity.HIGH},
    {"id": "GDPR-37",    "name": "Data Protection Officer",           "article": "Art. 37",      "severity": Severity.MEDIUM},
    {"id": "GDPR-44",    "name": "Transfers to Third Countries",      "article": "Art. 44",      "severity": Severity.HIGH},
    {"id": "GDPR-83",    "name": "Administrative Fines Assessment",   "article": "Art. 83",      "severity": Severity.MEDIUM},
]

ISO27001_CONTROLS = [
    {"id": "ISO-A.5.1",  "name": "Information Security Policies",      "article": "A.5.1",  "severity": Severity.HIGH},
    {"id": "ISO-A.6.1",  "name": "Internal Organisation",              "article": "A.6.1",  "severity": Severity.MEDIUM},
    {"id": "ISO-A.8.1",  "name": "Responsibility for Assets",          "article": "A.8.1",  "severity": Severity.HIGH},
    {"id": "ISO-A.9.1",  "name": "Business Requirements for Access Control", "article": "A.9.1", "severity": Severity.CRITICAL},
    {"id": "ISO-A.9.2",  "name": "User Access Management (RBAC)",      "article": "A.9.2",  "severity": Severity.CRITICAL},
    {"id": "ISO-A.9.4",  "name": "System & Application Access Control","article": "A.9.4",  "severity": Severity.HIGH},
    {"id": "ISO-A.10.1", "name": "Cryptographic Controls (AES-256)",   "article": "A.10.1", "severity": Severity.CRITICAL},
    {"id": "ISO-A.12.1", "name": "Operational Procedures",             "article": "A.12.1", "severity": Severity.MEDIUM},
    {"id": "ISO-A.12.4", "name": "Logging and Monitoring",             "article": "A.12.4", "severity": Severity.HIGH},
    {"id": "ISO-A.12.6", "name": "Technical Vulnerability Management", "article": "A.12.6", "severity": Severity.HIGH},
    {"id": "ISO-A.13.1", "name": "Network Security Management",        "article": "A.13.1", "severity": Severity.HIGH},
    {"id": "ISO-A.14.2", "name": "Security in Development/Support",    "article": "A.14.2", "severity": Severity.MEDIUM},
    {"id": "ISO-A.16.1", "name": "Incident Management",                "article": "A.16.1", "severity": Severity.HIGH},
    {"id": "ISO-A.17.1", "name": "Business Continuity",                "article": "A.17.1", "severity": Severity.MEDIUM},
    {"id": "ISO-A.18.1", "name": "Compliance with Legal Requirements", "article": "A.18.1", "severity": Severity.CRITICAL},
]

# SRN PPI — Standar Regulasi Nasional Perlindungan Privasi Indonesia
# Based on Indonesia's Personal Data Protection Law (UU PDP 2022)
SRN_PPI_CONTROLS = [
    {"id": "PPI-1.1",  "name": "Dasar Pemrosesan Data Pribadi (Lawful Basis)",   "article": "Pasal 20",   "severity": Severity.CRITICAL},
    {"id": "PPI-1.2",  "name": "Persetujuan Pemilik Data (Explicit Consent)",    "article": "Pasal 21",   "severity": Severity.CRITICAL},
    {"id": "PPI-1.3",  "name": "Pembatasan Tujuan (Purpose Limitation)",         "article": "Pasal 23",   "severity": Severity.HIGH},
    {"id": "PPI-2.1",  "name": "Pemberitahuan kepada Pemilik Data (Notice)",     "article": "Pasal 25",   "severity": Severity.HIGH},
    {"id": "PPI-2.2",  "name": "Hak Akses Pemilik Data (Right of Access)",       "article": "Pasal 34",   "severity": Severity.HIGH},
    {"id": "PPI-2.3",  "name": "Hak Koreksi Data (Right to Rectification)",      "article": "Pasal 35",   "severity": Severity.MEDIUM},
    {"id": "PPI-2.4",  "name": "Hak Penghapusan Data (Right to Erasure)",        "article": "Pasal 37",   "severity": Severity.HIGH},
    {"id": "PPI-3.1",  "name": "Keamanan Data Pribadi (Data Security)",          "article": "Pasal 46",   "severity": Severity.CRITICAL},
    {"id": "PPI-3.2",  "name": "Enkripsi Data (Encryption Requirement)",         "article": "Pasal 46(2)", "severity": Severity.CRITICAL},
    {"id": "PPI-3.3",  "name": "Pemberitahuan Kebocoran Data (Breach Notification)","article": "Pasal 47","severity": Severity.HIGH},
    {"id": "PPI-4.1",  "name": "Transfer Data Lintas Batas (Cross-border Transfer)","article": "Pasal 56","severity": Severity.HIGH},
    {"id": "PPI-4.2",  "name": "Penunjukan DPO (Data Protection Officer)",       "article": "Pasal 53",   "severity": Severity.MEDIUM},
    # Carbon/Environmental data (SRN context for data centers in Indonesia)
    {"id": "PPI-5.1",  "name": "Pelaporan Jejak Karbon (Carbon Reporting)",      "article": "Perpres 98/2021", "severity": Severity.HIGH},
    {"id": "PPI-5.2",  "name": "Audit Lingkungan Berkala (Environmental Audit)", "article": "Perpres 98/2021", "severity": Severity.MEDIUM},
]

SOC2_CONTROLS = [
    {"id": "SOC2-CC1", "name": "Control Environment",          "article": "CC1", "severity": Severity.HIGH},
    {"id": "SOC2-CC2", "name": "Communication & Information",  "article": "CC2", "severity": Severity.MEDIUM},
    {"id": "SOC2-CC3", "name": "Risk Assessment",              "article": "CC3", "severity": Severity.HIGH},
    {"id": "SOC2-CC6", "name": "Logical & Physical Access",    "article": "CC6", "severity": Severity.CRITICAL},
    {"id": "SOC2-CC7", "name": "System Operations",            "article": "CC7", "severity": Severity.HIGH},
    {"id": "SOC2-CC8", "name": "Change Management",            "article": "CC8", "severity": Severity.MEDIUM},
    {"id": "SOC2-A1",  "name": "Availability",                 "article": "A1",  "severity": Severity.HIGH},
    {"id": "SOC2-C1",  "name": "Confidentiality",              "article": "C1",  "severity": Severity.CRITICAL},
    {"id": "SOC2-P1",  "name": "Privacy Notice",               "article": "P1",  "severity": Severity.HIGH},
]

FRAMEWORK_REGISTRY = {
    "GDPR":     {"controls": GDPR_CONTROLS,    "version": "2016/679"},
    "ISO27001": {"controls": ISO27001_CONTROLS, "version": "2022"},
    "SRN_PPI":  {"controls": SRN_PPI_CONTROLS,  "version": "UU PDP 2022"},
    "SOC2":     {"controls": SOC2_CONTROLS,     "version": "2017"},
}


# ---------------------------------------------------------------------------
# Compliance Checker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """
    Evaluates a system configuration against compliance framework controls.
    """

    def __init__(self, system_config: Optional[dict] = None):
        self.config = system_config or {}

    def check_control(self, control: dict, framework: str) -> ComplianceFinding:
        """Evaluate a single control. Returns a ComplianceFinding."""
        ctrl_id = control["id"]
        method_name = f"_check_{ctrl_id.replace('-', '_').replace('.', '_').lower()}"
        method = getattr(self, method_name, self._default_check)
        return method(control, framework)

    def _default_check(self, control: dict, framework: str) -> ComplianceFinding:
        """Generic check based on system_config keys."""
        ctrl_id = control["id"]

        # Map controls to config keys
        config_checks = {
            # Auth / RBAC
            "GDPR-5.1f": ("encryption_enabled", True),
            "GDPR-25":   ("privacy_by_design", True),
            "GDPR-32":   ("encryption_algorithm", "AES-256"),
            "GDPR-33":   ("breach_notification_policy", True),
            "GDPR-35":   ("dpia_completed", True),
            "GDPR-37":   ("dpo_appointed", True),
            "ISO-A.9.2": ("rbac_enabled", True),
            "ISO-A.10.1": ("encryption_enabled", True),
            "ISO-A.12.4": ("audit_logging_enabled", True),
            "ISO-A.9.1": ("access_control_policy", True),
            "PPI-3.2":   ("encryption_algorithm", "AES-256"),
            "PPI-3.3":   ("breach_notification_policy", True),
            "PPI-4.2":   ("dpo_appointed", True),
            "PPI-5.1":   ("carbon_reporting_enabled", True),
            "SOC2-CC6":  ("rbac_enabled", True),
            "SOC2-C1":   ("encryption_enabled", True),
        }

        check = config_checks.get(ctrl_id)
        if check:
            key, expected = check
            actual = self.config.get(key)
            passed = actual == expected

            return ComplianceFinding(
                control_id=ctrl_id,
                control_name=control["name"],
                framework=framework,
                status=ComplianceStatus.PASSED if passed else ComplianceStatus.FAILED,
                severity=control["severity"],
                description=f"{'✅' if passed else '❌'} {control['name']}",
                evidence=f"Config '{key}' = {actual!r} (expected: {expected!r})",
                remediation="" if passed else f"Set '{key}' = {expected!r} in system configuration.",
                article_ref=control.get("article", ""),
            )

        # Default: treat as WARNING (unknown / not testable automatically)
        return ComplianceFinding(
            control_id=ctrl_id,
            control_name=control["name"],
            framework=framework,
            status=ComplianceStatus.WARNING,
            severity=Severity.LOW,
            description=f"⚠️ Manual review required: {control['name']}",
            evidence="Automated check not available — requires manual assessment.",
            remediation="Perform manual assessment per framework guidelines.",
            article_ref=control.get("article", ""),
        )


# ---------------------------------------------------------------------------
# Main Compliance Scanner
# ---------------------------------------------------------------------------

class ComplianceScanner:
    """
    Full compliance scanning engine with EthicsSwarm consensus.

    Scans one or more frameworks, gathers findings, then passes
    the composite score through EthicsSwarm for consensus evaluation.

    Parameters
    ----------
    system_config : dict of system state flags for automated checks
    provenance    : ProvenanceChain for audit recording
    swarm_threshold : EthicsSwarm approval threshold
    """

    def __init__(
        self,
        system_config: Optional[dict] = None,
        provenance: Optional[ProvenanceChain] = None,
        swarm_threshold: float = 0.7,
    ):
        # Default system config reflects what our app actually has
        self.system_config = {
            "encryption_enabled":       True,
            "encryption_algorithm":     "AES-256",
            "rbac_enabled":             True,
            "audit_logging_enabled":    True,
            "access_control_policy":    True,
            "breach_notification_policy": False,  # needs manual process
            "dpia_completed":           False,    # needs manual DPIA
            "dpo_appointed":            False,    # needs human DPO
            "privacy_by_design":        True,
            "carbon_reporting_enabled": True,
            "tls_enforced":             True,
            "mfa_enabled":              False,    # optional in current build
            "data_retention_policy":    False,    # needs manual policy
            "vulnerability_scanning":   False,    # needs CI/CD integration
        }
        if system_config:
            self.system_config.update(system_config)

        self.checker = ComplianceChecker(self.system_config)
        self.provenance = provenance or ProvenanceChain(model_id="compliance-scanner")
        self.swarm = EthicsSwarm(n_agents=5, threshold=swarm_threshold)
        self._scan_history: List[ComplianceScanResult] = []

        logger.info("[Compliance] Scanner initialised with %d config keys.", len(self.system_config))

    def scan_framework(self, framework_name: str) -> ComplianceScanResult:
        """
        Scan a single compliance framework.

        Parameters
        ----------
        framework_name : "GDPR" | "ISO27001" | "SRN_PPI" | "SOC2"

        Returns
        -------
        ComplianceScanResult
        """
        fw = FRAMEWORK_REGISTRY.get(framework_name)
        if not fw:
            raise ValueError(
                f"Unknown framework: '{framework_name}'. "
                f"Available: {list(FRAMEWORK_REGISTRY.keys())}"
            )

        logger.info("[Compliance] Scanning framework: %s %s", framework_name, fw["version"])

        findings: List[ComplianceFinding] = []
        for control in fw["controls"]:
            finding = self.checker.check_control(control, framework_name)
            findings.append(finding)

        # Compute compliance score
        passed    = sum(1 for f in findings if f.status == ComplianceStatus.PASSED)
        failed    = sum(1 for f in findings if f.status == ComplianceStatus.FAILED)
        warnings  = sum(1 for f in findings if f.status == ComplianceStatus.WARNING)
        total     = len(findings)

        # Weight CRITICAL failures more heavily
        weighted_score = self._compute_weighted_score(findings)

        # Determine overall status
        if failed == 0 and warnings == 0:
            overall = ComplianceStatus.PASSED
        elif failed == 0:
            overall = ComplianceStatus.WARNING
        elif failed <= total * 0.20:
            overall = ComplianceStatus.PARTIAL
        else:
            overall = ComplianceStatus.FAILED

        # EthicsSwarm consensus — use compliance score as COS proxy
        cos_proxy = COSScore(
            entropy_score=weighted_score,
            symmetry_score=weighted_score,
            information_score=weighted_score,
            fairness_score=weighted_score,
        )
        cos_proxy.compute_composite()
        swarm_result = self.swarm.deliberate(cos_proxy)

        # Build recommendations
        recs = self._build_recommendations(findings, framework_name)

        result = ComplianceScanResult(
            framework=framework_name,
            version=fw["version"],
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall,
            compliance_score=round(weighted_score, 4),
            swarm_consensus=swarm_result["consensus_score"],
            swarm_approved=swarm_result["approved"],
            total_controls=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            findings=findings,
            recommendations=recs,
            metadata={
                "system_config_snapshot": self.system_config,
                "swarm_summary": swarm_result["summary"],
            },
        )

        self._scan_history.append(result)
        self.provenance.record(f"compliance.{framework_name}", {
            "framework":        framework_name,
            "overall_status":   overall.value,
            "compliance_score": result.compliance_score,
            "passed":           passed,
            "failed":           failed,
            "swarm_approved":   swarm_result["approved"],
        })

        logger.info("[Compliance] %s", result.summary())
        return result

    def scan_all(self) -> Dict[str, ComplianceScanResult]:
        """Scan all registered frameworks. Returns dict of results."""
        results = {}
        for framework_name in FRAMEWORK_REGISTRY:
            results[framework_name] = self.scan_framework(framework_name)
        return results

    def get_history(self) -> List[ComplianceScanResult]:
        return list(self._scan_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weighted_score(self, findings: List[ComplianceFinding]) -> float:
        """Weighted score: CRITICAL failures penalise more than LOW ones."""
        weights = {
            Severity.CRITICAL: 4.0,
            Severity.HIGH:     3.0,
            Severity.MEDIUM:   2.0,
            Severity.LOW:      1.0,
            Severity.INFO:     0.5,
        }
        total_weight = sum(weights.get(f.severity, 1.0) for f in findings)
        if total_weight == 0:
            return 0.0

        passed_weight = sum(
            weights.get(f.severity, 1.0)
            for f in findings
            if f.status == ComplianceStatus.PASSED
        )
        # Partial credit for warnings
        warning_weight = sum(
            weights.get(f.severity, 1.0) * 0.5
            for f in findings
            if f.status == ComplianceStatus.WARNING
        )
        return round((passed_weight + warning_weight) / total_weight, 4)

    def _build_recommendations(
        self, findings: List[ComplianceFinding], framework: str
    ) -> List[str]:
        """Extract and prioritise remediation recommendations."""
        critical_recs = []
        high_recs = []
        other_recs = []

        for f in findings:
            if f.status in (ComplianceStatus.FAILED, ComplianceStatus.WARNING) and f.remediation:
                entry = f"[{f.control_id}] {f.remediation}"
                if f.severity == Severity.CRITICAL:
                    critical_recs.append(entry)
                elif f.severity == Severity.HIGH:
                    high_recs.append(entry)
                else:
                    other_recs.append(entry)

        return (
            [f"🔴 CRITICAL: {r}" for r in critical_recs]
            + [f"🟠 HIGH: {r}" for r in high_recs]
            + [f"🟡 MEDIUM/LOW: {r}" for r in other_recs[:5]]
        )
