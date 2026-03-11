"""
reports/report_generator.py
============================
Shareable report generation for Eco AI Data Center.

Generates:
  - HTML compliance + KPI reports (self-contained, embeddable)
  - JSON report bundles (for API consumers)
  - PDF-ready HTML (via print CSS)

Features:
  - Shareable token-based URL generation
  - Version-linked reports (tied to simulation_vc snapshot)
  - CATERYA COS score watermarking on each report
  - Report registry (in-memory; persist to DB in production)

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

from __future__ import annotations

import hashlib
import json
import secrets
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from provenance import ProvenanceChain

# ---------------------------------------------------------------------------
# Report registry (in-memory)
# ---------------------------------------------------------------------------

_REPORT_REGISTRY: Dict[str, dict] = {}


def get_report(token: str) -> Optional[dict]:
    return _REPORT_REGISTRY.get(token)


def list_reports() -> List[dict]:
    return [
        {"token": t, "title": r["title"], "created_at": r["created_at"],
         "author": r["author"], "report_type": r["report_type"]}
        for t, r in _REPORT_REGISTRY.items()
    ]


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    title: str
    author: str
    report_type: str           # "kpi" | "compliance" | "full" | "simulation"
    include_kpis: bool = True
    include_cos: bool = True
    include_compliance: bool = False
    include_simulation: bool = False
    snapshot_id: Optional[str] = None
    cos_threshold: float = 0.7


class ReportGenerator:
    """
    Generates HTML and JSON reports with shareable tokens.
    """

    def __init__(self, provenance: Optional[ProvenanceChain] = None):
        self.provenance = provenance

    def generate(
        self,
        config: ReportConfig,
        kpi_data: Optional[dict] = None,
        cos_data: Optional[dict] = None,
        compliance_data: Optional[dict] = None,
        simulation_log: Optional[List[dict]] = None,
    ) -> dict:
        """
        Generate a full report bundle.

        Returns dict with:
          - report_id, share_token, share_url
          - html (self-contained HTML string)
          - json_bundle (JSON-serialisable dict)
        """
        token = secrets.token_urlsafe(16)
        report_id = f"rpt-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()

        # Build JSON bundle
        bundle = {
            "report_id":    report_id,
            "share_token":  token,
            "title":        config.title,
            "author":       config.author,
            "report_type":  config.report_type,
            "created_at":   created_at,
            "snapshot_id":  config.snapshot_id,
            "kpis":         kpi_data or {},
            "cos":          cos_data or {},
            "compliance":   compliance_data or {},
            "simulation":   simulation_log or [],
            "powered_by":   "CATERYA Framework — CateryaTech",
        }

        # Build HTML
        html = self._build_html(config, bundle)

        full_report = {
            "report_id":   report_id,
            "share_token": token,
            "share_url":   f"/api/v1/reports/{token}",
            "title":       config.title,
            "author":      config.author,
            "report_type": config.report_type,
            "created_at":  created_at,
            "html":        html,
            "json_bundle": bundle,
        }

        _REPORT_REGISTRY[token] = full_report

        if self.provenance:
            self.provenance.record("report_generated", {
                "report_id": report_id, "token": token[:8], "author": config.author,
            })

        return full_report

    def _build_html(self, config: ReportConfig, bundle: dict) -> str:
        """Build a self-contained, print-ready HTML report."""
        kpis = bundle.get("kpis", {})
        cos = bundle.get("cos", {})
        compliance = bundle.get("compliance", {})

        pue_val = kpis.get("pue", {}).get("pue", "N/A")
        wue_val = kpis.get("wue", {}).get("wue", "N/A")
        carbon  = kpis.get("carbon", {}).get("total_co2_tonnes", "N/A")
        cos_val = cos.get("composite", "N/A")
        cos_pass = cos.get("passed", False)

        pue_display = f"{float(pue_val):.3f}" if pue_val != "N/A" else "N/A"
        wue_display = f"{float(wue_val):.3f}" if wue_val != "N/A" else "N/A"
        carbon_display = f"{float(carbon):.2f} t" if carbon != "N/A" else "N/A"
        cos_display = f"{float(cos_val):.4f}" if cos_val != "N/A" else "N/A"
        cos_status = "✅ PASSED" if cos_pass else "⚠️ REVIEW"
        cos_color = "#00c853" if cos_pass else "#ff6d00"

        # Compliance table rows
        compliance_rows = ""
        for fw, result in compliance.items():
            score = result.get("compliance_score", 0)
            status = result.get("overall_status", "N/A")
            status_color = "#00c853" if status == "PASSED" else "#ff6d00" if status == "WARNING" else "#f44336"
            compliance_rows += f"""
            <tr>
              <td><strong>{fw}</strong></td>
              <td>{result.get("passed", 0)}/{result.get("total_controls", 0)}</td>
              <td style="color:{status_color};font-weight:700">{status}</td>
              <td>{score:.1%}</td>
              <td>{'✅' if result.get('swarm_approved') else '⚠️'} {result.get('swarm_consensus', 0):.4f}</td>
            </tr>
            """

        return textwrap.dedent(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>{config.title} — Eco AI Data Center</title>
          <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
            *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
            :root {{
              --green:  #00c853;
              --amber:  #ff6d00;
              --blue:   #0288d1;
              --dark:   #0a0e1a;
              --mid:    #121929;
              --border: #1e3a5f;
              --text:   #e0e0e0;
              --muted:  #7b8794;
              --mono:   'IBM Plex Mono', monospace;
              --sans:   'IBM Plex Sans', sans-serif;
            }}
            body {{ font-family: var(--sans); background: var(--dark); color: var(--text);
                   min-height: 100vh; padding: 0; }}
            .page {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }}

            /* Header */
            .header {{ display: flex; justify-content: space-between; align-items: flex-start;
                        border-bottom: 2px solid var(--green); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
            .header-left h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
                                color: #fff; line-height: 1.2; }}
            .header-left .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }}
            .header-right {{ text-align: right; font-family: var(--mono); font-size: 0.72rem;
                              color: var(--muted); line-height: 1.7; }}
            .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
                      font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
                      background: rgba(0,200,83,0.15); color: var(--green);
                      border: 1px solid rgba(0,200,83,0.3); }}

            /* KPI Grid */
            .section-title {{ font-size: 0.7rem; font-family: var(--mono); letter-spacing: 0.12em;
                               text-transform: uppercase; color: var(--muted); margin-bottom: 1rem;
                               padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
                          gap: 1rem; margin-bottom: 2rem; }}
            .kpi-card {{ background: var(--mid); border: 1px solid var(--border);
                          border-radius: 10px; padding: 1.2rem 1.4rem; position: relative;
                          overflow: hidden; }}
            .kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0;
                                  height: 2px; background: var(--green); }}
            .kpi-label {{ font-size: 0.68rem; font-family: var(--mono); letter-spacing: 0.1em;
                           text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }}
            .kpi-value {{ font-size: 1.8rem; font-weight: 700; font-family: var(--mono);
                           color: #fff; line-height: 1; }}
            .kpi-sub {{ font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }}

            /* COS Panel */
            .cos-panel {{ background: var(--mid); border: 1px solid var(--border);
                           border-radius: 10px; padding: 1.4rem 1.6rem; margin-bottom: 2rem;
                           display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; }}
            .cos-score-big {{ font-size: 3rem; font-weight: 700; font-family: var(--mono);
                               color: {cos_color}; line-height: 1; }}
            .cos-status {{ font-size: 0.9rem; font-weight: 600; color: {cos_color}; margin-top: 0.3rem; }}
            .cos-bars {{ flex: 1; min-width: 200px; }}
            .cos-bar-row {{ display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }}
            .cos-bar-label {{ font-size: 0.7rem; font-family: var(--mono); color: var(--muted);
                               width: 90px; text-align: right; }}
            .cos-bar-track {{ flex: 1; height: 6px; background: rgba(255,255,255,0.08);
                               border-radius: 3px; overflow: hidden; }}
            .cos-bar-fill {{ height: 100%; border-radius: 3px; background: var(--green); }}
            .cos-bar-val {{ font-size: 0.7rem; font-family: var(--mono); color: var(--muted);
                             width: 40px; }}

            /* Compliance Table */
            .compliance-table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem;
                                  font-size: 0.85rem; }}
            .compliance-table th {{ background: rgba(255,255,255,0.04); padding: 0.6rem 0.8rem;
                                     text-align: left; font-family: var(--mono); font-size: 0.68rem;
                                     letter-spacing: 0.08em; text-transform: uppercase;
                                     color: var(--muted); border-bottom: 1px solid var(--border); }}
            .compliance-table td {{ padding: 0.7rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04);
                                     vertical-align: middle; }}
            .compliance-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

            /* Footer */
            .footer {{ border-top: 1px solid var(--border); padding-top: 1.2rem; margin-top: 2rem;
                        display: flex; justify-content: space-between; align-items: center;
                        font-size: 0.72rem; font-family: var(--mono); color: var(--muted);
                        flex-wrap: wrap; gap: 0.5rem; }}
            .footer .brand {{ color: var(--green); font-weight: 600; }}

            @media print {{
              body {{ background: white; color: #111; }}
              .kpi-card, .cos-panel {{ background: #f8f8f8; border-color: #ddd; }}
              :root {{ --green: #007c3d; --text: #111; --mid: #f5f5f5; --muted: #666; }}
            }}

            @media (max-width: 600px) {{
              .header {{ flex-direction: column; gap: 1rem; }}
              .kpi-grid {{ grid-template-columns: 1fr 1fr; }}
              .cos-panel {{ flex-direction: column; }}
            }}
          </style>
        </head>
        <body>
        <div class="page">
          <header class="header">
            <div class="header-left">
              <h1>🌿 {config.title}</h1>
              <div class="subtitle">Eco AI Data Center — Ethical AI Report</div>
            </div>
            <div class="header-right">
              <div>Author: <strong style="color:#e0e0e0">{config.author}</strong></div>
              <div>Generated: {bundle['created_at'][:10]}</div>
              <div>Report ID: <span style="color:#e0e0e0">{bundle['report_id']}</span></div>
              <div style="margin-top:0.4rem"><span class="badge">CATERYA Certified</span></div>
            </div>
          </header>

          <p class="section-title">Key Performance Indicators</p>
          <div class="kpi-grid">
            <div class="kpi-card">
              <div class="kpi-label">PUE</div>
              <div class="kpi-value">{pue_display}</div>
              <div class="kpi-sub">Power Usage Effectiveness</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">WUE</div>
              <div class="kpi-value">{wue_display}</div>
              <div class="kpi-sub">Water Usage Effectiveness</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">CO₂</div>
              <div class="kpi-value">{carbon_display}</div>
              <div class="kpi-sub">Total Carbon Footprint</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">COS Score</div>
              <div class="kpi-value" style="color:{cos_color}">{cos_display}</div>
              <div class="kpi-sub">CATERYA Open Score</div>
            </div>
          </div>

          <p class="section-title">CATERYA Open Score (COS)</p>
          <div class="cos-panel">
            <div>
              <div class="cos-score-big">{cos_display}</div>
              <div class="cos-status">{cos_status}</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem">
                Threshold: {config.cos_threshold:.2f}
              </div>
            </div>
            <div class="cos-bars">
              {''.join([
                f'''<div class="cos-bar-row">
                  <span class="cos-bar-label">{label}</span>
                  <div class="cos-bar-track">
                    <div class="cos-bar-fill" style="width:{min(val*100, 100):.1f}%"></div>
                  </div>
                  <span class="cos-bar-val">{val:.3f}</span>
                </div>'''
                for label, val in [
                  ("entropy", cos.get("entropy_score", 0) or 0),
                  ("symmetry", cos.get("symmetry_score", 0) or 0),
                  ("information", cos.get("information_score", 0) or 0),
                  ("fairness", cos.get("fairness_score", 0) or 0),
                ]
              ])}
            </div>
          </div>

          {'<p class="section-title">Compliance Summary</p><table class="compliance-table"><thead><tr><th>Framework</th><th>Controls</th><th>Status</th><th>Score</th><th>Swarm</th></tr></thead><tbody>' + compliance_rows + '</tbody></table>' if compliance else ''}

          <div class="footer">
            <span>🌿 <span class="brand">Eco AI Data Center</span> — CateryaTech</span>
            <span>cateryatech@proton.me</span>
            <span>github.com/cateryatech/Eco-AI-Data-Center</span>
            <span>Share token: <strong style="color:#e0e0e0">{bundle['share_token'][:8]}…</strong></span>
          </div>
        </div>
        </body>
        </html>
        """).strip()
