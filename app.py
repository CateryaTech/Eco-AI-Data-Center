"""
app.py — Eco AI Data Center
============================
Full-stack enterprise dashboard integrating:
  - CATERYA ethical AI framework (COS scoring, provenance, ethics swarm)
  - Real-time monitoring (IoT sensors, Prometheus, InfluxDB)
  - LSTM predictive maintenance + Quantum Symmetry Index
  - Quantum thermal circuit simulation (PennyLane / numpy fallback)
  - Celery background tasks with custom alerting
  - JWT / RBAC security with role-gated UI
  - AES-256 data encryption at-rest/in-transit
  - Comprehensive audit logging
  - GDPR / ISO 27001 / SRN PPI / SOC2 compliance scanner
  - Git-like simulation version control
  - Shareable HTML/JSON report generation
  - FastAPI hybrid backend at :8000

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluator import CATERYAEvaluator
from fairness import QuantumFairnessEvaluator
from provenance import ProvenanceChain
from swarm import EthicsSwarm
from scoring import COSScore
from caterya_integration import EcoAIDataCenterCATERYA
from optimizer import full_optimization, compute_pue, compute_wue, compute_carbon
from real_time_monitor import RealTimeMonitor, QuantumSymmetryAnalyser, MockSensorBackend
from thermal_circuit import QuantumThermalSimulator
from tasks import (
    run_monitor_poll,
    run_cos_evaluation,
    run_quantum_thermal,
    run_predictive_check,
    get_task_status,
    CELERY_AVAILABLE,
)
from auth import AuthService, User, Role, get_auth_service
from audit_logger import AuditLogger, AuditEventType, get_audit_logger
from encryption import EncryptionEngine, hash_data
from compliance_scanner import ComplianceScanner, FRAMEWORK_REGISTRY
from version_control import SimulationVersionControl
from report_generator import ReportGenerator, ReportConfig, list_reports, get_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eco_ai.app")

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Eco AI Data Center",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/cateryatech/Eco-AI-Data-Center",
        "About": "Eco AI Data Center | CateryaTech | cateryatech@proton.me",
    },
)

# ---------------------------------------------------------------------------
# CSS Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main gradient header */
    .main-header {
        background: linear-gradient(135deg, #0d2137 0%, #1a4a6b 40%, #0f7c5e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e8f5e9; font-size: 2.2rem; margin: 0; letter-spacing: 0.5px; }
    .main-header p  { color: #a5d6a7; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* COS score card */
    .cos-card {
        background: #0d1b2a;
        border: 1.5px solid #1b5e20;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .cos-score { font-size: 3.5rem; font-weight: 700; color: #69f0ae; }
    .cos-label { font-size: 0.85rem; color: #80cbc4; text-transform: uppercase; letter-spacing: 1px; }

    /* Warning banner */
    .warning-banner {
        background: linear-gradient(90deg, #b71c1c, #880e4f);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #fff;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .success-banner {
        background: linear-gradient(90deg, #1b5e20, #004d40);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #e8f5e9;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    /* Metric cards */
    .stMetric { background: #0d1b2a; border-radius: 10px; padding: 1rem; border: 1px solid #1e3a5f; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { background: #0d2137; border-radius: 8px 8px 0 0; color: #80cbc4; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #1a4a6b; color: #69f0ae; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
def _init_session_state():
    defaults = {
        # CATERYA
        "caterya": None,
        "last_cos": None,
        "last_swarm": None,
        "last_fairness": None,
        "last_opt_result": None,
        "uploaded_data": None,
        "cos_history": [],
        "cos_threshold": 0.7,
        "evaluation_count": 0,
        # Real-time monitoring
        "monitor": None,
        "monitor_backend": "mock",
        "monitor_history": [],
        "monitor_alerts": [],
        "monitor_poll_count": 0,
        # Quantum
        "last_thermal_result": None,
        "last_qsi_results": {},
        "thermal_history": [],
        # Tasks
        "last_task_result": None,
        # Alerts config
        "alert_slack_webhook": "",
        "alert_email": "",
        # Security / Auth
        "current_user": None,
        "auth_token": None,
        "auth_service": None,
        "audit_logger": None,
        "encryption_engine": None,
        # Compliance
        "compliance_scanner": None,
        "last_compliance_results": {},
        # Version control
        "simulation_vc": None,
        # Reports
        "report_generator": None,
        "last_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()


def get_integration() -> EcoAIDataCenterCATERYA:
    """Get or create the CATERYA integration instance."""
    if st.session_state["caterya"] is None:
        st.session_state["caterya"] = EcoAIDataCenterCATERYA(
            cos_threshold=st.session_state["cos_threshold"],
            model_id="eco-ai-optimizer",
            verbose=False,
        )
    return st.session_state["caterya"]


def _get_monitor() -> RealTimeMonitor:
    """Get or create the RealTimeMonitor instance."""
    backend = st.session_state.get("monitor_backend", "mock")
    existing = st.session_state.get("monitor")
    if existing is None or getattr(existing, "backend_name", None) != backend:
        mon = RealTimeMonitor(
            backend=backend,
            poll_interval=5.0,
            alert_slack_webhook=st.session_state.get("alert_slack_webhook") or None,
            alert_email_to=st.session_state.get("alert_email") or None,
        )
        st.session_state["monitor"] = mon
    return st.session_state["monitor"]


def _get_auth_service() -> AuthService:
    if st.session_state["auth_service"] is None:
        st.session_state["auth_service"] = get_auth_service()
    return st.session_state["auth_service"]


def _get_audit_logger() -> AuditLogger:
    if st.session_state["audit_logger"] is None:
        st.session_state["audit_logger"] = get_audit_logger()
    return st.session_state["audit_logger"]


def _get_encryption_engine() -> EncryptionEngine:
    if st.session_state["encryption_engine"] is None:
        st.session_state["encryption_engine"] = EncryptionEngine()
    return st.session_state["encryption_engine"]


def _get_compliance_scanner() -> ComplianceScanner:
    if st.session_state["compliance_scanner"] is None:
        st.session_state["compliance_scanner"] = ComplianceScanner()
    return st.session_state["compliance_scanner"]


def _get_simulation_vc() -> SimulationVersionControl:
    if st.session_state["simulation_vc"] is None:
        user = st.session_state.get("current_user")
        author = user.username if user else "anonymous"
        st.session_state["simulation_vc"] = SimulationVersionControl(default_author=author)
    return st.session_state["simulation_vc"]


def _get_report_generator() -> ReportGenerator:
    if st.session_state["report_generator"] is None:
        st.session_state["report_generator"] = ReportGenerator(
            provenance=get_integration().provenance
        )
    return st.session_state["report_generator"]


def _current_user() -> Optional[User]:
    return st.session_state.get("current_user")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌿 Eco AI Data Center")
        st.markdown("**CateryaTech Framework**")
        st.markdown("---")

        st.markdown("### ⚙️ CATERYA Settings")
        threshold = st.slider(
            "COS Threshold (Guardrail)",
            min_value=0.0, max_value=1.0,
            value=st.session_state["cos_threshold"],
            step=0.05,
            help="Minimum COS score for ethical deployment approval.",
        )
        if threshold != st.session_state["cos_threshold"]:
            st.session_state["cos_threshold"] = threshold
            st.session_state["caterya"] = None

        st.markdown("---")
        st.markdown("### 🔴 Real-Time Monitor")
        backend = st.selectbox(
            "Sensor Backend",
            ["mock", "prometheus", "influxdb"],
            index=["mock", "prometheus", "influxdb"].index(
                st.session_state.get("monitor_backend", "mock")
            ),
        )
        st.session_state["monitor_backend"] = backend

        if st.button("🔄 Poll Once", use_container_width=True):
            monitor = _get_monitor()
            with st.spinner("Polling sensors…"):
                snapshot = monitor.poll_once()
                st.session_state["monitor_poll_count"] += 1
                n_readings = len(snapshot.get("readings", {}))
                alerts = snapshot.get("alerts", [])
                if alerts:
                    st.session_state["monitor_alerts"].extend(alerts)
                st.success(f"✅ {n_readings} metrics read")
                if alerts:
                    st.warning(f"⚠️ {len(alerts)} alert(s) fired")

        poll_count = st.session_state["monitor_poll_count"]
        if poll_count:
            st.caption(f"Total polls: {poll_count}")

        st.markdown("---")
        st.markdown("### 🔔 Alert Config")
        st.session_state["alert_slack_webhook"] = st.text_input(
            "Slack Webhook URL", value=st.session_state.get("alert_slack_webhook", ""),
            type="password", placeholder="https://hooks.slack.com/…"
        )
        st.session_state["alert_email"] = st.text_input(
            "Alert Email", value=st.session_state.get("alert_email", ""),
            placeholder="ops@company.com"
        )

        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        cos_val = st.session_state.get("last_cos")
        if cos_val:
            st.metric("Last COS Score", f"{cos_val.composite:.4f}")
            st.metric("Evaluations Run", st.session_state["evaluation_count"])
            status = "✅ Passed" if cos_val.passed else "⚠️ Failed"
            st.metric("Status", status)

        st.markdown("---")
        integration = get_integration()
        st.markdown("### 🔧 Dependencies")
        st.markdown(f"{'✅' if integration.is_dask_available else '❌'} Dask (Big Data)")
        st.markdown(f"{'✅' if integration.is_aws_available else '❌'} boto3 (AWS Scaling)")

        from real_time_monitor import TORCH_AVAILABLE, INFLUX_AVAILABLE, REQUESTS_AVAILABLE
        from thermal_circuit import PENNYLANE_AVAILABLE
        st.markdown(f"{'✅' if TORCH_AVAILABLE else '⚠️'} PyTorch (LSTM)")
        st.markdown(f"{'✅' if PENNYLANE_AVAILABLE else '⚠️'} PennyLane (Quantum)")
        st.markdown(f"{'✅' if CELERY_AVAILABLE else '⚠️'} Celery (Background Tasks)")
        st.markdown(f"{'✅' if INFLUX_AVAILABLE else '⚠️'} InfluxDB Client")

        chain_ok = integration.chain_integrity_ok()
        st.markdown(f"{'✅' if chain_ok else '⛔'} Provenance Chain Integrity")

        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[📁 GitHub](https://github.com/cateryatech/Eco-AI-Data-Center)")
        st.markdown("[📧 Contact](mailto:cateryatech@proton.me)")


# ---------------------------------------------------------------------------
# Data Generation / Upload
# ---------------------------------------------------------------------------
def generate_sample_data(n_rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
        "total_power_kw":            rng.normal(1200, 80, n_rows).clip(900, 1600),
        "it_power_kw":               rng.normal(700,  50, n_rows).clip(500, 950),
        "water_usage_litres":        rng.normal(500,  40, n_rows).clip(300, 800),
        "it_energy_kwh":             rng.normal(350,  25, n_rows).clip(200, 500),
        "energy_kwh":                rng.normal(1100, 60, n_rows).clip(800, 1500),
        "carbon_intensity_kg_per_kwh": rng.normal(0.25, 0.05, n_rows).clip(0.05, 0.6),
        "cooling_power_kw":          rng.normal(300,  30, n_rows).clip(150, 500),
        "server_utilization_pct":    rng.normal(72,   12, n_rows).clip(10, 100),
        "temperature_celsius":       rng.normal(22,    2, n_rows).clip(16, 35),
    })


def get_data() -> pd.DataFrame:
    """Return uploaded data or sample data."""
    if st.session_state["uploaded_data"] is not None:
        return st.session_state["uploaded_data"]
    return generate_sample_data()


# ---------------------------------------------------------------------------
# COS Score Panel
# ---------------------------------------------------------------------------
def render_cos_panel(cos: COSScore):
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        color = "#69f0ae" if cos.passed else "#ff5252"
        st.markdown(f"""
        <div class="cos-card">
            <div class="cos-label">CATERYA Open Score (COS)</div>
            <div class="cos-score" style="color:{color}">{cos.composite:.4f}</div>
            <div class="cos-label">Threshold: {cos.threshold} | 
                {'✅ PASSED' if cos.passed else '⚠️ BELOW THRESHOLD'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Sub-scores
    mc = st.columns(4)
    mc[0].metric("🔵 Entropy",     f"{cos.entropy_score:.4f}",
                 delta=f"{cos.entropy_score - cos.threshold:+.3f}")
    mc[1].metric("🟢 Symmetry",    f"{cos.symmetry_score:.4f}",
                 delta=f"{cos.symmetry_score - cos.threshold:+.3f}")
    mc[2].metric("🟡 Information", f"{cos.information_score:.4f}",
                 delta=f"{cos.information_score - cos.threshold:+.3f}")
    mc[3].metric("🔴 Fairness",    f"{cos.fairness_score:.4f}",
                 delta=f"{cos.fairness_score - cos.threshold:+.3f}")

    # Warnings / Success banner
    if not cos.passed:
        st.markdown(f"""
        <div class="warning-banner">
            ⚠️ COS GUARDRAIL TRIGGERED — Score {cos.composite:.4f} is below threshold 
            {cos.threshold}. Deployment not recommended until issues are remediated.
        </div>
        """, unsafe_allow_html=True)
        if cos.warnings:
            with st.expander("🔍 Detailed Warnings"):
                for w in cos.warnings:
                    st.warning(w)
    else:
        st.markdown(f"""
        <div class="success-banner">
            ✅ All CATERYA ethical checks passed — COS {cos.composite:.4f} 
            exceeds threshold {cos.threshold}. Model approved for deployment.
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Radar Chart
# ---------------------------------------------------------------------------
def render_radar_chart(cos: COSScore):
    categories = ["Entropy", "Symmetry", "Information", "Fairness"]
    values = [
        cos.entropy_score,
        cos.symmetry_score,
        cos.information_score,
        cos.fairness_score,
    ]
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(105, 240, 174, 0.2)",
        line=dict(color="#69f0ae", width=2),
        name="COS Sub-scores",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[cos.threshold] * len(categories),
        theta=categories,
        line=dict(color="#ff5252", width=1.5, dash="dash"),
        name=f"Threshold ({cos.threshold})",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1b2a",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e3a5f", color="#80cbc4"),
            angularaxis=dict(gridcolor="#1e3a5f", color="#80cbc4"),
        ),
        paper_bgcolor="#0d1b2a",
        font=dict(color="#e0e0e0"),
        showlegend=True,
        legend=dict(bgcolor="#0d2137", bordercolor="#1e3a5f"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# COS History Chart
# ---------------------------------------------------------------------------
def render_cos_history():
    history = st.session_state["cos_history"]
    if len(history) < 2:
        st.info("Run at least 2 evaluations to see COS trend.")
        return

    df = pd.DataFrame(history)
    fig = px.line(
        df, x="run", y="composite",
        title="COS Score History",
        labels={"run": "Evaluation Run", "composite": "COS Score"},
        color_discrete_sequence=["#69f0ae"],
    )
    fig.add_hline(
        y=st.session_state["cos_threshold"],
        line_dash="dash", line_color="#ff5252",
        annotation_text=f"Threshold ({st.session_state['cos_threshold']})",
    )
    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f", range=[0, 1]),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Swarm Visualisation
# ---------------------------------------------------------------------------
def render_swarm_chart(swarm_result: dict):
    votes = swarm_result.get("votes", {})
    if not votes:
        return

    agents = list(votes.keys())
    scores = [votes[a]["vote"] for a in agents]
    rationales = [votes[a]["rationale"] for a in agents]
    colors = ["#69f0ae" if s >= st.session_state["cos_threshold"] else "#ff5252" for s in scores]

    fig = go.Figure(go.Bar(
        x=agents, y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        hovertext=rationales,
        hoverinfo="text+y",
    ))
    fig.add_hline(
        y=st.session_state["cos_threshold"],
        line_dash="dash", line_color="#ff9800",
        annotation_text="Threshold",
    )
    fig.update_layout(
        title="Ethics Swarm Agent Votes",
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", range=[0, 1]),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Eco AI Data Center</h1>
        <p>Powered by CATERYA Ethical AI Framework · CateryaTech · cateryatech@proton.me</p>
    </div>
    """, unsafe_allow_html=True)

    # Top-level tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "⚖️ CATERYA Evaluation",
        "🔴 Real-Time Monitor",
        "⚛️ Quantum Thermal",
        "🔮 Predictive Maintenance",
        "📈 Analytics",
        "🔒 Audit & Provenance",
        "☁️ Cloud Scaling",
        "⚙️ Background Tasks",
        "🛡️ Security & Auth",
        "✅ Compliance",
        "🗂️ Version Control",
        "📋 Reports",
        "📁 Data",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Dashboard
    # -----------------------------------------------------------------------
    with tabs[0]:
        st.subheader("🏠 Live Dashboard")

        data = get_data()
        integration = get_integration()

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button("🔄 Refresh COS Score", use_container_width=True, type="primary"):
                with st.spinner("Computing CATERYA Open Score …"):
                    cos = integration.score_data(data)
                    st.session_state["last_cos"] = cos
                    st.session_state["evaluation_count"] += 1
                    st.session_state["cos_history"].append({
                        "run": st.session_state["evaluation_count"],
                        "composite": cos.composite,
                        "entropy": cos.entropy_score,
                        "symmetry": cos.symmetry_score,
                        "information": cos.information_score,
                        "fairness": cos.fairness_score,
                    })

        with col_info:
            st.caption(
                "COS is computed from: entropy quality, distributional symmetry, "
                "data completeness, and fairness across resource metrics."
            )

        cos = st.session_state.get("last_cos")
        if cos:
            render_cos_panel(cos)
        else:
            st.info("👈 Click **Refresh COS Score** to evaluate your dataset.")

        st.divider()

        # Quick metrics from optimizer
        st.subheader("⚡ Key Performance Indicators")
        kpi_cols = st.columns(4)
        try:
            pue_r = compute_pue(data)
            kpi_cols[0].metric("PUE", f"{pue_r['pue']:.3f}", help="Power Usage Effectiveness (lower=better)")
            kpi_cols[1].metric("Efficiency", pue_r["efficiency_rating"])
        except Exception:
            kpi_cols[0].metric("PUE", "N/A")

        try:
            wue_r = compute_wue(data)
            kpi_cols[2].metric("WUE", f"{wue_r['wue']:.3f}", help="Water Usage Effectiveness")
            kpi_cols[3].metric("WUE Rating", wue_r["rating"])
        except Exception:
            kpi_cols[2].metric("WUE", "N/A")

        try:
            carbon_r = compute_carbon(data)
            st.markdown(f"**🌍 Total CO₂:** {carbon_r['total_co2_tonnes']:.3f} tonnes "
                       f"| Avg Intensity: {carbon_r['avg_carbon_intensity']:.4f} kg/kWh")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # TAB 2: CATERYA Evaluation
    # -----------------------------------------------------------------------
    with tabs[1]:
        st.subheader("⚖️ Full CATERYA Ethical AI Evaluation")
        st.caption(
            "Runs your optimiser through the complete pipeline: "
            "CATERYAEvaluator → QuantumFairnessEvaluator → EthicsSwarm consensus."
        )

        data = get_data()
        integration = get_integration()

        if st.button("▶ Run Full CATERYA Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running full ethical AI evaluation pipeline …"):
                pipeline_result = integration.run_with_evaluation(
                    optimizer_fn=full_optimization,
                    data=data,
                )
                st.session_state["last_cos"] = pipeline_result["cos"]
                st.session_state["last_swarm"] = pipeline_result["swarm"]
                st.session_state["last_fairness"] = pipeline_result["fairness"]
                st.session_state["last_opt_result"] = pipeline_result["result"]
                st.session_state["evaluation_count"] += 1
                st.session_state["cos_history"].append({
                    "run": st.session_state["evaluation_count"],
                    "composite": pipeline_result["cos"].composite,
                    "entropy": pipeline_result["cos"].entropy_score,
                    "symmetry": pipeline_result["cos"].symmetry_score,
                    "information": pipeline_result["cos"].information_score,
                    "fairness": pipeline_result["cos"].fairness_score,
                })

        cos = st.session_state.get("last_cos")
        swarm = st.session_state.get("last_swarm")
        fairness = st.session_state.get("last_fairness")

        if cos:
            st.divider()
            st.markdown("#### 📊 COS Score Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                render_cos_panel(cos)
            with c2:
                render_radar_chart(cos)

        if swarm:
            st.divider()
            st.markdown("#### 🤖 Ethics Swarm Consensus")
            st.info(swarm["summary"])
            render_swarm_chart(swarm)

            with st.expander("🗳️ Individual Agent Rationales"):
                for agent_name, info in swarm["votes"].items():
                    icon = "✅" if info["vote"] >= st.session_state["cos_threshold"] else "⚠️"
                    st.markdown(
                        f"**{icon} {agent_name}** (focus: `{info['focus']}`, "
                        f"weight: `{info['weight']}`) → score `{info['vote']:.4f}`  \n"
                        f"> {info['rationale']}"
                    )

        if fairness:
            st.divider()
            st.markdown("#### ⚖️ Quantum Fairness Evaluation")
            fair_score = fairness.get("fairness_score", 0)
            fair_status = "✅ Passed" if fairness.get("passed") else "⚠️ Failed"
            st.metric("Quantum Fairness Score", f"{fair_score:.4f}", delta=fair_status)

            if fairness.get("disparity_map"):
                disparity_df = pd.DataFrame.from_dict(
                    fairness["disparity_map"], orient="index", columns=["Disparity"]
                ).reset_index().rename(columns={"index": "Column"})
                fig = px.bar(
                    disparity_df, x="Column", y="Disparity",
                    title="Per-column Disparity (lower is fairer)",
                    color="Disparity",
                    color_continuous_scale=["#69f0ae", "#ff9800", "#ff5252"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                    font=dict(color="#e0e0e0"), height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 3: Real-Time Monitor
    # -----------------------------------------------------------------------
    with tabs[2]:
        st.subheader("🔴 Real-Time IoT / Sensor Monitoring")
        st.caption(
            "Live sensor data from Mock / Prometheus / InfluxDB backends. "
            "Auto-alerting with email & Slack notifications."
        )

        monitor = _get_monitor()

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("📡 Poll Sensors Now", type="primary", use_container_width=True):
                with st.spinner("Reading sensors…"):
                    snapshot = monitor.poll_once()
                    st.session_state["monitor_poll_count"] += 1
                    alerts = snapshot.get("alerts", [])
                    if alerts:
                        st.session_state["monitor_alerts"].extend(alerts[-10:])

        with col_p2:
            n_quick = st.number_input("Quick-fill polls", 1, 100, 20, key="quick_fill_n")
        with col_p3:
            if st.button("⚡ Quick-Fill History", use_container_width=True):
                with st.spinner(f"Running {n_quick} poll cycles…"):
                    for _ in range(n_quick):
                        monitor.poll_once()
                    st.session_state["monitor_poll_count"] += n_quick
                    st.success(f"✅ {n_quick} polls completed")

        snapshot = monitor.read_snapshot()
        readings = snapshot.get("readings", {})

        if readings:
            st.markdown(f"**Last update:** `{snapshot.get('timestamp', 'N/A')}`")
            st.divider()

            # Live metric cards
            st.markdown("#### 📊 Live Readings")
            metric_cols = st.columns(4)
            items = list(readings.items())
            for i, (name, info) in enumerate(items[:12]):
                col = metric_cols[i % 4]
                col.metric(
                    name.replace("_", " ").title(),
                    f"{info['value']:.3f} {info.get('unit', '')}",
                )

            # Historical chart
            hist_df = monitor.get_history_df()
            if not hist_df.empty and len(hist_df) > 2:
                st.divider()
                st.markdown("#### 📈 Metric Trend")
                numeric_cols = [c for c in hist_df.columns if c != "timestamp"]
                selected = st.multiselect(
                    "Select metrics to plot",
                    numeric_cols,
                    default=["total_power_kw", "it_power_kw", "temperature_celsius"][:len(numeric_cols)],
                    key="monitor_trend_select",
                )
                if selected:
                    plot_df = hist_df[["timestamp"] + selected].copy()
                    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
                    fig = px.line(
                        plot_df.melt(id_vars="timestamp", value_vars=selected),
                        x="timestamp", y="value", color="variable",
                        title="Real-Time Sensor Trends",
                    )
                    fig.update_layout(
                        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                        font=dict(color="#e0e0e0"), height=360,
                        xaxis=dict(gridcolor="#1e3a5f"),
                        yaxis=dict(gridcolor="#1e3a5f"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Click **Poll Sensors Now** or use the sidebar **Poll Once** button to fetch live data.")

        # Alert log
        st.divider()
        st.markdown("#### 🚨 Alert Log")
        all_alerts = monitor.get_alerts()
        recent_alerts = st.session_state.get("monitor_alerts", [])

        if all_alerts:
            for alert in reversed(all_alerts[-15:]):
                icon = "🔴" if alert.severity == "critical" else "⚠️"
                bg = "#4a0000" if alert.severity == "critical" else "#3a2800"
                st.markdown(
                    f'<div style="background:{bg};padding:0.5rem 1rem;border-radius:8px;'
                    f'margin:0.3rem 0;font-size:0.9rem;">'
                    f'{icon} <b>{alert.timestamp[:19]}</b> — {alert.message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No active alerts — all metrics within thresholds.")

    # -----------------------------------------------------------------------
    # TAB 4: Quantum Thermal Simulation
    # -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("⚛️ Quantum Thermal Distribution Simulation")
        st.caption(
            "Encodes rack/zone temperatures as quantum state amplitudes. "
            "Interference patterns reveal hot-spots and unequal heat distribution. "
            "Fairness metrics computed via QuantumFairnessEvaluator."
        )

        from thermal_circuit import PENNYLANE_AVAILABLE as PL_AVAIL
        if PL_AVAIL:
            st.success("🔬 PennyLane detected — using real quantum circuit simulation.")
        else:
            st.info(
                "🔢 PennyLane not installed — using classical numpy quantum simulation. "
                "Install with: `pip install pennylane`"
            )

        integration = get_integration()

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            n_zones = st.slider("Number of Zones / Racks", 4, 16, 8)
            n_qubits = st.selectbox("Qubits (simulation states)", [2, 3, 4, 5], index=2)
            hotspot_thresh = st.slider("Hot-Spot Threshold", 0.01, 0.30, 0.08, 0.01)
        with col_q2:
            st.markdown("**Manual Temperature Input (optional)**")
            temp_input = st.text_area(
                "Enter zone temperatures (comma-separated):",
                placeholder="22.0, 24.5, 31.2, 19.8, 26.1, 28.3, 21.4, 23.7",
                height=80,
            )

        if st.button("⚛️ Run Quantum Thermal Simulation", type="primary", use_container_width=True):
            with st.spinner("Running quantum circuit simulation…"):
                # Parse manual input or generate random
                if temp_input.strip():
                    try:
                        temps = [float(t.strip()) for t in temp_input.split(",") if t.strip()]
                        n_zones = len(temps)
                    except ValueError:
                        st.error("Invalid temperature input. Use comma-separated numbers.")
                        temps = None
                else:
                    rng = np.random.default_rng()
                    temps = rng.normal(22, 4, n_zones).clip(15, 45).tolist()

                if temps:
                    labels = [f"Rack-{chr(65 + i // 4)}{(i % 4) + 1}" for i in range(len(temps))]
                    sim = QuantumThermalSimulator(
                        n_qubits=n_qubits,
                        hotspot_threshold=hotspot_thresh,
                        provenance=integration.provenance,
                    )
                    result = sim.simulate(temps, zone_labels=labels)
                    st.session_state["last_thermal_result"] = result
                    st.session_state["thermal_history"].append({
                        "timestamp": result.timestamp,
                        "fairness_score": result.fairness_score,
                        "n_hotspots": len(result.hotspot_zones),
                    })

        result = st.session_state.get("last_thermal_result")
        if result:
            st.divider()
            st.info(result.summary())

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Backend", result.backend)
            col_r2.metric("Hotspot Zones", len(result.hotspot_zones))
            col_r3.metric("Fairness Score", f"{result.fairness_score:.4f}")
            col_r4.metric("Circuit Depth", result.circuit_depth)

            if result.hotspot_zones:
                st.warning(f"🌡️ Hot-spot zones detected: **{', '.join(result.hotspot_zones)}** — check cooling in these areas.")
            else:
                st.success("✅ No hot-spots detected — thermal distribution is within acceptable range.")

            # Thermal bar chart
            zone_df = pd.DataFrame({
                "Zone": result.zone_labels,
                "Thermal Score": result.thermal_scores,
                "Is Hotspot": [z in result.hotspot_zones for z in result.zone_labels],
            })
            fig = px.bar(
                zone_df, x="Zone", y="Thermal Score",
                color="Is Hotspot",
                color_discrete_map={True: "#ff5252", False: "#69f0ae"},
                title="Quantum Thermal Distribution by Zone",
            )
            fig.add_hline(
                y=result.hotspot_threshold,
                line_dash="dash", line_color="#ff9800",
                annotation_text=f"Threshold ({result.hotspot_threshold:.3f})",
            )
            fig.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                font=dict(color="#e0e0e0"), height=380,
                xaxis=dict(gridcolor="#1e3a5f"),
                yaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Fairness disparity
            if result.disparity_map:
                st.markdown("#### ⚖️ Zone Fairness Disparity")
                disp_df = pd.DataFrame.from_dict(
                    result.disparity_map, orient="index", columns=["Disparity"]
                ).reset_index().rename(columns={"index": "Zone"})
                fig2 = px.bar(
                    disp_df, x="Zone", y="Disparity",
                    title="Thermal Fairness Disparity per Zone",
                    color="Disparity",
                    color_continuous_scale=["#69f0ae", "#ff9800", "#ff5252"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                    font=dict(color="#e0e0e0"), height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 5: Predictive Maintenance
    # -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("🔮 LSTM Predictive Maintenance")
        st.caption(
            "LSTM-based failure prediction from historical sensor streams. "
            "Wrapped with QuantumSymmetryIndex for energy asymmetry detection."
        )

        from real_time_monitor import TORCH_AVAILABLE as TORCH_AVAIL
        if TORCH_AVAIL:
            st.success("🔥 PyTorch detected — using real LSTM model.")
        else:
            st.info("📐 PyTorch not installed — using Holt's double exponential smoothing fallback.")

        monitor = _get_monitor()

        # Feed data before predicting
        col_pm1, col_pm2 = st.columns(2)
        with col_pm1:
            n_feed = st.slider("Historical samples to feed", 30, 200, 60)
        with col_pm2:
            qsi_thresh = st.slider("QSI Asymmetry Threshold", 0.1, 0.8, 0.35, 0.05)

        if st.button("🔮 Run Predictive Maintenance Analysis", type="primary", use_container_width=True):
            with st.spinner("Training LSTM and generating predictions…"):
                # Feed historical data via mock sensor
                mock_sensor = MockSensorBackend()
                for _ in range(n_feed):
                    readings = mock_sensor.read_all()
                    for r in readings:
                        monitor.predictor.update(r.metric_name, r.value)

                # Get predictions
                target_metrics = [
                    "total_power_kw", "temperature_celsius",
                    "server_utilization_pct", "ups_efficiency_pct",
                    "cooling_power_kw", "humidity_pct",
                ]
                predictions = {}
                for metric in target_metrics:
                    pred = monitor.predictor.predict(metric)
                    if pred:
                        predictions[metric] = pred

                # QSI for each metric
                qsa = QuantumSymmetryAnalyser(asymmetry_threshold=qsi_thresh)
                qsi_results = {}
                for metric in target_metrics:
                    buf = monitor._metric_buffers.get(metric)
                    if buf and len(buf) >= 20:
                        qsi = qsa.analyse(metric, list(buf))
                        qsi_results[metric] = qsi

                st.session_state["_pred_results"] = predictions
                st.session_state["_qsi_results"] = qsi_results

        preds = st.session_state.get("_pred_results", {})
        qsi_map = st.session_state.get("_qsi_results", {})

        if preds:
            st.divider()
            st.markdown("#### 🎯 Failure Probability Predictions")

            pred_rows = []
            for metric, p in preds.items():
                pred_rows.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Predicted Next": f"{p.predicted_next:.3f}",
                    "Failure Prob": f"{p.failure_probability:.3f}",
                    "Maintenance": "🔴 YES" if p.maintenance_recommended else "✅ NO",
                    "Model": p.model_type,
                    "Confidence": f"{p.confidence:.3f}",
                })

            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Failure probability bar chart
            fig = go.Figure()
            metrics_list = [p.metric_name.replace("_", " ").title() for p in preds.values()]
            probs = [p.failure_probability for p in preds.values()]
            colors = ["#ff5252" if p > 0.7 else "#ff9800" if p > 0.4 else "#69f0ae" for p in probs]
            fig.add_trace(go.Bar(x=metrics_list, y=probs, marker_color=colors,
                                text=[f"{p:.3f}" for p in probs], textposition="outside"))
            fig.add_hline(y=0.7, line_dash="dash", line_color="#ff5252",
                         annotation_text="Critical threshold (0.7)")
            fig.update_layout(
                title="Failure Probability by Metric",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                font=dict(color="#e0e0e0"), height=360,
                yaxis=dict(range=[0, 1], gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig, use_container_width=True)

        if qsi_map:
            st.divider()
            st.markdown("#### 🌀 Quantum Symmetry Index (Energy Asymmetry Detection)")
            st.caption(
                "QSI combines classical skewness, quantum probability divergence, "
                "and superposition entropy to detect asymmetric energy patterns."
            )

            qsi_rows = []
            for metric, qsi in qsi_map.items():
                qsi_rows.append({
                    "Metric": metric.replace("_", " ").title(),
                    "QSI Score": f"{qsi.qsi_score:.4f}",
                    "Classical Skew": f"{qsi.classical_skew:.4f}",
                    "Q-Divergence": f"{qsi.quantum_divergence:.4f}",
                    "S-Entropy": f"{qsi.superposition_entropy:.4f}",
                    "Asymmetric?": "⚠️ YES" if qsi.asymmetry_detected else "✅ NO",
                })
            qsi_df = pd.DataFrame(qsi_rows)
            st.dataframe(qsi_df, use_container_width=True, hide_index=True)

            fig_qsi = go.Figure()
            qsi_metrics = [q.metric_name.replace("_", " ").title() for q in qsi_map.values()]
            qsi_scores = [q.qsi_score for q in qsi_map.values()]
            qsi_colors = ["#ff5252" if q > qsi_thresh else "#69f0ae" for q in qsi_scores]
            fig_qsi.add_trace(go.Bar(
                x=qsi_metrics, y=qsi_scores, marker_color=qsi_colors,
                text=[f"{q:.4f}" for q in qsi_scores], textposition="outside",
            ))
            fig_qsi.add_hline(
                y=qsi_thresh, line_dash="dash", line_color="#ff9800",
                annotation_text=f"Asymmetry threshold ({qsi_thresh})",
            )
            fig_qsi.update_layout(
                title="Quantum Symmetry Index by Metric",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                font=dict(color="#e0e0e0"), height=360,
                yaxis=dict(range=[0, 1], gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_qsi, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 6: Analytics
    # -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("📈 Data Center Analytics")
        data = get_data()

        col_a, col_b = st.columns(2)
        with col_a:
            if "total_power_kw" in data.columns and "it_power_kw" in data.columns:
                fig = px.line(
                    data.reset_index(),
                    x=data.index if "timestamp" not in data.columns else "timestamp",
                    y=["total_power_kw", "it_power_kw"],
                    title="Power Consumption Over Time",
                    color_discrete_map={
                        "total_power_kw": "#69f0ae",
                        "it_power_kw": "#40c4ff",
                    },
                )
                fig.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                    font=dict(color="#e0e0e0"), height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "carbon_intensity_kg_per_kwh" in data.columns:
                fig = px.histogram(
                    data, x="carbon_intensity_kg_per_kwh",
                    title="Carbon Intensity Distribution",
                    nbins=30,
                    color_discrete_sequence=["#ff9800"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                    font=dict(color="#e0e0e0"), height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

        if "server_utilization_pct" in data.columns:
            fig = px.box(
                data.melt(
                    value_vars=[
                        c for c in ["server_utilization_pct", "temperature_celsius",
                                    "cooling_power_kw"]
                        if c in data.columns
                    ]
                ),
                x="variable", y="value",
                title="Resource Distribution (Box Plot)",
                color="variable",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#0d2137",
                font=dict(color="#e0e0e0"), height=340, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 📉 COS Score Trend")
        render_cos_history()

    # -----------------------------------------------------------------------
    # TAB 7: Audit & Provenance
    # -----------------------------------------------------------------------
    with tabs[6]:
        st.subheader("🔒 Audit Trail & Provenance Chain")
        integration = get_integration()

        col1, col2, col3 = st.columns(3)
        report = integration.get_audit_report()
        col1.metric("Chain Entries", report["chain_length"])
        col2.metric("Integrity", "✅ Valid" if report["chain_integrity"] else "⛔ Tampered")
        col3.metric("Model ID", report["model_id"])

        st.caption(f"Head Hash: `{report['head_hash']}`")

        if st.button("📥 Export Audit Report (JSON)"):
            audit_json = integration.get_audit_json()
            st.download_button(
                label="⬇️ Download audit_report.json",
                data=audit_json,
                file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        if report["events"]:
            with st.expander(f"📋 View All {len(report['events'])} Provenance Events"):
                for event in reversed(report["events"]):
                    st.markdown(
                        f"**[{event['seq']:03d}]** `{event['event_type']}` "
                        f"— {event['timestamp']}  \n"
                        f"Hash: `{event['hash'][:24]}…`"
                    )

    # -----------------------------------------------------------------------
    # TAB 8: Cloud Scaling
    # -----------------------------------------------------------------------
    with tabs[7]:
        st.subheader("☁️ AWS Auto-Scaling")
        integration = get_integration()

        if not integration.is_aws_available:
            st.warning(
                "**boto3 not installed.** Simulation mode active.  \n"
                "Install with: `pip install boto3`  \n"
                "Then configure AWS credentials: `aws configure`"
            )

        st.markdown("Configure and trigger AWS Auto Scaling Group adjustments based on workload.")

        with st.form("aws_scale_form"):
            asg_name = st.text_input("Auto Scaling Group Name", value="eco-ai-datacenter-asg")
            desired = st.slider("Desired Capacity (instances)", 1, 50, 5)
            min_s = st.number_input("Min Size", 1, 20, 1)
            max_s = st.number_input("Max Size", 1, 100, 20)
            submitted = st.form_submit_button("🚀 Trigger Scale-Out", type="primary")

        if submitted:
            with st.spinner("Sending scale request …"):
                result = integration.trigger_aws_scale_out(
                    asg_name=asg_name,
                    desired_capacity=desired,
                    min_size=min_s,
                    max_size=max_s,
                )
            if result["status"] == "success":
                st.success(f"✅ Auto Scaling Group **{result['asg_name']}** scaled to **{result['desired']}** instances.")
            elif result["status"] == "simulated":
                st.info(f"🔮 **Simulation:** Would scale **{result['asg_name']}** to **{result['desired']}** instances.  \n"
                       f"_{result.get('note', '')}_")
            else:
                st.error(f"❌ Scaling failed: {result.get('error', 'Unknown error')}")

    # -----------------------------------------------------------------------
    # TAB 9: Background Tasks
    # -----------------------------------------------------------------------
    with tabs[8]:
        st.subheader("⚙️ Celery Background Tasks")

        task_status = get_task_status()
        mode_color = "#69f0ae" if task_status["celery_available"] else "#ff9800"
        st.markdown(
            f'<div style="background:#0d2137;padding:1rem;border-radius:10px;'
            f'border-left:4px solid {mode_color};">'
            f'<b>Mode:</b> <code>{task_status["mode"]}</code><br>'
            f'<b>Broker:</b> <code>{task_status["broker"]}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not task_status["celery_available"]:
            st.info(
                "**Celery not installed.** All tasks run synchronously (blocking).  \n"
                "For production: `pip install celery redis`  \n"
                "Then start worker: `celery -A workers.tasks worker --loglevel=info`  \n"
                "And beat: `celery -A workers.tasks beat --loglevel=info`"
            )

        st.divider()
        st.markdown("#### 🚀 Manual Task Triggers")

        t_cols = st.columns(2)

        with t_cols[0]:
            if st.button("📡 Monitor Poll Task", use_container_width=True):
                with st.spinner("Running monitor poll task…"):
                    task_result = run_monitor_poll.delay(backend="mock", n_polls=5)
                    result = task_result.get()
                    st.success(f"✅ Polls: {result['polls']} | Metrics: {len(result.get('last_snapshot', {}).get('readings', {}))}")
                    st.session_state["last_task_result"] = result

            if st.button("⚛️ Quantum Thermal Task", use_container_width=True):
                with st.spinner("Running quantum thermal task…"):
                    task_result = run_quantum_thermal.delay(n_zones=8)
                    result = task_result.get()
                    st.success(
                        f"✅ Backend: {result['backend']} | "
                        f"Hotspots: {len(result['hotspot_zones'])} | "
                        f"Fairness: {result['fairness_score']:.4f}"
                    )
                    st.session_state["last_task_result"] = result

        with t_cols[1]:
            if st.button("⚖️ COS Evaluation Task", use_container_width=True):
                with st.spinner("Running COS evaluation task…"):
                    task_result = run_cos_evaluation.delay(n_rows=200)
                    result = task_result.get()
                    st.success(
                        f"✅ COS: {result['cos_composite']:.4f} | "
                        f"{'Passed' if result['cos_passed'] else 'Failed'} | "
                        f"Swarm: {result['swarm_consensus']:.4f}"
                    )
                    st.session_state["last_task_result"] = result

            if st.button("🔮 Predictive Maintenance Task", use_container_width=True):
                with st.spinner("Running predictive maintenance task…"):
                    task_result = run_predictive_check.delay(n_history=60)
                    result = task_result.get()
                    alerts = result.get("maintenance_alerts", [])
                    if alerts:
                        st.warning(f"⚠️ Maintenance recommended for: {', '.join(alerts)}")
                    else:
                        st.success("✅ No maintenance alerts — all metrics nominal.")
                    st.session_state["last_task_result"] = result

        last_result = st.session_state.get("last_task_result")
        if last_result:
            with st.expander("📋 Last Task Result (raw)"):
                st.json(last_result)

    # -----------------------------------------------------------------------
    # TAB 10: Security & Auth
    # -----------------------------------------------------------------------
    with tabs[9]:
        st.subheader("🛡️ Security & Access Control")
        auth_svc = _get_auth_service()
        audit = _get_audit_logger()
        enc_engine = _get_encryption_engine()

        # Login panel
        current_user = _current_user()

        if current_user is None:
            st.markdown("#### 🔐 Login")
            st.info("Login to access role-gated features. All logins are audited.")

            col_l1, col_l2 = st.columns(2)
            with col_l1:
                username = st.text_input("Username", key="login_username",
                                         placeholder="e.g. analyst, admin, compliance_officer")
            with col_l2:
                password = st.text_input("Password", type="password", key="login_password")

            # Demo credentials hint
            with st.expander("💡 Demo Credentials"):
                cred_data = {
                    "Username":   ["admin", "analyst", "compliance_officer", "engineer", "viewer"],
                    "Password":   ["Admin@EcoAI2025!", "Analyst@EcoAI2025!", "Comply@EcoAI2025!",
                                   "Engineer@EcoAI2025!", "Viewer@EcoAI2025!"],
                    "Role":       ["Admin", "Analyst", "Compliance", "Engineer", "Viewer"],
                }
                st.dataframe(pd.DataFrame(cred_data), use_container_width=True, hide_index=True)

            if st.button("🔑 Login", type="primary", use_container_width=True):
                result = auth_svc.login(username, password)
                if result:
                    st.session_state["current_user"] = auth_svc.get_current_user(result["access_token"])
                    st.session_state["auth_token"] = result["access_token"]
                    audit.log_login(username, success=True)
                    st.success(f"✅ Logged in as **{username}** ({result['user']['role']})")
                    st.rerun()
                else:
                    audit.log_login(username, success=False)
                    st.error("❌ Invalid credentials. Check the demo credentials above.")
        else:
            # Logged in view
            st.markdown(f"#### 👤 Logged in as: **{current_user.username}** `{current_user.role.value}`")
            col_u1, col_u2, col_u3, col_u4 = st.columns(4)
            col_u1.metric("Role", current_user.role.value.title())
            col_u2.metric("Department", current_user.department or "—")
            col_u3.metric("User ID", current_user.user_id)
            col_u4.metric("Active", "✅ Yes")

            if st.button("🚪 Logout", type="secondary"):
                token = st.session_state.get("auth_token", "")
                auth_svc.logout(token)
                audit.log(AuditEventType.LOGOUT, current_user.username, "Logout", "auth", "success")
                st.session_state["current_user"] = None
                st.session_state["auth_token"] = None
                st.rerun()

            st.divider()
            st.markdown("#### 🔐 Permission Matrix")
            st.caption("Your role's access rights in this system.")
            from auth import PERMISSIONS
            perm_rows = []
            for perm, roles in sorted(PERMISSIONS.items()):
                has_it = current_user.role in roles
                perm_rows.append({
                    "Permission": perm,
                    "Access": "✅ Granted" if has_it else "🔒 Denied",
                    "Allowed Roles": ", ".join(r.value for r in roles),
                })
            perm_df = pd.DataFrame(perm_rows)
            st.dataframe(perm_df, use_container_width=True, hide_index=True, height=300)

        st.divider()
        st.markdown("#### 🔒 Data Encryption")
        st.caption(f"Encryption backend: {'AES-256-GCM (cryptography lib)' if enc_engine else 'XOR-HMAC fallback'}")

        enc_col1, enc_col2 = st.columns(2)
        with enc_col1:
            plaintext_input = st.text_area("Text to encrypt:", placeholder="Enter sensitive data here…", height=80)
            if st.button("🔒 Encrypt", use_container_width=True):
                if plaintext_input:
                    encrypted = enc_engine.encrypt_string(plaintext_input, actor=current_user.username if current_user else "anonymous")
                    st.session_state["last_encrypted"] = encrypted
                    st.code(encrypted[:60] + "…", language=None)
                    st.success("✅ Encrypted with AES-256-GCM")
        with enc_col2:
            enc_input = st.text_area("Ciphertext to decrypt:", placeholder="Paste encrypted base64 here…", height=80)
            if st.button("🔓 Decrypt", use_container_width=True):
                if enc_input:
                    try:
                        decrypted = enc_engine.decrypt_string(enc_input, actor=current_user.username if current_user else "anonymous")
                        st.success(f"✅ Decrypted: `{decrypted[:80]}`")
                    except Exception as e:
                        st.error(f"❌ Decryption failed: {e}")

        st.divider()
        st.markdown("#### 📋 Security Audit Log")
        audit_entries = audit.get_recent(30)
        if audit_entries:
            audit_rows = []
            for e in reversed(audit_entries[-15:]):
                risk_emoji = "🔴" if e.risk_score >= 0.7 else "🟠" if e.risk_score >= 0.4 else "🟢"
                audit_rows.append({
                    "Time":       e.timestamp[11:19],
                    "Event":      e.event_type.value,
                    "Actor":      e.actor,
                    "Action":     e.action[:50],
                    "Outcome":    e.outcome,
                    "Risk":       f"{risk_emoji} {e.risk_score:.2f}",
                })
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

            audit_stats = audit.get_summary_stats()
            s_cols = st.columns(4)
            s_cols[0].metric("Total Events", audit_stats.get("total_entries", 0))
            s_cols[1].metric("High Risk", audit_stats.get("high_risk_events", 0))
            s_cols[2].metric("Avg Risk", f"{audit_stats.get('avg_risk_score', 0):.3f}")
            failed_logins = sum(audit_stats.get("failed_logins", {}).values())
            s_cols[3].metric("Failed Logins", failed_logins)

            if st.button("📥 Export Audit Log (JSONL)"):
                st.download_button(
                    "⬇️ Download audit_log.jsonl",
                    data=audit.export_jsonl(),
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                    mime="application/x-ndjson",
                )
        else:
            st.info("No audit events yet — login or trigger an operation above.")

    # -----------------------------------------------------------------------
    # TAB 11: Compliance
    # -----------------------------------------------------------------------
    with tabs[10]:
        st.subheader("✅ Compliance Scanner")
        st.caption(
            "Automated compliance checks for GDPR, ISO 27001, SRN PPI (Indonesia), "
            "and SOC 2. Results are validated through EthicsSwarm consensus."
        )

        scanner = _get_compliance_scanner()
        audit = _get_audit_logger()
        current_user = _current_user()

        # Check compliance permission
        can_scan = current_user is None or current_user.has_permission("compliance:scan")
        if not can_scan:
            st.error("🔒 Permission denied: Compliance scanning requires `compliance` or `admin` role.")
        else:
            selected_frameworks = st.multiselect(
                "Select frameworks to scan:",
                options=list(FRAMEWORK_REGISTRY.keys()),
                default=["GDPR", "ISO27001"],
                format_func=lambda x: {
                    "GDPR": "🇪🇺 GDPR (EU Data Protection)",
                    "ISO27001": "🏢 ISO 27001 (Info Security)",
                    "SRN_PPI": "🇮🇩 SRN PPI (Indonesia Privacy Law)",
                    "SOC2": "🏦 SOC 2 Type II",
                }.get(x, x),
            )

            if st.button("🔍 Run Compliance Scan", type="primary", use_container_width=True):
                if not selected_frameworks:
                    st.warning("Please select at least one framework.")
                else:
                    with st.spinner(f"Scanning {len(selected_frameworks)} framework(s)…"):
                        results = {}
                        for fw in selected_frameworks:
                            results[fw] = scanner.scan_framework(fw)
                            actor = current_user.username if current_user else "anonymous"
                            passed = results[fw].overall_status.value in ("PASSED", "WARNING", "PARTIAL")
                            audit.log_compliance_scan(actor, fw, passed, results[fw].failed)
                        st.session_state["last_compliance_results"] = results
                    st.success(f"✅ Scanned {len(results)} frameworks.")

            results = st.session_state.get("last_compliance_results", {})
            if results:
                st.divider()

                # Overview cards
                fw_cols = st.columns(len(results))
                for i, (fw, result) in enumerate(results.items()):
                    score_color = "#69f0ae" if result.compliance_score >= 0.8 else "#ff9800" if result.compliance_score >= 0.5 else "#ff5252"
                    fw_cols[i].metric(
                        fw,
                        f"{result.compliance_score:.1%}",
                        delta=f"{result.passed}/{result.total_controls} passed",
                    )

                # Detail tabs per framework
                if len(results) > 0:
                    fw_detail_tabs = st.tabs(list(results.keys()))
                    for i, (fw, result) in enumerate(results.items()):
                        with fw_detail_tabs[i]:
                            st.markdown(f"**{result.summary()}**")

                            d_cols = st.columns(5)
                            d_cols[0].metric("Score", f"{result.compliance_score:.1%}")
                            d_cols[1].metric("Passed", result.passed)
                            d_cols[2].metric("Failed", result.failed)
                            d_cols[3].metric("Warnings", result.warnings)
                            d_cols[4].metric("Swarm", f"{'✅' if result.swarm_approved else '⚠️'} {result.swarm_consensus:.4f}")

                            # Findings table
                            findings_df = pd.DataFrame([
                                {
                                    "Control": f.control_id,
                                    "Name": f.control_name,
                                    "Status": f.status.value,
                                    "Severity": f.severity.value,
                                    "Article": f.article_ref,
                                    "Remediation": f.remediation[:60] + "…" if len(f.remediation) > 60 else f.remediation,
                                }
                                for f in result.findings
                            ])
                            st.dataframe(findings_df, use_container_width=True, hide_index=True, height=280)

                            if result.recommendations:
                                st.markdown("**📋 Top Recommendations:**")
                                for rec in result.recommendations[:6]:
                                    st.markdown(f"- {rec}")

                            # Export
                            if st.button(f"📥 Export {fw} Report", key=f"export_{fw}"):
                                export_data = json.dumps({
                                    "framework": fw,
                                    "scan_timestamp": result.scan_timestamp,
                                    "compliance_score": result.compliance_score,
                                    "overall_status": result.overall_status.value,
                                    "swarm_consensus": result.swarm_consensus,
                                    "findings": [
                                        {"id": f.control_id, "name": f.control_name,
                                         "status": f.status.value, "severity": f.severity.value,
                                         "article": f.article_ref, "remediation": f.remediation}
                                        for f in result.findings
                                    ],
                                    "recommendations": result.recommendations,
                                }, indent=2)
                                st.download_button(
                                    f"⬇️ {fw}_compliance_{datetime.now().strftime('%Y%m%d')}.json",
                                    data=export_data,
                                    file_name=f"{fw}_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                )

    # -----------------------------------------------------------------------
    # TAB 12: Version Control
    # -----------------------------------------------------------------------
    with tabs[11]:
        st.subheader("🗂️ Simulation Version Control")
        st.caption("Git-inspired version control for simulation states. Track, compare, and branch experiments.")

        vc = _get_simulation_vc()
        current_user = _current_user()

        col_vc1, col_vc2, col_vc3, col_vc4 = st.columns(4)
        col_vc1.metric("Total Commits", vc.total_commits)
        col_vc2.metric("Branches", len(vc.branches))
        col_vc3.metric("Tags", len(vc.all_tags))
        col_vc4.metric("Current Branch", f"🌿 {vc.current_branch}")

        st.divider()
        vc_tabs = st.tabs(["📝 Commit", "🌿 Branches", "📜 Log", "🔍 Diff", "🏷️ Tags", "📦 Export"])

        with vc_tabs[0]:  # Commit
            st.markdown("#### Save Current Simulation State")
            data = get_data()
            commit_msg = st.text_input("Commit message:", placeholder="e.g. Improved cooling config — PUE reduced to 1.3")
            author_name = current_user.username if current_user else st.text_input("Author:", value="analyst", key="vc_commit_author")

            if st.button("💾 Commit Snapshot", type="primary", use_container_width=True):
                if not commit_msg:
                    st.warning("Please enter a commit message.")
                else:
                    with st.spinner("Snapshotting simulation state…"):
                        integration = get_integration()
                        cos = integration.score_data(data)
                        try:
                            pue_r = compute_pue(data)
                            pue_val = pue_r.get("pue")
                        except Exception:
                            pue_val = None
                        try:
                            carbon_r = compute_carbon(data)
                            carbon_val = carbon_r.get("total_co2_tonnes")
                        except Exception:
                            carbon_val = None

                        state = {
                            "cos_score": cos.composite,
                            "cos_passed": cos.passed,
                            "pue": pue_val,
                            "carbon_tonnes": carbon_val,
                            "n_rows": len(data),
                            "columns": list(data.columns),
                            "data_hash": hash_data(data),
                        }
                        snap_id = vc.commit(
                            state, commit_msg,
                            author=author_name if isinstance(author_name, str) else author_name,
                            cos_score=cos.composite,
                        )
                        if current_user:
                            _get_audit_logger().log(
                                AuditEventType.SIMULATION_RUN,
                                current_user.username,
                                f"Simulation committed: {commit_msg}",
                                "simulation_vc",
                                details={"snapshot_id": snap_id[:16], "cos": cos.composite},
                            )
                        st.success(f"✅ Committed `{snap_id[:12]}…` — COS: {cos.composite:.4f}")

        with vc_tabs[1]:  # Branches
            st.markdown("#### Branch Management")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**Existing branches:**")
                for branch in vc.branches:
                    head = vc._branches.get(branch, "")
                    marker = "🌿 current" if branch == vc.current_branch else ""
                    st.markdown(f"- `{branch}` — HEAD: `{head[:8] if head else 'empty'}` {marker}")
            with bc2:
                new_branch_name = st.text_input("New branch name:", placeholder="experiment-cooling-v2")
                if st.button("🌿 Create Branch", use_container_width=True):
                    if new_branch_name:
                        vc.branch(new_branch_name)
                        st.success(f"✅ Branch `{new_branch_name}` created.")
                        st.rerun()

                switch_to = st.selectbox("Switch to branch:", vc.branches, key="branch_switch")
                if st.button("⇄ Switch Branch", use_container_width=True):
                    vc.switch_branch(switch_to)
                    st.success(f"✅ Switched to `{switch_to}`.")
                    st.rerun()

        with vc_tabs[2]:  # Log
            st.markdown(f"#### Commit History — `{vc.current_branch}`")
            history = vc.log(n=20)
            if history:
                log_rows = [
                    {
                        "ID": s.short_id(),
                        "Author": s.author,
                        "Message": s.message,
                        "COS Score": f"{s.cos_score:.4f}" if s.cos_score else "—",
                        "Tags": ", ".join(s.tags) if s.tags else "—",
                        "Time": s.timestamp[11:19],
                        "Date": s.timestamp[:10],
                    }
                    for s in history
                ]
                st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No commits yet on this branch.")

        with vc_tabs[3]:  # Diff
            st.markdown("#### Compare Two Snapshots")
            if vc.total_commits >= 2:
                all_snaps = list(vc._snapshots.keys())
                dc1, dc2 = st.columns(2)
                from_sel = dc1.selectbox("From:", all_snaps, format_func=lambda x: x[:12])
                to_sel   = dc2.selectbox("To:",   all_snaps, format_func=lambda x: x[:12], index=min(1, len(all_snaps)-1))

                if st.button("🔍 Compute Diff", use_container_width=True):
                    diff = vc.diff(from_sel, to_sel)
                    st.info(diff.summary())
                    if diff.added:
                        st.markdown("**➕ Added:**")
                        st.json(diff.added)
                    if diff.removed:
                        st.markdown("**➖ Removed:**")
                        st.json(diff.removed)
                    if diff.changed:
                        st.markdown("**🔄 Changed:**")
                        st.json({k: {"from": v[0], "to": v[1]} for k, v in diff.changed.items()})
            else:
                st.info("Need at least 2 commits to diff.")

        with vc_tabs[4]:  # Tags
            st.markdown("#### Tag Management")
            if vc.all_tags:
                for tag, snap_id in vc.all_tags.items():
                    st.markdown(f"🏷️ **{tag}** → `{snap_id[:12]}…`")
            else:
                st.info("No tags yet.")

            tag_label = st.text_input("New tag:", placeholder="v1.0-baseline")
            if st.button("🏷️ Tag Current HEAD", use_container_width=True):
                if tag_label and vc.head:
                    vc.tag(tag_label)
                    st.success(f"✅ Tagged HEAD as `{tag_label}`")
                    st.rerun()
                elif not vc.head:
                    st.warning("No commits to tag yet.")

        with vc_tabs[5]:  # Export
            st.markdown("#### Export Version History")
            bundle_json = vc.export_bundle_json()
            st.download_button(
                "⬇️ Download VC Bundle (JSON)",
                data=bundle_json,
                file_name=f"simulation_vc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            with st.expander("Preview bundle structure"):
                bundle = vc.export_bundle()
                st.json({k: v for k, v in bundle.items() if k != "snapshots"})

    # -----------------------------------------------------------------------
    # TAB 13: Reports
    # -----------------------------------------------------------------------
    with tabs[12]:
        st.subheader("📋 Shareable Reports")
        st.caption("Generate self-contained HTML reports with KPIs, COS scores, and compliance summaries.")

        gen = _get_report_generator()
        current_user = _current_user()
        audit = _get_audit_logger()

        can_report = current_user is None or current_user.has_permission("reports:create")
        if not can_report:
            st.error("🔒 Permission denied: Report creation requires `analyst` role or higher.")
        else:
            rc1, rc2 = st.columns(2)
            with rc1:
                report_title = st.text_input("Report Title:", value="Eco AI Data Center — Operational Report")
                report_type = st.selectbox("Report Type:", ["full", "kpi", "compliance", "simulation"])
            with rc2:
                report_author = st.text_input(
                    "Author:", value=current_user.username if current_user else "analyst",
                    key="report_author_input"
                )
                incl_compliance = st.checkbox("Include compliance data", value=True)

            if st.button("📋 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Building report…"):
                    data = get_data()
                    integration = get_integration()

                    # Gather data
                    cos = integration.score_data(data)
                    kpi_data = {}
                    try: kpi_data["pue"] = compute_pue(data)
                    except Exception: pass
                    try: kpi_data["wue"] = compute_wue(data)
                    except Exception: pass
                    try: kpi_data["carbon"] = compute_carbon(data)
                    except Exception: pass

                    cos_data = cos.to_dict() if hasattr(cos, "to_dict") else {}

                    compliance_data = {}
                    if incl_compliance:
                        scanner = _get_compliance_scanner()
                        for fw in ["GDPR", "SRN_PPI"]:
                            result = scanner.scan_framework(fw)
                            compliance_data[fw] = {
                                "overall_status": result.overall_status.value,
                                "compliance_score": result.compliance_score,
                                "passed": result.passed,
                                "failed": result.failed,
                                "total_controls": result.total_controls,
                                "swarm_approved": result.swarm_approved,
                                "swarm_consensus": result.swarm_consensus,
                            }

                    config = ReportConfig(
                        title=report_title,
                        author=report_author,
                        report_type=report_type,
                        include_compliance=incl_compliance,
                        cos_threshold=st.session_state["cos_threshold"],
                    )

                    report = gen.generate(config, kpi_data=kpi_data, cos_data=cos_data,
                                          compliance_data=compliance_data)
                    st.session_state["last_report"] = report

                    if current_user:
                        audit.log(
                            AuditEventType.DATA_EXPORT, current_user.username,
                            f"Report generated: {report['report_id']}",
                            "reports",
                            details={"report_id": report["report_id"]},
                        )
                    st.success(f"✅ Report generated! ID: `{report['report_id']}`")

            last_report = st.session_state.get("last_report")
            if last_report:
                st.divider()
                rc_cols = st.columns(3)
                rc_cols[0].metric("Report ID", last_report["report_id"])
                rc_cols[1].metric("Share Token", last_report["share_token"][:12] + "…")
                rc_cols[2].metric("API URL", last_report["share_url"])

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ Download HTML Report",
                        data=last_report["html"].encode(),
                        file_name=f"eco_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                    )
                with dl_col2:
                    st.download_button(
                        "⬇️ Download JSON Bundle",
                        data=json.dumps(last_report["json_bundle"], indent=2, default=str),
                        file_name=f"eco_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

                with st.expander("🖼️ Report Preview"):
                    st.components.v1.html(last_report["html"], height=500, scrolling=True)

            # Existing reports
            existing = list_reports()
            if existing:
                st.divider()
                st.markdown("#### 📚 Report Registry")
                st.dataframe(pd.DataFrame(existing), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # TAB 14: Data
    # -----------------------------------------------------------------------
    with tabs[13]:
        st.subheader("📁 Dataset Management")

        col_upload, col_sample = st.columns(2)
        with col_upload:
            uploaded = st.file_uploader(
                "Upload CSV Dataset",
                type=["csv"],
                help="Upload your data center metrics CSV file.",
            )
            if uploaded:
                df_uploaded = pd.read_csv(uploaded)
                st.session_state["uploaded_data"] = df_uploaded
                st.success(f"✅ Loaded {len(df_uploaded):,} rows × {len(df_uploaded.columns)} columns.")

        with col_sample:
            n_rows = st.slider("Sample dataset size", 50, 2000, 100, 50)
            if st.button("🎲 Use Sample Data"):
                st.session_state["uploaded_data"] = None
                st.success(f"Using generated sample data ({n_rows} rows).")

        st.divider()
        data = get_data()
        st.markdown(f"**Active Dataset:** {len(data):,} rows × {len(data.columns)} columns")

        with st.expander("👁️ Preview Data (first 20 rows)"):
            st.dataframe(data.head(20), use_container_width=True)

        with st.expander("📊 Descriptive Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

        csv_bytes = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Current Dataset (CSV)",
            data=csv_bytes,
            file_name="eco_ai_datacenter_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
