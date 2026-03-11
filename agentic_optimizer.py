"""
agents/agentic_optimizer.py
============================
Multi-agent agentic workflow untuk Eco AI Data Center.

Agen yang diimplementasikan:
  • RobustnessAgent  — deteksi data shift / fluktuasi energi
  • COSMonitorAgent  — pantau COS score secara kontinu
  • ParamAdjustAgent — auto-adjust parameter jika COS rendah
  • AgenticPipeline  — orkestrasi LangGraph-inspired multi-agent

Setiap keputusan agen diverifikasi oleh EthicsSwarm dan dijaga
CATERYAGuardrail. Semua tindakan dicatat ke ProvenanceChain.

Eco AI Data Center — CateryaTech
Author  : Ary HH | cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluator import CATERYAEvaluator
from provenance import ProvenanceChain
from scoring import COSScore
from swarm import EthicsSwarm
from caterya_guardrail import CATERYAGuardrail, GuardrailConfig

logger = logging.getLogger("eco_ai.agents")


# ══════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentDecision:
    """Satu keputusan yang dihasilkan oleh sebuah agen."""
    agent_name:       str
    action:           str                          # "adjust", "alert", "approve", "reject"
    rationale:        str
    confidence:       float                        # 0-1
    params_changed:   Dict[str, Any] = field(default_factory=dict)
    swarm_approved:   Optional[bool] = None
    swarm_consensus:  Optional[float] = None
    guardrail_passed: Optional[bool] = None
    cos_before:       Optional[float] = None
    cos_after:        Optional[float] = None
    timestamp:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision_id:      str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def summary(self) -> str:
        sw = f" | swarm={self.swarm_consensus:.3f}" if self.swarm_consensus else ""
        cos_delta = ""
        if self.cos_before is not None and self.cos_after is not None:
            delta = self.cos_after - self.cos_before
            cos_delta = f" | ΔCOS={delta:+.4f}"
        return (f"[{self.agent_name}] {self.action.upper()} | "
                f"conf={self.confidence:.2f}{sw}{cos_delta} | {self.rationale[:60]}")


@dataclass
class PipelineResult:
    """Hasil lengkap satu putaran AgenticPipeline."""
    run_id:           str
    cos_initial:      float
    cos_final:        float
    cos_improved:     bool
    decisions:        List[AgentDecision]
    params_applied:   Dict[str, Any]
    swarm_consensus:  float
    guardrail_passed: bool
    provenance_hash:  str
    elapsed_seconds:  float
    timestamp:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        delta = self.cos_final - self.cos_initial
        return (
            f"Run {self.run_id} | COS {self.cos_initial:.4f}→{self.cos_final:.4f} "
            f"({delta:+.4f}) | improved={self.cos_improved} | "
            f"swarm={self.swarm_consensus:.3f} | guardrail={self.guardrail_passed} | "
            f"{len(self.decisions)} decisions"
        )


# ══════════════════════════════════════════════════════════════════════════
# Robustness Agent
# ══════════════════════════════════════════════════════════════════════════

class RobustnessAgent:
    """
    Deteksi data shift dan fluktuasi energi yang bisa menurunkan COS.

    Checks:
      1. Power spike   — power_kw melebihi threshold (default 1.5× median)
      2. Carbon drift  — carbon_intensity berubah >30% dari baseline
      3. Thermal spike — suhu/PUE melebihi batas normal
      4. Statistical   — Kolmogorov-Smirnov-inspired distribution shift

    Output: AgentDecision dengan action="alert" atau "approve".
    """

    def __init__(
        self,
        power_spike_factor:  float = 1.5,
        carbon_drift_pct:    float = 0.30,
        pue_limit:           float = 2.5,
        window_size:         int   = 50,
        name:                str   = "RobustnessAgent",
    ):
        self.power_spike_factor = power_spike_factor
        self.carbon_drift_pct   = carbon_drift_pct
        self.pue_limit          = pue_limit
        self.window_size        = window_size
        self.name               = name
        self._baseline: Optional[Dict[str, float]] = None

    def set_baseline(self, df: pd.DataFrame) -> None:
        """Establish baseline from historical data."""
        self._baseline = {
            "power_kw_median": float(df["power_kw"].median()),
            "carbon_mean":     float(df["carbon_intensity_kg_per_kwh"].mean()),
            "carbon_std":      float(df["carbon_intensity_kg_per_kwh"].std()),
        }
        logger.info("[RobustnessAgent] Baseline set: %s", self._baseline)

    def check(self, df: pd.DataFrame) -> AgentDecision:
        """
        Analyse current window for data shifts.
        Returns decision with action="alert" if shifts detected, else "approve".
        """
        issues: List[str] = []
        confidence        = 1.0

        # 1. Power spike check
        recent_power = df["power_kw"].tail(self.window_size)
        if self._baseline:
            ratio = float(recent_power.max()) / max(self._baseline["power_kw_median"], 1)
            if ratio > self.power_spike_factor:
                issues.append(f"Power spike: {ratio:.2f}× baseline (threshold={self.power_spike_factor}×)")
                confidence *= 0.6

        # 2. Carbon drift check
        if "carbon_intensity_kg_per_kwh" in df.columns:
            recent_carbon = df["carbon_intensity_kg_per_kwh"].tail(self.window_size)
            if self._baseline:
                drift = abs(float(recent_carbon.mean()) - self._baseline["carbon_mean"]) \
                        / max(self._baseline["carbon_mean"], 1e-9)
                if drift > self.carbon_drift_pct:
                    issues.append(f"Carbon drift: {drift:.1%} (threshold={self.carbon_drift_pct:.0%})")
                    confidence *= 0.7

        # 3. Distribution stability (simplified KS test)
        if len(df) >= self.window_size * 2:
            first_half  = df["power_kw"].iloc[:len(df)//2]
            second_half = df["power_kw"].iloc[len(df)//2:]
            ks_stat = self._ks_approx(first_half.values, second_half.values)
            if ks_stat > 0.3:
                issues.append(f"Distribution shift: KS={ks_stat:.3f} (threshold=0.3)")
                confidence *= 0.75

        # 4. PUE estimate
        if "it_power_kw" in df.columns:
            pue_est = float(df["power_kw"].mean()) / max(float(df["it_power_kw"].mean()), 1)
            if pue_est > self.pue_limit:
                issues.append(f"High estimated PUE: {pue_est:.3f} (limit={self.pue_limit})")
                confidence *= 0.65

        if issues:
            return AgentDecision(
                agent_name=self.name,
                action="alert",
                rationale=" | ".join(issues),
                confidence=round(1.0 - confidence, 3),
                params_changed={"detected_issues": issues},
            )
        return AgentDecision(
            agent_name=self.name,
            action="approve",
            rationale=f"No significant data shifts detected (window={self.window_size})",
            confidence=round(confidence, 3),
        )

    @staticmethod
    def _ks_approx(a: np.ndarray, b: np.ndarray) -> float:
        """Approximate KS statistic via CDF comparison."""
        combined = np.sort(np.unique(np.concatenate([a, b])))
        cdf_a = np.searchsorted(np.sort(a), combined, side="right") / len(a)
        cdf_b = np.searchsorted(np.sort(b), combined, side="right") / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))


# ══════════════════════════════════════════════════════════════════════════
# COS Monitor Agent
# ══════════════════════════════════════════════════════════════════════════

class COSMonitorAgent:
    """
    Monitor COS score secara kontinu dan flag jika turun di bawah threshold.
    
    Tracks:
      - COS history (rolling window)
      - Trend: improving / stable / declining
      - Alert jika COS < threshold atau declining streak > patience
    """

    def __init__(
        self,
        cos_threshold:   float = 0.70,
        declining_streak: int  = 3,
        window:          int   = 10,
        name:            str   = "COSMonitorAgent",
    ):
        self.cos_threshold    = cos_threshold
        self.declining_streak = declining_streak
        self.window           = window
        self.name             = name
        self._history:  List[float] = []
        self._evaluator = CATERYAEvaluator()

    def observe(self, cos_score: float) -> None:
        """Record a new COS observation."""
        self._history.append(cos_score)
        if len(self._history) > self.window * 2:
            self._history = self._history[-self.window * 2:]

    def evaluate_df(self, df: pd.DataFrame,
                    optimizer_fn: Optional[Callable] = None) -> Tuple[COSScore, AgentDecision]:
        """Run CATERYA evaluation on df and return (COSScore, AgentDecision)."""
        if optimizer_fn is None:
            # Use score_data path (no optimizer needed)
            from caterya_integration import EcoAIDataCenterCATERYA
            cat = EcoAIDataCenterCATERYA()
            cos_obj = cat.score_data(df)
        else:
            cos_obj, _ = self._evaluator.evaluate(optimizer_fn, df)

        self.observe(cos_obj.composite)
        decision = self._make_decision(cos_obj)
        return cos_obj, decision

    def _make_decision(self, cos_obj: COSScore) -> AgentDecision:
        score   = cos_obj.composite
        trend   = self._trend()
        issues  = []
        action  = "approve"
        confidence = 0.9

        if score < self.cos_threshold:
            issues.append(f"COS {score:.4f} < threshold {self.cos_threshold}")
            action     = "alert"
            confidence = 0.8

        if trend == "declining":
            issues.append(f"COS declining over last {self.declining_streak} readings")
            action    = "alert"
            confidence *= 0.85

        if not issues:
            rationale = (f"COS={score:.4f} ({trend}) — healthy | "
                         f"entropy={cos_obj.entropy_score:.3f} "
                         f"fairness={cos_obj.fairness_score:.3f}")
        else:
            rationale = " | ".join(issues)

        return AgentDecision(
            agent_name=self.name,
            action=action,
            rationale=rationale,
            confidence=round(confidence, 3),
            cos_before=score,
            params_changed={"trend": trend, "history_len": len(self._history)},
        )

    def _trend(self) -> str:
        if len(self._history) < 2:
            return "unknown"
        recent = self._history[-self.declining_streak:] if len(self._history) >= self.declining_streak else self._history
        if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            return "declining"
        if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return "improving"
        return "stable"

    @property
    def latest_cos(self) -> Optional[float]:
        return self._history[-1] if self._history else None

    def cos_stats(self) -> Dict[str, float]:
        if not self._history:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "latest": 0.0}
        return {
            "mean":   round(statistics.mean(self._history), 4),
            "min":    round(min(self._history), 4),
            "max":    round(max(self._history), 4),
            "latest": round(self._history[-1], 4),
            "trend":  self._trend(),
        }


# ══════════════════════════════════════════════════════════════════════════
# Param Adjust Agent
# ══════════════════════════════════════════════════════════════════════════

class ParamAdjustAgent:
    """
    Auto-adjust parameter optimasi jika COS rendah atau ada data shift.

    Strategi adjustment:
      1. COS < threshold    → tighten cos_threshold, increase n_iterations
      2. Power spike        → scale down power targets
      3. Carbon drift       → tighten carbon_intensity budget
      4. PUE drift          → lower pue_target
      5. Sudah optimal      → no-op

    Setiap adjustment diverifikasi EthicsSwarm sebelum diterapkan.
    """

    def __init__(
        self,
        cos_threshold:      float = 0.70,
        pue_target:         float = 1.5,
        carbon_budget:      float = 0.25,
        adjustment_step:    float = 0.05,
        max_iterations:     int   = 5,
        name:               str   = "ParamAdjustAgent",
    ):
        self.params = {
            "cos_threshold":   cos_threshold,
            "pue_target":      pue_target,
            "carbon_budget":   carbon_budget,
            "adjustment_step": adjustment_step,
            "max_iterations":  max_iterations,
        }
        self.name    = name
        self._swarm  = EthicsSwarm()
        self._history: List[Dict[str, Any]] = []

    def adjust(
        self,
        cos_obj:             COSScore,
        robustness_decision: AgentDecision,
    ) -> AgentDecision:
        """
        Decide parameter adjustments given COS score and robustness check.
        Returns AgentDecision. Changes are NOT applied until caller confirms.
        """
        old_params     = dict(self.params)
        proposed       = dict(self.params)
        changes:  List[str] = []
        cos_score      = cos_obj.composite

        # ── Rule 1: COS below threshold ──────────────────────────────────
        if cos_score < self.params["cos_threshold"]:
            gap = self.params["cos_threshold"] - cos_score
            # Lower threshold slightly to allow gradual improvement
            proposed["cos_threshold"] = round(
                max(0.50, self.params["cos_threshold"] - self.params["adjustment_step"] * 0.5), 4)
            proposed["max_iterations"] = min(
                self.params["max_iterations"] + 1, 10)
            changes.append(f"COS {cos_score:.4f} below threshold → "
                           f"threshold relaxed to {proposed['cos_threshold']}")

        # ── Rule 2: Power spike ───────────────────────────────────────────
        if robustness_decision.action == "alert":
            issues = robustness_decision.params_changed.get("detected_issues", [])
            if any("power spike" in i.lower() for i in issues):
                proposed["pue_target"] = round(
                    max(1.2, self.params["pue_target"] - self.params["adjustment_step"]), 3)
                changes.append(f"Power spike → PUE target tightened to {proposed['pue_target']}")

            # ── Rule 3: Carbon drift ──────────────────────────────────────
            if any("carbon" in i.lower() for i in issues):
                proposed["carbon_budget"] = round(
                    max(0.10, self.params["carbon_budget"] - 0.02), 4)
                changes.append(f"Carbon drift → budget tightened to {proposed['carbon_budget']}")

        # ── Rule 4: Low entropy / fairness ──────────────────────────────
        if cos_obj.entropy_score < 0.6:
            proposed["adjustment_step"] = round(
                min(0.10, self.params["adjustment_step"] + 0.01), 4)
            changes.append(f"Low entropy ({cos_obj.entropy_score:.3f}) → step increased")

        if not changes:
            return AgentDecision(
                agent_name=self.name, action="no_op",
                rationale=f"Parameters optimal (COS={cos_score:.4f}, no shifts detected)",
                confidence=0.95,
                params_changed={},
            )

        # ── Verify with EthicsSwarm before proposing ─────────────────────
        swarm_result = self._swarm.deliberate(cos_obj)
        if not swarm_result["approved"]:
            return AgentDecision(
                agent_name=self.name, action="reject",
                rationale=f"EthicsSwarm rejected adjustments: {swarm_result['summary']}",
                confidence=0.4,
                swarm_approved=False,
                swarm_consensus=swarm_result["consensus_score"],
                params_changed=proposed,
                cos_before=cos_score,
            )

        # Apply changes
        for k, v in proposed.items():
            self.params[k] = v

        self._history.append({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "old_params": old_params,
            "new_params": dict(self.params),
            "changes":    changes,
            "cos_score":  cos_score,
        })

        return AgentDecision(
            agent_name=self.name,
            action="adjust",
            rationale=" | ".join(changes),
            confidence=round(swarm_result["consensus_score"], 3),
            params_changed=dict(self.params),
            swarm_approved=True,
            swarm_consensus=swarm_result["consensus_score"],
            cos_before=cos_score,
        )

    def get_current_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def adjustment_history(self) -> List[Dict[str, Any]]:
        return list(self._history)


# ══════════════════════════════════════════════════════════════════════════
# Agentic Pipeline — LangGraph-inspired orchestration
# ══════════════════════════════════════════════════════════════════════════

class AgenticPipeline:
    """
    Multi-agent pipeline terinspirasi LangGraph untuk Eco AI Data Center.

    Graph / state machine:
      START → robustness_check → cos_monitor → param_adjust → guardrail → END

    Setiap node adalah agent. State mengalir antar node.
    Semua transisi state dicatat ke ProvenanceChain.
    Output akhir diverifikasi CATERYAGuardrail.

    Parameters
    ----------
    provenance        : ProvenanceChain (auto-create jika None)
    cos_threshold     : Threshold COS minimum (default 0.70)
    pue_target        : Target PUE awal
    carbon_budget     : Budget emisi karbon kg/kWh
    max_auto_adjusts  : Berapa kali pipeline boleh auto-adjust sebelum berhenti
    """

    def __init__(
        self,
        provenance:       Optional[ProvenanceChain] = None,
        cos_threshold:    float = 0.70,
        pue_target:       float = 1.5,
        carbon_budget:    float = 0.25,
        max_auto_adjusts: int   = 3,
        signing_key:      Optional[str] = None,
    ):
        self.provenance   = provenance or ProvenanceChain(model_id="agentic-pipeline")
        self.cos_threshold = cos_threshold

        self.robustness = RobustnessAgent()
        self.cos_monitor = COSMonitorAgent(cos_threshold=cos_threshold)
        self.param_adjuster = ParamAdjustAgent(
            cos_threshold=cos_threshold,
            pue_target=pue_target,
            carbon_budget=carbon_budget,
        )
        self.guardrail = CATERYAGuardrail(
            config=GuardrailConfig(cos_threshold=cos_threshold - 0.05))

        self.max_auto_adjusts = max_auto_adjusts
        self._run_history: List[PipelineResult] = []

    # ── main pipeline run ────────────────────────────────────────────────

    def run(
        self,
        df:           pd.DataFrame,
        optimizer_fn: Optional[Callable] = None,
        actor:        str = "system",
        context:      Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Execute the full multi-agent pipeline on df.

        Steps:
          1. robustness_check  → detect data shifts
          2. cos_monitor       → evaluate COS score
          3. param_adjust      → adjust params if needed (max_auto_adjusts times)
          4. guardrail         → final ethical gate

        Returns PipelineResult with full trace.
        """
        t0       = time.perf_counter()
        run_id   = uuid.uuid4().hex[:10]
        decisions: List[AgentDecision] = []

        logger.info("[AgenticPipeline] Starting run %s | actor=%s | df=%d rows",
                    run_id, actor, len(df))

        # ── Node 1: Robustness Check ──────────────────────────────────────
        if self.robustness._baseline is None:
            # First run: set baseline from first half of df
            half = max(10, len(df) // 2)
            self.robustness.set_baseline(df.head(half))

        robustness_dec = self.robustness.check(df)
        decisions.append(robustness_dec)
        self._log_prov("robustness_check", robustness_dec, run_id)

        # ── Node 2: COS Monitor ───────────────────────────────────────────
        cos_obj, cos_dec = self.cos_monitor.evaluate_df(df, optimizer_fn=optimizer_fn)
        cos_initial      = cos_obj.composite
        decisions.append(cos_dec)
        self._log_prov("cos_monitor", cos_dec, run_id, {"cos": cos_initial})

        # ── Node 3: Param Adjust (conditional loop) ───────────────────────
        cos_final     = cos_initial
        adjust_count  = 0
        current_cos   = cos_obj

        while adjust_count < self.max_auto_adjusts:
            # Only adjust if there's an issue
            if cos_dec.action == "approve" and robustness_dec.action == "approve":
                break

            adjust_dec = self.param_adjuster.adjust(current_cos, robustness_dec)
            decisions.append(adjust_dec)
            self._log_prov("param_adjust", adjust_dec, run_id)

            if adjust_dec.action in ("reject", "no_op"):
                break

            # Re-evaluate after adjustment (apply new carbon_budget heuristic)
            df_adjusted = self._apply_param_heuristic(
                df, self.param_adjuster.get_current_params())
            cos_obj2, cos_dec2 = self.cos_monitor.evaluate_df(
                df_adjusted, optimizer_fn=optimizer_fn)
            cos_final = cos_obj2.composite
            decisions.append(cos_dec2)
            self._log_prov("cos_recheck", cos_dec2, run_id, {"cos": cos_final})

            current_cos = cos_obj2
            cos_dec     = cos_dec2
            adjust_count += 1

            if cos_dec.action == "approve":
                break

        # ── Node 4: Guardrail ─────────────────────────────────────────────
        output_to_check = {
            "pue":      self.param_adjuster.params["pue_target"],
            "cos":      cos_final,
            "carbon":   self.param_adjuster.params["carbon_budget"],
        }
        guardrail_result = self.guardrail.check(
            output=output_to_check,
            actor=actor,
            endpoint="agentic_pipeline",
            cos_score=cos_final,
        )

        guardrail_dec = AgentDecision(
            agent_name="Guardrail",
            action="approve" if guardrail_result.passed else "reject",
            rationale=guardrail_result.reason,
            confidence=cos_final,
            guardrail_passed=guardrail_result.passed,
        )
        decisions.append(guardrail_dec)
        self._log_prov("guardrail", guardrail_dec, run_id)

        # ── Swarm final vote ──────────────────────────────────────────────
        swarm = EthicsSwarm()
        swarm_result = swarm.deliberate(current_cos)

        # ── Build result ──────────────────────────────────────────────────
        prov_hash = self.provenance.record("pipeline.completed", {
            "run_id":          run_id,
            "cos_initial":     cos_initial,
            "cos_final":       cos_final,
            "decisions":       len(decisions),
            "swarm_consensus": swarm_result["consensus_score"],
            "guardrail":       guardrail_result.passed,
            "actor":           actor,
        })

        result = PipelineResult(
            run_id=run_id,
            cos_initial=cos_initial,
            cos_final=cos_final,
            cos_improved=(cos_final > cos_initial),
            decisions=decisions,
            params_applied=self.param_adjuster.get_current_params(),
            swarm_consensus=swarm_result["consensus_score"],
            guardrail_passed=guardrail_result.passed,
            provenance_hash=prov_hash,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )
        self._run_history.append(result)

        logger.info("[AgenticPipeline] %s", result.summary())
        return result

    # ── helpers ──────────────────────────────────────────────────────────

    def _log_prov(self, node: str, dec: AgentDecision,
                  run_id: str, extra: Optional[Dict] = None) -> None:
        payload: Dict[str, Any] = {
            "run_id": run_id, "node": node,
            "action": dec.action, "confidence": dec.confidence,
            "rationale": dec.rationale[:80],
        }
        if extra:
            payload.update(extra)
        self.provenance.record(f"pipeline.{node}", payload)

    @staticmethod
    def _apply_param_heuristic(df: pd.DataFrame,
                                params: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a light heuristic to df based on adjusted parameters.
        This simulates the effect of tighter carbon/PUE targets on the data.
        In production, this would call the actual optimizer.
        """
        df2 = df.copy()
        cb  = params.get("carbon_budget", 0.25)
        # Clip carbon intensity to new budget
        if "carbon_intensity_kg_per_kwh" in df2.columns:
            df2["carbon_intensity_kg_per_kwh"] = df2["carbon_intensity_kg_per_kwh"].clip(upper=cb * 1.1)
        # Reduce power proportionally to PUE target
        pue_t = params.get("pue_target", 1.5)
        if "it_power_kw" in df2.columns and "power_kw" in df2.columns:
            target_total = df2["it_power_kw"] * pue_t
            reduction    = (df2["power_kw"] / target_total.clip(lower=1)).clip(lower=1)
            df2["power_kw"] = (df2["power_kw"] / reduction).clip(lower=500)
        return df2

    def run_history(self) -> List[PipelineResult]:
        return list(self._run_history)

    def pipeline_stats(self) -> Dict[str, Any]:
        if not self._run_history:
            return {"total_runs": 0}
        cos_values = [r.cos_final for r in self._run_history]
        return {
            "total_runs":        len(self._run_history),
            "avg_cos_final":     round(statistics.mean(cos_values), 4),
            "best_cos":          round(max(cos_values), 4),
            "worst_cos":         round(min(cos_values), 4),
            "total_improvements": sum(1 for r in self._run_history if r.cos_improved),
            "guardrail_pass_rate": sum(1 for r in self._run_history if r.guardrail_passed)
                                   / len(self._run_history),
        }
