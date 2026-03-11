"""
workers/tasks.py
=================
Celery background task definitions for Eco AI Data Center.

Tasks:
  - run_monitor_poll        : single sensor poll cycle
  - run_cos_evaluation      : full CATERYA COS evaluation
  - run_quantum_thermal     : quantum thermal simulation
  - run_predictive_check    : LSTM maintenance check across all metrics
  - send_alert_notification : async notification dispatch

Graceful degradation: if Celery/Redis are not installed, tasks run
synchronously inline via the FallbackTaskRunner.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("eco_ai.workers")

# ---------------------------------------------------------------------------
# Optional Celery — graceful fallback
# ---------------------------------------------------------------------------

CELERY_AVAILABLE = False
celery_app = None

try:
    from celery import Celery  # type: ignore

    _broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    _backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    celery_app = Celery(
        "eco_ai_datacenter",
        broker=_broker,
        backend=_backend,
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_soft_time_limit=60,
        task_time_limit=120,
        worker_prefetch_multiplier=1,
        # Beat schedule for periodic tasks
        beat_schedule={
            "monitor-poll-every-10s": {
                "task": "workers.tasks.run_monitor_poll",
                "schedule": 10.0,
                "kwargs": {"backend": "mock"},
            },
            "cos-eval-every-5min": {
                "task": "workers.tasks.run_cos_evaluation",
                "schedule": 300.0,
            },
            "quantum-thermal-every-30min": {
                "task": "workers.tasks.run_quantum_thermal",
                "schedule": 1800.0,
            },
        },
    )
    CELERY_AVAILABLE = True
    logger.info("[Workers] Celery initialised | broker=%s", _broker)

except ImportError:
    logger.info(
        "[Workers] Celery not installed — tasks run synchronously. "
        "Install with: pip install celery redis"
    )


# ---------------------------------------------------------------------------
# Synchronous fallback task runner
# ---------------------------------------------------------------------------

class FallbackTaskResult:
    """Mimics Celery AsyncResult interface for sync fallback."""

    def __init__(self, result: Any):
        self._result = result
        self.id = "sync-task"
        self.status = "SUCCESS"

    def get(self, timeout: Optional[int] = None) -> Any:
        return self._result

    @property
    def ready(self) -> bool:
        return True


class FallbackTaskRunner:
    """
    Runs tasks synchronously when Celery is not available.
    Drop-in replacement — same interface as Celery task .delay() / .apply_async().
    """

    def __init__(self, fn):
        self.fn = fn

    def delay(self, *args, **kwargs) -> FallbackTaskResult:
        result = self.fn(*args, **kwargs)
        return FallbackTaskResult(result)

    def apply_async(self, args=None, kwargs=None, **options) -> FallbackTaskResult:
        result = self.fn(*(args or []), **(kwargs or {}))
        return FallbackTaskResult(result)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Task definitions (work with or without Celery)
# ---------------------------------------------------------------------------

def _define_task(fn):
    """Wrap function as Celery task if available, else as FallbackTaskRunner."""
    if CELERY_AVAILABLE and celery_app is not None:
        return celery_app.task(name=f"workers.tasks.{fn.__name__}")(fn)
    return FallbackTaskRunner(fn)


# --- Task 1: Monitor poll ---

def _run_monitor_poll(backend: str = "mock", n_polls: int = 1) -> Dict:
    """
    Run one (or more) monitor poll cycles and return the latest snapshot.
    """
    from real_time_monitor import RealTimeMonitor
    monitor = RealTimeMonitor(backend=backend)
    results = []
    for _ in range(n_polls):
        snapshot = monitor.poll_once()
        results.append(snapshot)
    return {
        "status": "ok",
        "polls": n_polls,
        "last_snapshot": results[-1] if results else {},
    }


run_monitor_poll = _define_task(_run_monitor_poll)


# --- Task 2: COS evaluation ---

def _run_cos_evaluation(n_rows: int = 200) -> Dict:
    """
    Generate dummy (or load real) data and compute full CATERYA COS.
    """
    import numpy as np
    import pandas as pd
    from caterya_integration import EcoAIDataCenterCATERYA
    from optimizer import full_optimization

    rng = np.random.default_rng()
    df = pd.DataFrame({
        "total_power_kw":              rng.normal(1200, 80, n_rows).clip(900, 1600),
        "it_power_kw":                 rng.normal(700,  50, n_rows).clip(500, 950),
        "water_usage_litres":          rng.normal(500,  40, n_rows).clip(300, 800),
        "it_energy_kwh":               rng.normal(350,  25, n_rows).clip(200, 500),
        "energy_kwh":                  rng.normal(1100, 60, n_rows).clip(800, 1500),
        "carbon_intensity_kg_per_kwh": rng.normal(0.25, 0.05, n_rows).clip(0.05, 0.6),
    })

    eco = EcoAIDataCenterCATERYA(cos_threshold=0.7, model_id="celery-cos-eval")
    pipeline = eco.run_with_evaluation(full_optimization, df)

    return {
        "status": "ok",
        "cos_composite": pipeline["cos"].composite,
        "cos_passed": pipeline["cos"].passed,
        "approved": pipeline["approved"],
        "swarm_consensus": pipeline["swarm"]["consensus_score"],
        "audit_hash": pipeline["audit_hash"],
    }


run_cos_evaluation = _define_task(_run_cos_evaluation)


# --- Task 3: Quantum thermal simulation ---

def _run_quantum_thermal(n_zones: int = 8) -> Dict:
    """
    Run quantum thermal distribution simulation for N zones.
    """
    import numpy as np
    from thermal_circuit import QuantumThermalSimulator

    rng = np.random.default_rng()
    temperatures = rng.normal(22, 4, n_zones).clip(15, 45).tolist()
    zone_labels = [f"Rack-{chr(65 + i // 4)}{(i % 4) + 1}" for i in range(n_zones)]

    sim = QuantumThermalSimulator(n_qubits=4)
    result = sim.simulate(temperatures, zone_labels=zone_labels)

    return {
        "status": "ok",
        "backend": result.backend,
        "hotspot_zones": result.hotspot_zones,
        "fairness_score": result.fairness_score,
        "fairness_passed": result.fairness_passed,
        "zone_scores": dict(zip(result.zone_labels, result.thermal_scores)),
        "summary": result.summary(),
    }


run_quantum_thermal = _define_task(_run_quantum_thermal)


# --- Task 4: Predictive maintenance check ---

def _run_predictive_check(n_history: int = 60) -> Dict:
    """
    Simulate historical data feed and run LSTM predictive maintenance.
    """
    import numpy as np
    from real_time_monitor import PredictiveMaintenance, MockSensorBackend

    sensor = MockSensorBackend()
    predictor = PredictiveMaintenance(seq_len=30)

    # Feed historical data
    for _ in range(n_history):
        readings = sensor.read_all()
        for r in readings:
            predictor.update(r.metric_name, r.value)

    # Collect predictions
    predictions = {}
    target_metrics = [
        "total_power_kw", "temperature_celsius",
        "server_utilization_pct", "ups_efficiency_pct",
    ]
    for metric in target_metrics:
        pred = predictor.predict(metric)
        if pred:
            predictions[metric] = {
                "predicted_next": pred.predicted_next,
                "failure_probability": pred.failure_probability,
                "maintenance_recommended": pred.maintenance_recommended,
                "model_type": pred.model_type,
            }

    maintenance_alerts = [
        m for m, p in predictions.items()
        if p.get("maintenance_recommended")
    ]

    return {
        "status": "ok",
        "predictions": predictions,
        "maintenance_alerts": maintenance_alerts,
        "n_history_points": n_history,
    }


run_predictive_check = _define_task(_run_predictive_check)


# --- Task 5: Alert notification ---

def _send_alert_notification(
    metric_name: str,
    current_value: float,
    threshold: float,
    severity: str = "warning",
    channels: Optional[list] = None,
) -> Dict:
    """
    Dispatch an alert notification asynchronously.
    """
    from real_time_monitor import AlertManager, AlertEvent

    manager = AlertManager(
        slack_webhook_url=os.getenv("ALERT_SLACK_WEBHOOK"),
        alert_email_to=os.getenv("ALERT_EMAIL_TO"),
    )
    event = AlertEvent(
        metric_name=metric_name,
        current_value=current_value,
        threshold=threshold,
        direction="above",
        severity=severity,
        message=(
            f"[{severity.upper()}] {metric_name} = {current_value:.3f} "
            f"(threshold: {threshold:.3f})"
        ),
    )
    manager._dispatch(event)
    return {"status": "dispatched", "metric": metric_name, "severity": severity}


send_alert_notification = _define_task(_send_alert_notification)


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------

def get_task_status() -> Dict:
    """Return status of background task infrastructure."""
    return {
        "celery_available": CELERY_AVAILABLE,
        "mode": "celery" if CELERY_AVAILABLE else "synchronous-fallback",
        "broker": os.getenv("CELERY_BROKER_URL", "not configured") if CELERY_AVAILABLE else "n/a",
        "tasks": [
            "run_monitor_poll",
            "run_cos_evaluation",
            "run_quantum_thermal",
            "run_predictive_check",
            "send_alert_notification",
        ],
    }
