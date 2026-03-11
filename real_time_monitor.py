"""
real_time_monitor.py
====================
Real-time monitoring engine for Eco AI Data Center.

Integrates with:
  - Prometheus  (pull metrics via HTTP scrape)
  - InfluxDB    (time-series read/write)
  - Mock sensor (built-in dummy for development / offline mode)

Also provides:
  - LSTM predictive maintenance (PyTorch or sklearn fallback)
  - QuantumSymmetryIndex for asymmetry detection in energy streams
  - Event-driven alert system (email, Slack)
  - Audit logging for all monitor events via ProvenanceChain

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import logging
import math
import os
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from provenance import ProvenanceChain
from fairness import QuantumFairnessEvaluator

logger = logging.getLogger("eco_ai.monitor")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    logger.info("[Monitor] PyTorch not found — using sklearn LSTM fallback.")

try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.info("[Monitor] requests not installed — Prometheus/InfluxDB disabled.")

try:
    from influxdb_client import InfluxDBClient, Point  # type: ignore
    from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False

try:
    import smtplib
    SMTP_AVAILABLE = True
except ImportError:
    SMTP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """A single timestamped reading from one sensor/metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "mock"
    tags: dict = field(default_factory=dict)


@dataclass
class AlertEvent:
    """Fired when a metric crosses a threshold."""
    metric_name: str
    current_value: float
    threshold: float
    direction: str           # "above" | "below"
    severity: str            # "warning" | "critical"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged: bool = False


@dataclass
class PredictionResult:
    """Output from the LSTM predictive maintenance model."""
    metric_name: str
    predicted_next: float
    failure_probability: float     # 0-1
    maintenance_recommended: bool
    horizon_minutes: int
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_type: str = "unknown"


@dataclass
class QuantumSymmetryIndex:
    """
    Quantum-inspired symmetry index for asymmetry detection in energy streams.
    Combines classical skewness with a quantum-superposition-inspired divergence.
    """
    metric_name: str
    qsi_score: float            # 0 = perfectly symmetric, 1 = maximally asymmetric
    classical_skew: float
    quantum_divergence: float
    superposition_entropy: float
    asymmetry_detected: bool    # True if qsi_score > threshold
    threshold: float = 0.35
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        status = "⚠️ ASYMMETRIC" if self.asymmetry_detected else "✅ SYMMETRIC"
        return (
            f"{self.metric_name} [{status}] | QSI={self.qsi_score:.4f} "
            f"(skew={self.classical_skew:.3f}, "
            f"q-div={self.quantum_divergence:.3f}, "
            f"s-entropy={self.superposition_entropy:.3f})"
        )


# ---------------------------------------------------------------------------
# Sensor backends
# ---------------------------------------------------------------------------

class MockSensorBackend:
    """
    Built-in random walk sensor — no external dependencies.
    Simulates realistic data center telemetry with diurnal patterns.
    """

    METRICS = {
        "temperature_celsius":         (22.0, 1.5, 16.0, 40.0),
        "humidity_pct":                (45.0, 3.0, 20.0, 80.0),
        "total_power_kw":              (1200.0, 60.0, 800.0, 1800.0),
        "it_power_kw":                 (700.0, 40.0, 400.0, 1100.0),
        "cooling_power_kw":            (310.0, 25.0, 150.0, 550.0),
        "water_usage_litres":          (500.0, 30.0, 250.0, 900.0),
        "it_energy_kwh":               (350.0, 20.0, 180.0, 550.0),
        "carbon_intensity_kg_per_kwh": (0.25, 0.03, 0.05, 0.65),
        "server_utilization_pct":      (72.0, 8.0, 5.0, 100.0),
        "network_throughput_gbps":     (12.0, 2.0, 1.0, 40.0),
        "ups_efficiency_pct":          (96.0, 1.0, 85.0, 99.5),
        "pdu_load_pct":                (68.0, 5.0, 20.0, 95.0),
    }

    UNITS = {
        "temperature_celsius":         "°C",
        "humidity_pct":                "%",
        "total_power_kw":              "kW",
        "it_power_kw":                 "kW",
        "cooling_power_kw":            "kW",
        "water_usage_litres":          "L",
        "it_energy_kwh":               "kWh",
        "carbon_intensity_kg_per_kwh": "kg/kWh",
        "server_utilization_pct":      "%",
        "network_throughput_gbps":     "Gbps",
        "ups_efficiency_pct":          "%",
        "pdu_load_pct":                "%",
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._state = {k: v[0] for k, v in self.METRICS.items()}
        self._tick = 0

    def read_all(self) -> List[SensorReading]:
        self._tick += 1
        # Diurnal pattern: hour-of-day sine wave influence
        hour_phase = math.sin(2 * math.pi * (self._tick % 1440) / 1440)
        readings = []

        for name, (base, noise, lo, hi) in self.METRICS.items():
            # Random walk with mean reversion
            drift = (base - self._state[name]) * 0.05
            step = drift + self.rng.normal(0, noise * 0.15)
            # Diurnal uplift for power/cooling metrics
            if "power" in name or "utilization" in name:
                step += base * 0.05 * hour_phase
            new_val = float(np.clip(self._state[name] + step, lo, hi))
            self._state[name] = new_val

            readings.append(SensorReading(
                metric_name=name,
                value=round(new_val, 4),
                unit=self.UNITS.get(name, ""),
                source="mock",
            ))

        return readings


class PrometheusBackend:
    """
    Pull metrics from a Prometheus HTTP endpoint.
    Falls back gracefully to empty list if unavailable.
    """

    def __init__(self, url: str, timeout: int = 5):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._metric_names = [
            "datacenter_power_total_kw",
            "datacenter_it_power_kw",
            "datacenter_temperature_celsius",
            "datacenter_humidity_pct",
            "datacenter_pue",
        ]

    def read_all(self) -> List[SensorReading]:
        if not REQUESTS_AVAILABLE:
            return []
        readings = []
        for metric in self._metric_names:
            try:
                resp = requests.get(
                    f"{self.url}/api/v1/query",
                    params={"query": metric},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    val = float(results[0]["value"][1])
                    readings.append(SensorReading(
                        metric_name=metric,
                        value=val,
                        unit="",
                        source="prometheus",
                        tags=results[0].get("metric", {}),
                    ))
            except Exception as exc:
                logger.debug("[Prometheus] %s: %s", metric, exc)
        return readings


class InfluxDBBackend:
    """
    Read latest metrics from InfluxDB v2.
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        measurement: str = "datacenter_metrics",
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement
        self._client = None

    def _get_client(self):
        if not INFLUX_AVAILABLE:
            return None
        if self._client is None:
            self._client = InfluxDBClient(
                url=self.url, token=self.token, org=self.org
            )
        return self._client

    def read_all(self) -> List[SensorReading]:
        client = self._get_client()
        if client is None:
            return []
        try:
            query_api = client.query_api()
            flux = (
                f'from(bucket: "{self.bucket}") '
                f"|> range(start: -1m) "
                f'|> filter(fn: (r) => r._measurement == "{self.measurement}") '
                f"|> last()"
            )
            tables = query_api.query(flux, org=self.org)
            readings = []
            for table in tables:
                for record in table.records:
                    readings.append(SensorReading(
                        metric_name=record.get_field(),
                        value=float(record.get_value()),
                        unit="",
                        source="influxdb",
                        timestamp=record.get_time().isoformat(),
                    ))
            return readings
        except Exception as exc:
            logger.warning("[InfluxDB] Read failed: %s", exc)
            return []

    def write(self, readings: List[SensorReading]) -> bool:
        """Write readings back to InfluxDB for persistence."""
        client = self._get_client()
        if client is None:
            return False
        try:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            points = []
            for r in readings:
                p = (
                    Point(self.measurement)
                    .tag("source", r.source)
                    .field(r.metric_name, r.value)
                    .time(r.timestamp)
                )
                points.append(p)
            write_api.write(bucket=self.bucket, org=self.org, record=points)
            return True
        except Exception as exc:
            logger.warning("[InfluxDB] Write failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Simple LSTM for univariate time-series prediction (PyTorch).
    Used for predictive maintenance: predict next value + failure probability.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class PredictiveMaintenance:
    """
    LSTM-based predictive maintenance.

    Uses PyTorch if available, otherwise falls back to a lightweight
    exponential smoothing + trend extrapolation approach (no ML deps required).

    Parameters
    ----------
    seq_len           : int   — lookback window (default 30)
    failure_threshold : float — anomaly z-score threshold for failure flag
    horizon_minutes   : int   — prediction horizon
    """

    def __init__(
        self,
        seq_len: int = 30,
        failure_threshold: float = 2.5,
        horizon_minutes: int = 15,
    ):
        self.seq_len = seq_len
        self.failure_threshold = failure_threshold
        self.horizon_minutes = horizon_minutes
        self._models: Dict[str, any] = {}
        self._scalers: Dict[str, any] = {}
        self._history: Dict[str, Deque[float]] = {}
        logger.info(
            "[PredMaint] Initialised | backend=%s | seq_len=%d | threshold=%.1f",
            "pytorch" if TORCH_AVAILABLE else "fallback",
            seq_len, failure_threshold,
        )

    def update(self, metric_name: str, value: float) -> None:
        """Push a new observation into the rolling buffer."""
        if metric_name not in self._history:
            self._history[metric_name] = deque(maxlen=self.seq_len * 3)
        self._history[metric_name].append(value)

    def predict(self, metric_name: str) -> Optional[PredictionResult]:
        """Run prediction for a single metric. Returns None if insufficient data."""
        history = self._history.get(metric_name)
        if history is None or len(history) < max(10, self.seq_len):
            return None

        values = list(history)

        if TORCH_AVAILABLE and SKLEARN_AVAILABLE:
            return self._predict_torch(metric_name, values)
        else:
            return self._predict_fallback(metric_name, values)

    def _predict_torch(self, metric_name: str, values: List[float]) -> PredictionResult:
        """PyTorch LSTM prediction."""
        scaler = self._scalers.get(metric_name)
        if scaler is None:
            scaler = MinMaxScaler()
            self._scalers[metric_name] = scaler

        arr = np.array(values).reshape(-1, 1)
        arr_scaled = scaler.fit_transform(arr)

        # Build sequences
        seq = arr_scaled[-self.seq_len:]
        x = torch.FloatTensor(seq).unsqueeze(0)   # (1, seq_len, 1)

        model = self._models.get(metric_name)
        if model is None:
            model = LSTMModel(input_size=1, hidden_size=32, num_layers=2)
            # Quick online training on available history
            self._quick_train(model, arr_scaled)
            self._models[metric_name] = model

        model.eval()
        with torch.no_grad():
            pred_scaled = model(x).item()

        pred_value = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        failure_prob, confidence = self._anomaly_score(values, pred_value)

        return PredictionResult(
            metric_name=metric_name,
            predicted_next=round(pred_value, 4),
            failure_probability=failure_prob,
            maintenance_recommended=failure_prob > 0.7,
            horizon_minutes=self.horizon_minutes,
            confidence=confidence,
            model_type="pytorch-lstm",
        )

    def _quick_train(self, model: LSTMModel, arr_scaled: np.ndarray, epochs: int = 15):
        """Quick online training with gradient descent."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for i in range(self.seq_len, len(arr_scaled)):
                seq = arr_scaled[i - self.seq_len:i]
                x = torch.FloatTensor(seq).unsqueeze(0)
                y = torch.FloatTensor([[arr_scaled[i][0]]])
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _predict_fallback(self, metric_name: str, values: List[float]) -> PredictionResult:
        """
        Lightweight fallback: double exponential smoothing (Holt's method).
        No ML dependencies needed.
        """
        alpha, beta = 0.3, 0.1
        level = values[0]
        trend = values[1] - values[0]

        for v in values[1:]:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        pred_value = level + trend

        # Clamp to reasonable bounds based on historical min/max
        lo, hi = min(values), max(values)
        pred_value = float(np.clip(pred_value, lo * 0.8, hi * 1.2))
        failure_prob, confidence = self._anomaly_score(values, pred_value)

        return PredictionResult(
            metric_name=metric_name,
            predicted_next=round(pred_value, 4),
            failure_probability=failure_prob,
            maintenance_recommended=failure_prob > 0.7,
            horizon_minutes=self.horizon_minutes,
            confidence=confidence,
            model_type="holt-smoothing",
        )

    def _anomaly_score(
        self, values: List[float], predicted: float
    ) -> Tuple[float, float]:
        """
        Compute failure probability based on z-score of the predicted value
        relative to recent distribution.
        """
        if len(values) < 4:
            return 0.0, 0.5

        mean = statistics.mean(values[-20:])
        std = statistics.stdev(values[-20:]) if len(values) >= 4 else 1.0
        if std == 0:
            return 0.0, 0.9

        z = abs(predicted - mean) / std
        # Sigmoid transform of z-score → probability
        failure_prob = float(1 / (1 + math.exp(-0.8 * (z - self.failure_threshold))))
        confidence = float(np.clip(1.0 - (std / (abs(mean) + 1e-9)), 0.1, 0.99))
        return round(failure_prob, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Quantum Symmetry Index
# ---------------------------------------------------------------------------

class QuantumSymmetryAnalyser:
    """
    Quantum-inspired asymmetry detection for streaming energy data.

    Computes a Quantum Symmetry Index (QSI) that combines:
      - Classical skewness
      - Quantum probability divergence (von Neumann entropy-inspired)
      - Superposition state entropy across data splits

    QSI = 0 → perfectly symmetric
    QSI = 1 → maximally asymmetric
    """

    def __init__(self, asymmetry_threshold: float = 0.35, n_bins: int = 16):
        self.asymmetry_threshold = asymmetry_threshold
        self.n_bins = n_bins
        self.qfe = QuantumFairnessEvaluator(n_samples=200, seed=42)

    def analyse(self, metric_name: str, values: List[float]) -> QuantumSymmetryIndex:
        """Compute QSI for a list of values."""
        if len(values) < 4:
            return QuantumSymmetryIndex(
                metric_name=metric_name,
                qsi_score=0.0, classical_skew=0.0,
                quantum_divergence=0.0, superposition_entropy=0.0,
                asymmetry_detected=False,
                threshold=self.asymmetry_threshold,
            )

        arr = np.array(values, dtype=float)

        # 1. Classical skewness (normalised to 0-1)
        classical_skew = self._compute_skew(arr)
        norm_skew = min(abs(classical_skew) / 3.0, 1.0)

        # 2. Quantum probability divergence
        q_div = self._quantum_divergence(arr)

        # 3. Superposition entropy across random splits
        sup_entropy = self._superposition_entropy(arr)

        # 4. Composite QSI
        qsi_score = round(
            0.40 * norm_skew + 0.35 * q_div + 0.25 * sup_entropy, 4
        )

        return QuantumSymmetryIndex(
            metric_name=metric_name,
            qsi_score=qsi_score,
            classical_skew=round(classical_skew, 4),
            quantum_divergence=round(q_div, 4),
            superposition_entropy=round(sup_entropy, 4),
            asymmetry_detected=qsi_score > self.asymmetry_threshold,
            threshold=self.asymmetry_threshold,
        )

    def _compute_skew(self, arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float(((arr - mean) ** 3).mean() / (std ** 3))

    def _quantum_divergence(self, arr: np.ndarray) -> float:
        """
        Split array into two halves, compute KL divergence of their
        histograms — mirrors quantum state overlap measurement.
        """
        half = len(arr) // 2
        p_hist, bins = np.histogram(arr[:half], bins=self.n_bins, density=True)
        q_hist, _    = np.histogram(arr[half:], bins=bins, density=True)

        # Add smoothing to avoid log(0)
        eps = 1e-10
        p = p_hist + eps
        q = q_hist + eps
        p /= p.sum()
        q /= q.sum()

        kl = float(np.sum(p * np.log(p / q)))
        # Normalise KL divergence to [0, 1] via sigmoid-like transform
        return float(1 - math.exp(-kl))

    def _superposition_entropy(self, arr: np.ndarray, n_splits: int = 8) -> float:
        """
        Simulate quantum superposition: compute entropy across N random splits.
        High variance in split entropies → high superposition entropy → asymmetric.
        """
        rng = np.random.default_rng(99)
        entropies = []
        for _ in range(n_splits):
            idx = rng.choice(len(arr), size=len(arr) // 2, replace=False)
            subset = arr[idx]
            counts, _ = np.histogram(subset, bins=self.n_bins)
            probs = counts / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            h = -float(np.sum(probs * np.log2(probs + 1e-10)))
            max_h = math.log2(self.n_bins)
            entropies.append(h / max_h if max_h > 0 else 0)

        if not entropies:
            return 0.0
        # Variance in entropies across splits = asymmetry signal
        variance = statistics.variance(entropies) if len(entropies) > 1 else 0.0
        return float(np.clip(variance * 10, 0, 1))


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------

ALERT_THRESHOLDS = {
    "total_power_kw":     {"warning": 1400.0, "critical": 1600.0, "direction": "above"},
    "temperature_celsius": {"warning": 27.0,  "critical": 32.0,   "direction": "above"},
    "humidity_pct":        {"warning": 65.0,  "critical": 75.0,   "direction": "above"},
    "server_utilization_pct": {"warning": 85.0, "critical": 95.0, "direction": "above"},
    "pdu_load_pct":        {"warning": 80.0,  "critical": 90.0,   "direction": "above"},
    "ups_efficiency_pct":  {"warning": 92.0,  "critical": 88.0,   "direction": "below"},
}


class AlertManager:
    """
    Threshold-based alert manager with email and Slack notification support.
    Uses a deduplication window to avoid alert storms.
    """

    def __init__(
        self,
        dedup_seconds: int = 300,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        alert_email_to: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
    ):
        self.dedup_seconds = dedup_seconds
        self.smtp_host = smtp_host or os.getenv("ALERT_SMTP_HOST")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.getenv("ALERT_SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("ALERT_SMTP_PASSWORD")
        self.alert_email_to = alert_email_to or os.getenv("ALERT_EMAIL_TO")
        self.slack_webhook_url = slack_webhook_url or os.getenv("ALERT_SLACK_WEBHOOK")
        self._last_fired: Dict[str, float] = {}
        self._alert_history: List[AlertEvent] = []
        self._callbacks: List[Callable[[AlertEvent], None]] = []

    def register_callback(self, fn: Callable[[AlertEvent], None]) -> None:
        """Register a function to be called when any alert fires."""
        self._callbacks.append(fn)

    def check(self, reading: SensorReading) -> Optional[AlertEvent]:
        """Check a reading against thresholds. Returns AlertEvent if triggered."""
        rules = ALERT_THRESHOLDS.get(reading.metric_name)
        if rules is None:
            return None

        direction = rules["direction"]
        severity = None

        if direction == "above":
            if reading.value >= rules.get("critical", float("inf")):
                severity = "critical"
            elif reading.value >= rules.get("warning", float("inf")):
                severity = "warning"
        elif direction == "below":
            if reading.value <= rules.get("critical", float("-inf")):
                severity = "critical"
            elif reading.value <= rules.get("warning", float("-inf")):
                severity = "warning"

        if severity is None:
            return None

        # Deduplication check
        key = f"{reading.metric_name}:{severity}"
        now = time.monotonic()
        if now - self._last_fired.get(key, 0) < self.dedup_seconds:
            return None
        self._last_fired[key] = now

        threshold = rules.get(severity, 0.0)
        event = AlertEvent(
            metric_name=reading.metric_name,
            current_value=reading.value,
            threshold=threshold,
            direction=direction,
            severity=severity,
            message=(
                f"[{severity.upper()}] {reading.metric_name} "
                f"is {reading.value:.2f} {reading.unit} "
                f"({'above' if direction == 'above' else 'below'} "
                f"threshold {threshold:.2f})"
            ),
        )

        self._alert_history.append(event)
        logger.warning("[Alert] %s", event.message)
        self._dispatch(event)
        return event

    def _dispatch(self, event: AlertEvent) -> None:
        """Fire all registered callbacks and notification channels."""
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.error("[Alert] Callback error: %s", exc)

        if self.slack_webhook_url:
            self._send_slack(event)
        if self.smtp_host and self.alert_email_to:
            self._send_email(event)

    def _send_slack(self, event: AlertEvent) -> None:
        if not REQUESTS_AVAILABLE:
            return
        icon = "🔴" if event.severity == "critical" else "⚠️"
        payload = {
            "text": f"{icon} *Eco AI Data Center Alert*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{icon} *{event.severity.upper()} Alert*\n"
                            f"*Metric:* `{event.metric_name}`\n"
                            f"*Value:* {event.current_value:.3f}\n"
                            f"*Threshold:* {event.threshold:.3f}\n"
                            f"*Time:* {event.timestamp}\n"
                            f"_{event.message}_"
                        ),
                    },
                }
            ],
        }
        try:
            requests.post(self.slack_webhook_url, json=payload, timeout=5)
            logger.info("[Alert] Slack notification sent for %s", event.metric_name)
        except Exception as exc:
            logger.warning("[Alert] Slack send failed: %s", exc)

    def _send_email(self, event: AlertEvent) -> None:
        if not (SMTP_AVAILABLE and self.smtp_user and self.smtp_password):
            return
        subject = f"[Eco AI DC Alert] {event.severity.upper()}: {event.metric_name}"
        body = (
            f"Eco AI Data Center Alert\n"
            f"========================\n\n"
            f"Severity : {event.severity.upper()}\n"
            f"Metric   : {event.metric_name}\n"
            f"Value    : {event.current_value:.4f}\n"
            f"Threshold: {event.threshold:.4f}\n"
            f"Direction: {event.direction}\n"
            f"Time     : {event.timestamp}\n\n"
            f"Message  : {event.message}\n\n"
            f"-- Eco AI Data Center | CateryaTech --\n"
            f"   cateryatech@proton.me\n"
        )
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.alert_email_to

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_password)
                s.sendmail(self.smtp_user, [self.alert_email_to], msg.as_string())
            logger.info("[Alert] Email sent to %s", self.alert_email_to)
        except Exception as exc:
            logger.warning("[Alert] Email send failed: %s", exc)

    @property
    def recent_alerts(self) -> List[AlertEvent]:
        return self._alert_history[-50:]  # last 50


# ---------------------------------------------------------------------------
# Real-Time Monitor — main orchestrator
# ---------------------------------------------------------------------------

class RealTimeMonitor:
    """
    Main real-time monitoring orchestrator for Eco AI Data Center.

    Runs a background polling thread that:
      1. Reads sensor data (mock / Prometheus / InfluxDB)
      2. Checks alert thresholds
      3. Updates LSTM predictive maintenance buffers
      4. Computes QuantumSymmetryIndex for key energy metrics
      5. Records all events to ProvenanceChain

    Thread-safe: safe to call read_snapshot() from Streamlit at any time.

    Parameters
    ----------
    backend         : "mock" | "prometheus" | "influxdb"
    poll_interval   : seconds between polls (default 5)
    history_len     : how many snapshots to keep in memory (default 500)
    provenance      : ProvenanceChain instance for audit logging
    """

    QUANTUM_METRICS = [
        "total_power_kw", "it_power_kw", "cooling_power_kw",
        "carbon_intensity_kg_per_kwh", "server_utilization_pct",
    ]

    def __init__(
        self,
        backend: str = "mock",
        poll_interval: float = 5.0,
        history_len: int = 500,
        provenance: Optional[ProvenanceChain] = None,
        prometheus_url: str = "http://localhost:9090",
        influx_url: str = "http://localhost:8086",
        influx_token: str = "",
        influx_org: str = "cateryatech",
        influx_bucket: str = "datacenter",
        alert_slack_webhook: Optional[str] = None,
        alert_email_to: Optional[str] = None,
    ):
        self.backend_name = backend
        self.poll_interval = poll_interval
        self.history_len = history_len
        self.provenance = provenance or ProvenanceChain(model_id="realtime-monitor")

        # Sensor backend
        if backend == "prometheus":
            self.sensor = PrometheusBackend(url=prometheus_url)
        elif backend == "influxdb":
            self.sensor = InfluxDBBackend(
                url=influx_url, token=influx_token,
                org=influx_org, bucket=influx_bucket,
            )
        else:
            self.sensor = MockSensorBackend()

        # Sub-systems
        self.predictor = PredictiveMaintenance(seq_len=30)
        self.qsa = QuantumSymmetryAnalyser(asymmetry_threshold=0.35)
        self.alert_manager = AlertManager(
            slack_webhook_url=alert_slack_webhook,
            alert_email_to=alert_email_to,
        )

        # Thread-safe data store
        self._lock = threading.Lock()
        self._history: Deque[Dict] = deque(maxlen=history_len)
        self._latest_snapshot: Dict = {}
        self._latest_predictions: Dict[str, PredictionResult] = {}
        self._latest_qsi: Dict[str, QuantumSymmetryIndex] = {}
        self._metric_buffers: Dict[str, Deque[float]] = {}

        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_queue: queue.Queue = queue.Queue(maxsize=100)

        logger.info(
            "[Monitor] Initialised | backend=%s | interval=%.1fs",
            backend, poll_interval,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="eco-monitor-thread",
            daemon=True,
        )
        self._thread.start()
        logger.info("[Monitor] Background polling started.")

    def stop(self) -> None:
        """Stop background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("[Monitor] Stopped.")

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def read_snapshot(self) -> Dict:
        """Return the latest snapshot (non-blocking)."""
        with self._lock:
            return dict(self._latest_snapshot)

    def get_history_df(self) -> pd.DataFrame:
        """Return the rolling history as a DataFrame."""
        with self._lock:
            rows = list(self._history)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_predictions(self) -> Dict[str, PredictionResult]:
        with self._lock:
            return dict(self._latest_predictions)

    def get_qsi(self) -> Dict[str, QuantumSymmetryIndex]:
        with self._lock:
            return dict(self._latest_qsi)

    def get_alerts(self) -> List[AlertEvent]:
        return self.alert_manager.recent_alerts

    def is_running(self) -> bool:
        return self._running

    def poll_once(self) -> Dict:
        """
        Run a single poll cycle manually (useful for Streamlit without threads).
        Returns the snapshot dict.
        """
        return self._run_poll_cycle()

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._run_poll_cycle()
            except Exception as exc:
                logger.error("[Monitor] Poll error: %s", exc)
            time.sleep(self.poll_interval)

    def _run_poll_cycle(self) -> Dict:
        """Core poll logic: read → alert → predict → QSI → record."""
        readings = self.sensor.read_all()
        if not readings:
            return {}

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "readings": {},
        }

        fired_alerts = []

        for r in readings:
            snapshot["readings"][r.metric_name] = {
                "value": r.value,
                "unit": r.unit,
                "source": r.source,
            }

            # Buffer for QSI and predictions
            if r.metric_name not in self._metric_buffers:
                self._metric_buffers[r.metric_name] = deque(maxlen=200)
            self._metric_buffers[r.metric_name].append(r.value)

            # Predictive maintenance update
            self.predictor.update(r.metric_name, r.value)

            # Alert check
            alert = self.alert_manager.check(r)
            if alert:
                fired_alerts.append(alert.message)
                try:
                    self._alert_queue.put_nowait(alert)
                except queue.Full:
                    pass

        # Predictions (every 5th cycle to save CPU)
        if len(self._history) % 5 == 0:
            preds = {}
            for metric_name in self.QUANTUM_METRICS:
                pred = self.predictor.predict(metric_name)
                if pred:
                    preds[metric_name] = pred
            with self._lock:
                self._latest_predictions.update(preds)

        # Quantum Symmetry Index (every 10th cycle)
        if len(self._history) % 10 == 0:
            qsi_results = {}
            for metric_name in self.QUANTUM_METRICS:
                buf = self._metric_buffers.get(metric_name)
                if buf and len(buf) >= 20:
                    qsi = self.qsa.analyse(metric_name, list(buf))
                    qsi_results[metric_name] = qsi
                    if qsi.asymmetry_detected:
                        logger.warning(
                            "[Monitor-QSI] %s", qsi.summary()
                        )
            with self._lock:
                self._latest_qsi.update(qsi_results)

        # Provenance audit log (lightweight — every 30 cycles)
        cycle_num = len(self._history)
        if cycle_num % 30 == 0:
            self.provenance.record("monitor_snapshot", {
                "cycle": cycle_num,
                "n_metrics": len(readings),
                "alerts_fired": len(fired_alerts),
                "backend": self.backend_name,
            })

        snapshot["alerts"] = fired_alerts

        with self._lock:
            self._latest_snapshot = snapshot
            history_row = {
                r.metric_name: r.value for r in readings
            }
            history_row["timestamp"] = snapshot["timestamp"]
            self._history.append(history_row)

        return snapshot
