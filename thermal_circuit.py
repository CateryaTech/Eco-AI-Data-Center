"""
quantum/thermal_circuit.py
===========================
Quantum circuit simulations for thermal distribution modelling
in data center environments.

Uses PennyLane for quantum circuit simulation with a classical
numpy fallback when PennyLane is not installed.

Key features:
  - Thermal distribution simulation via amplitude encoding
  - Quantum interference patterns for hot-spot detection
  - Fairness metrics on quantum circuit outputs
  - QuantumFairnessEvaluator integration for output validation

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from fairness import QuantumFairnessEvaluator
from provenance import ProvenanceChain

logger = logging.getLogger("eco_ai.quantum")

# ---------------------------------------------------------------------------
# Optional PennyLane
# ---------------------------------------------------------------------------

try:
    import pennylane as qml  # type: ignore
    PENNYLANE_AVAILABLE = True
    logger.info("[QuantumCircuit] PennyLane available — using real QC simulation.")
except ImportError:
    qml = None
    PENNYLANE_AVAILABLE = False
    logger.info(
        "[QuantumCircuit] PennyLane not installed — using classical numpy simulation. "
        "Install with: pip install pennylane"
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ThermalSimResult:
    """Result from a thermal distribution quantum simulation."""
    n_qubits: int
    zone_labels: List[str]
    raw_probabilities: List[float]       # from quantum circuit output
    thermal_scores: List[float]          # normalised thermal load per zone
    hotspot_zones: List[str]             # zones above hotspot_threshold
    hotspot_threshold: float
    fairness_score: float                # QFE fairness on thermal distribution
    fairness_passed: bool
    disparity_map: dict
    backend: str                         # "pennylane" | "numpy-simulation"
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())
    circuit_depth: int = 0
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        hs = ", ".join(self.hotspot_zones) if self.hotspot_zones else "None"
        return (
            f"Thermal Simulation | {self.n_qubits}-qubit | "
            f"Hotspots: [{hs}] | "
            f"Fairness: {self.fairness_score:.4f} "
            f"({'✅' if self.fairness_passed else '⚠️'}) | "
            f"Backend: {self.backend}"
        )


# ---------------------------------------------------------------------------
# Classical numpy simulation of quantum thermal circuit
# ---------------------------------------------------------------------------

class ClassicalQuantumSimulator:
    """
    Classical simulation of quantum-like thermal distribution.
    
    Mimics quantum amplitude encoding + interference without requiring
    an actual quantum computer or PennyLane installation.

    The simulation encodes thermal readings as quantum state amplitudes,
    applies Hadamard-like mixing, and measures resulting probability
    distributions — revealing interference patterns (hot-spot correlations).
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits

    def _hadamard_transform(self, state: np.ndarray) -> np.ndarray:
        """Apply Hadamard transform to a state vector."""
        n = len(state)
        if n == 1:
            return state
        H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        result = state.copy()
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = result[i + j]
                    v = result[i + j + step]
                    result[i + j]        = (H[0, 0] * u + H[0, 1] * v)
                    result[i + j + step] = (H[1, 0] * u + H[1, 1] * v)
            step *= 2
        return result

    def _ry_rotation(self, state: np.ndarray, theta: float, qubit: int) -> np.ndarray:
        """Apply RY rotation to qubit k in the state vector."""
        cos_t = math.cos(theta / 2)
        sin_t = math.sin(theta / 2)
        result = state.copy()
        for i in range(len(state)):
            # Check if qubit k is 0 in state i
            if not (i >> qubit & 1):
                j = i | (1 << qubit)
                a, b = result[i], result[j]
                result[i] =  cos_t * a - sin_t * b
                result[j] =  sin_t * a + cos_t * b
        return result

    def _entangle_adjacent(self, state: np.ndarray) -> np.ndarray:
        """Apply CNOT-like entanglement between adjacent qubits."""
        result = state.copy()
        for qubit in range(self.n_qubits - 1):
            for i in range(len(state)):
                if (i >> qubit) & 1:  # control qubit is 1
                    j = i ^ (1 << (qubit + 1))
                    result[i], result[j] = result[j], result[i]
        return result

    def simulate(self, thermal_readings: List[float]) -> np.ndarray:
        """
        Encode thermal readings as rotation angles → apply circuit → measure.
        Returns probability distribution over 2^n states.
        """
        # Normalise thermal readings to rotation angles [0, π]
        arr = np.array(thermal_readings, dtype=float)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        thetas = arr_norm * math.pi

        # Initial |0⟩ state
        state = np.zeros(self.n_states)
        state[0] = 1.0

        # Apply Hadamard to all qubits (superposition)
        state = self._hadamard_transform(state)

        # Encode thermal data via RY rotations
        for qubit_idx in range(min(self.n_qubits, len(thetas))):
            state = self._ry_rotation(state, thetas[qubit_idx], qubit_idx)

        # Entanglement layer (correlation between zones)
        state = self._entangle_adjacent(state)

        # Final Hadamard (interference)
        state = self._hadamard_transform(state)

        # Measure: probabilities = |amplitude|²
        probabilities = state ** 2
        probabilities = np.abs(probabilities)
        probabilities /= probabilities.sum()  # normalise

        return probabilities


# ---------------------------------------------------------------------------
# PennyLane circuit (used when available)
# ---------------------------------------------------------------------------

def build_pennylane_circuit(n_qubits: int, thermal_readings: List[float]):
    """
    Build and execute a PennyLane quantum circuit for thermal simulation.
    Returns probability distribution.
    """
    if not PENNYLANE_AVAILABLE:
        raise RuntimeError("PennyLane not available.")

    dev = qml.device("default.qubit", wires=n_qubits)
    arr = np.array(thermal_readings[:n_qubits], dtype=float)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    thetas = arr_norm * np.pi

    @qml.qnode(dev)
    def circuit():
        # Superposition layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Thermal encoding via RY rotations
        for i, theta in enumerate(thetas):
            qml.RY(theta, wires=i)

        # Entanglement — simulate thermal correlations between zones
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Interference layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        return qml.probs(wires=range(n_qubits))

    probabilities = circuit()
    return np.array(probabilities), circuit.tape.depth if hasattr(circuit, "tape") else n_qubits * 3


# ---------------------------------------------------------------------------
# Main thermal simulation engine
# ---------------------------------------------------------------------------

class QuantumThermalSimulator:
    """
    Quantum-inspired thermal distribution simulator for data center zones.

    Encodes rack/zone temperatures as quantum state amplitudes and simulates
    thermal interference patterns to identify hot spots and unequal heat
    distribution.

    Integrates with QuantumFairnessEvaluator to compute fairness metrics
    on the resulting thermal probability distributions.

    Parameters
    ----------
    n_qubits          : int   — number of qubits (= log2 of simulation states)
    hotspot_threshold : float — probability threshold for hot-spot detection (0-1)
    provenance        : ProvenanceChain for audit logging
    """

    def __init__(
        self,
        n_qubits: int = 4,
        hotspot_threshold: float = 0.08,
        provenance: Optional[ProvenanceChain] = None,
    ):
        self.n_qubits = n_qubits
        self.hotspot_threshold = hotspot_threshold
        self.provenance = provenance
        self.qfe = QuantumFairnessEvaluator(n_samples=300, seed=42)
        self._classical_sim = ClassicalQuantumSimulator(n_qubits=n_qubits)
        logger.info(
            "[QuantumThermal] Initialised | qubits=%d | backend=%s",
            n_qubits,
            "pennylane" if PENNYLANE_AVAILABLE else "numpy-classical",
        )

    def simulate(
        self,
        thermal_readings: List[float],
        zone_labels: Optional[List[str]] = None,
    ) -> ThermalSimResult:
        """
        Run thermal distribution simulation.

        Parameters
        ----------
        thermal_readings : List[float]
            Temperature (or any thermal metric) readings per zone/rack.
            Values should be in Celsius or normalised form.
        zone_labels : List[str], optional
            Human-readable labels for each zone (e.g. "Rack-A1", "Row-2").
            Auto-generated if not provided.

        Returns
        -------
        ThermalSimResult
        """
        n = len(thermal_readings)
        if zone_labels is None:
            zone_labels = [f"Zone-{i+1:02d}" for i in range(n)]
        else:
            zone_labels = list(zone_labels[:n])

        # Pad readings to n_qubits if needed
        padded = thermal_readings + [thermal_readings[-1]] * max(0, self.n_qubits - n)
        padded = padded[:self.n_qubits]

        # Run quantum circuit
        circuit_depth = 0
        backend = "numpy-classical"
        try:
            if PENNYLANE_AVAILABLE:
                probabilities, circuit_depth = build_pennylane_circuit(self.n_qubits, padded)
                backend = "pennylane"
            else:
                probabilities = self._classical_sim.simulate(padded)
                circuit_depth = self.n_qubits * 3
        except Exception as exc:
            logger.warning("[QuantumThermal] Circuit failed (%s) — using classical sim.", exc)
            probabilities = self._classical_sim.simulate(padded)
            circuit_depth = self.n_qubits * 3

        # Map quantum output probabilities back to input zones
        # Take first n states and renormalise
        raw_probs = list(probabilities[:n])
        total = sum(raw_probs) + 1e-12
        thermal_scores = [p / total for p in raw_probs]

        # Hot-spot detection: zones whose score exceeds threshold
        per_zone_threshold = self.hotspot_threshold * max(thermal_scores) / (sum(thermal_scores) / n + 1e-12)
        hotspot_zones = [
            zone_labels[i]
            for i, score in enumerate(thermal_scores)
            if score > self.hotspot_threshold
        ]

        # Fairness evaluation on thermal distribution
        df_thermal = pd.DataFrame({
            label: [score]
            for label, score in zip(zone_labels, thermal_scores)
        })
        fairness_result = self.qfe.evaluate(df_thermal)

        # Audit
        if self.provenance:
            self.provenance.record("quantum_thermal_simulation", {
                "n_zones": n,
                "n_qubits": self.n_qubits,
                "backend": backend,
                "circuit_depth": circuit_depth,
                "hotspot_zones": hotspot_zones,
                "fairness_score": fairness_result["fairness_score"],
            })

        result = ThermalSimResult(
            n_qubits=self.n_qubits,
            zone_labels=zone_labels[:n],
            raw_probabilities=[round(p, 6) for p in probabilities[:n]],
            thermal_scores=[round(s, 6) for s in thermal_scores],
            hotspot_zones=hotspot_zones,
            hotspot_threshold=self.hotspot_threshold,
            fairness_score=fairness_result["fairness_score"],
            fairness_passed=fairness_result["passed"],
            disparity_map=fairness_result.get("disparity_map", {}),
            backend=backend,
            circuit_depth=circuit_depth,
            metadata={
                "input_readings": thermal_readings,
                "n_zones": n,
                "n_simulation_states": 2 ** self.n_qubits,
            },
        )

        logger.info("[QuantumThermal] %s", result.summary())
        return result

    def simulate_from_df(
        self,
        df: pd.DataFrame,
        temperature_col: str = "temperature_celsius",
        zone_col: Optional[str] = None,
    ) -> ThermalSimResult:
        """
        Convenience wrapper: run simulation from a DataFrame.

        Parameters
        ----------
        df              : DataFrame with at least one temperature column
        temperature_col : name of the temperature column
        zone_col        : optional column for zone labels
        """
        if temperature_col not in df.columns:
            raise ValueError(
                f"Column '{temperature_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        readings = df[temperature_col].dropna().tolist()
        labels = None
        if zone_col and zone_col in df.columns:
            labels = df[zone_col].dropna().tolist()

        return self.simulate(readings, zone_labels=labels)

    def batch_simulate(
        self,
        readings_over_time: List[List[float]],
        zone_labels: Optional[List[str]] = None,
    ) -> List[ThermalSimResult]:
        """
        Run simulation over multiple time-steps (batch mode).
        Returns a list of ThermalSimResults.
        """
        results = []
        for i, readings in enumerate(readings_over_time):
            logger.debug("[QuantumThermal] Batch simulation step %d/%d", i + 1, len(readings_over_time))
            result = self.simulate(readings, zone_labels=zone_labels)
            results.append(result)
        return results
