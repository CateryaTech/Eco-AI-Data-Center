"""
optimizer.py
============
Data center optimisation engine.
All public functions are CATERYA-compatible — they accept a pandas
DataFrame as first argument and return structured result dicts.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("eco_ai.optimizer")


# ---------------------------------------------------------------------------
# PUE / WUE / CUE helpers
# ---------------------------------------------------------------------------

def compute_pue(data: pd.DataFrame, **kwargs) -> dict:
    """
    Compute Power Usage Effectiveness (PUE).

    PUE = Total Facility Power / IT Equipment Power

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: 'total_power_kw', 'it_power_kw'

    Returns
    -------
    dict with keys: pue, efficiency_rating, recommendations
    """
    required = {"total_power_kw", "it_power_kw"}
    _validate_columns(data, required, "compute_pue")

    total_power = data["total_power_kw"].mean()
    it_power = data["it_power_kw"].mean()

    if it_power <= 0:
        raise ValueError("IT power must be > 0 kW.")

    pue = round(total_power / it_power, 4)
    efficiency_rating = _pue_rating(pue)
    recommendations = _pue_recommendations(pue)

    result = {
        "pue": pue,
        "total_power_kw": round(total_power, 2),
        "it_power_kw": round(it_power, 2),
        "efficiency_rating": efficiency_rating,
        "recommendations": recommendations,
    }

    logger.info("[Optimizer] PUE=%.4f (%s)", pue, efficiency_rating)
    return result


def compute_wue(data: pd.DataFrame, **kwargs) -> dict:
    """
    Compute Water Usage Effectiveness (WUE).

    WUE = Annual Site Water Usage (litres) / IT Equipment Energy (kWh)

    Parameters
    ----------
    data : pd.DataFrame
        Must contain: 'water_usage_litres', 'it_energy_kwh'
    """
    required = {"water_usage_litres", "it_energy_kwh"}
    _validate_columns(data, required, "compute_wue")

    water = data["water_usage_litres"].sum()
    energy = data["it_energy_kwh"].sum()

    if energy <= 0:
        raise ValueError("IT energy must be > 0 kWh.")

    wue = round(water / energy, 4)

    result = {
        "wue": wue,
        "total_water_litres": round(water, 2),
        "total_it_energy_kwh": round(energy, 2),
        "rating": "Excellent" if wue < 1.0 else "Good" if wue < 2.0 else "Needs Improvement",
    }

    logger.info("[Optimizer] WUE=%.4f", wue)
    return result


def compute_carbon(data: pd.DataFrame, **kwargs) -> dict:
    """
    Compute total carbon footprint and intensity.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain: 'energy_kwh', 'carbon_intensity_kg_per_kwh'
    """
    required = {"energy_kwh", "carbon_intensity_kg_per_kwh"}
    _validate_columns(data, required, "compute_carbon")

    total_co2 = (data["energy_kwh"] * data["carbon_intensity_kg_per_kwh"]).sum()
    avg_intensity = data["carbon_intensity_kg_per_kwh"].mean()

    result = {
        "total_co2_kg": round(total_co2, 2),
        "total_co2_tonnes": round(total_co2 / 1000, 4),
        "avg_carbon_intensity": round(avg_intensity, 6),
        "renewable_recommendation": (
            "Consider increasing renewable energy mix to reduce carbon intensity."
            if avg_intensity > 0.3 else
            "Carbon intensity is within acceptable range. Keep up the good work!"
        ),
    }

    logger.info("[Optimizer] Total CO₂=%.2f kg", total_co2)
    return result


def full_optimization(data: pd.DataFrame, **kwargs) -> dict:
    """
    Run all optimisation metrics in a single pass.
    This is the primary function wrapped by CATERYAEvaluator.
    """
    results: dict = {}

    if {"total_power_kw", "it_power_kw"}.issubset(data.columns):
        results["pue"] = compute_pue(data)
    if {"water_usage_litres", "it_energy_kwh"}.issubset(data.columns):
        results["wue"] = compute_wue(data)
    if {"energy_kwh", "carbon_intensity_kg_per_kwh"}.issubset(data.columns):
        results["carbon"] = compute_carbon(data)

    if not results:
        logger.warning("[Optimizer] No applicable metric columns found in dataset.")
        results["warning"] = "No recognised metric columns found."

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(data: pd.DataFrame, required: set, fn_name: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{fn_name}: Missing required columns: {missing}. "
            f"Available columns: {list(data.columns)}"
        )


def _pue_rating(pue: float) -> str:
    if pue <= 1.2:
        return "World Class"
    if pue <= 1.5:
        return "Efficient"
    if pue <= 2.0:
        return "Average"
    return "Inefficient"


def _pue_recommendations(pue: float) -> list:
    recs = []
    if pue > 1.2:
        recs.append("Review cooling system efficiency — hot/cold aisle containment recommended.")
    if pue > 1.5:
        recs.append("Upgrade to energy-efficient UPS systems (>97% efficiency).")
    if pue > 2.0:
        recs.append("Comprehensive energy audit urgently required.")
        recs.append("Consider migrating workloads to hyperscale cloud providers.")
    if not recs:
        recs.append("PUE is excellent. Continue monitoring and maintain current practices.")
    return recs
