"""
QuantumFairnessEvaluator
========================
Simulates quantum-inspired fairness evaluation using superposition-like
probabilistic sampling across resource distribution scenarios.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("caterya.fairness")


class QuantumFairnessEvaluator:
    """
    Quantum-inspired fairness evaluator.

    Uses Monte Carlo sampling to simulate superposition of fairness
    states across multiple resource allocation scenarios, producing
    a probabilistic fairness score.

    Parameters
    ----------
    n_samples   : int   — number of Monte Carlo draws (default 1000)
    seed        : int   — random seed for reproducibility
    verbose     : bool  — enable debug logging
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def evaluate(self, data: pd.DataFrame, sensitive_columns: Optional[list] = None) -> dict:
        """
        Run quantum fairness evaluation on the dataset.

        Parameters
        ----------
        data              : pd.DataFrame — input metrics
        sensitive_columns : list of column names considered "sensitive resources"

        Returns
        -------
        dict with keys: fairness_score, disparity_map, passed, details
        """
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return self._empty_result()

        target_cols = sensitive_columns if sensitive_columns else list(numeric.columns)
        target_cols = [c for c in target_cols if c in numeric.columns]

        if not target_cols:
            return self._empty_result()

        logger.debug("[QFE] Evaluating fairness on columns: %s", target_cols)

        disparity_map = {}
        col_scores = []

        for col in target_cols:
            values = numeric[col].dropna().values
            if len(values) < 2:
                continue

            # Quantum superposition simulation: sample N subsets
            disparities = []
            for _ in range(self.n_samples):
                idx_a = self.rng.choice(len(values), size=len(values) // 2, replace=False)
                idx_b = np.setdiff1d(np.arange(len(values)), idx_a)
                if len(idx_b) == 0:
                    continue
                mean_a = np.mean(values[idx_a])
                mean_b = np.mean(values[idx_b])
                denom = (abs(mean_a) + abs(mean_b)) / 2
                disparity = abs(mean_a - mean_b) / denom if denom != 0 else 0.0
                disparities.append(disparity)

            if disparities:
                mean_disparity = float(np.mean(disparities))
                col_score = max(0.0, 1.0 - mean_disparity)
                disparity_map[col] = round(mean_disparity, 4)
                col_scores.append(col_score)

        fairness_score = round(float(np.mean(col_scores)), 4) if col_scores else 0.5
        passed = fairness_score >= 0.6

        result = {
            "fairness_score": fairness_score,
            "disparity_map": disparity_map,
            "passed": passed,
            "details": {
                "n_samples": self.n_samples,
                "evaluated_columns": target_cols,
                "individual_scores": {
                    col: round(1.0 - disparity_map.get(col, 0.0), 4)
                    for col in target_cols
                    if col in disparity_map
                },
            },
        }

        logger.info(
            "[QFE] Fairness score: %.4f | %s",
            fairness_score,
            "PASSED" if passed else "FAILED",
        )
        return result

    def _empty_result(self) -> dict:
        return {
            "fairness_score": 0.5,
            "disparity_map": {},
            "passed": False,
            "details": {"error": "No numeric data available for fairness evaluation."},
        }
