"""
CATERYAEvaluator
================
Core ethical AI evaluator that wraps existing optimization functions
and computes COS (CATERYA Open Score) based on entropy, symmetry,
and information quality of energy/water/carbon datasets.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

import math
import logging
from typing import Callable, Any, Optional
import numpy as np
import pandas as pd

from scoring import COSScore

logger = logging.getLogger("caterya.evaluator")


class CATERYAEvaluator:
    """
    Wraps any optimization function and evaluates its ethical quality
    via the CATERYA Open Score (COS) framework.

    Usage
    -----
    >>> evaluator = CATERYAEvaluator(threshold=0.7)
    >>> result, cos = evaluator.evaluate(my_optimizer_fn, input_df)
    >>> print(cos.summary())
    """

    def __init__(
        self,
        threshold: float = 0.7,
        entropy_bins: int = 20,
        verbose: bool = False,
    ):
        self.threshold = threshold
        self.entropy_bins = entropy_bins
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        optimizer_fn: Callable,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple:
        """
        Run optimizer_fn(data, **kwargs) and attach a COS score to the result.

        Parameters
        ----------
        optimizer_fn : callable
            Any function that accepts a DataFrame as first argument.
        data         : pd.DataFrame
            Input dataset (energy, water, carbon metrics).
        **kwargs     : forwarded to optimizer_fn

        Returns
        -------
        (result, COSScore)
        """
        logger.info("[CATERYA] Starting ethical evaluation pipeline …")

        # Step 1: Run the wrapped optimizer
        result = optimizer_fn(data, **kwargs)

        # Step 2: Compute sub-scores
        cos = COSScore(threshold=self.threshold)
        cos.entropy_score = self._compute_entropy(data)
        cos.symmetry_score = self._compute_symmetry(data)
        cos.information_score = self._compute_information(data)
        cos.fairness_score = self._compute_fairness_proxy(data)
        cos.compute_composite()

        # Step 3: Collect warnings
        self._attach_warnings(cos, data)

        # Step 4: Metadata
        cos.metadata = {
            "rows": len(data),
            "columns": list(data.columns),
            "optimizer": getattr(optimizer_fn, "__name__", str(optimizer_fn)),
            "entropy_bins": self.entropy_bins,
        }

        logger.info("[CATERYA] %s", cos.summary())
        return result, cos

    def score_only(self, data: pd.DataFrame) -> COSScore:
        """
        Compute COS without running any optimizer — useful for dashboard refresh.
        """
        cos = COSScore(threshold=self.threshold)
        cos.entropy_score = self._compute_entropy(data)
        cos.symmetry_score = self._compute_symmetry(data)
        cos.information_score = self._compute_information(data)
        cos.fairness_score = self._compute_fairness_proxy(data)
        cos.compute_composite()
        self._attach_warnings(cos, data)
        cos.metadata = {"rows": len(data), "columns": list(data.columns)}
        return cos

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _compute_entropy(self, data: pd.DataFrame) -> float:
        """
        Shannon entropy averaged across numeric columns, normalised to [0, 1].
        Higher entropy → richer information → higher score.
        """
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            logger.warning("[CATERYA] No numeric columns found for entropy computation.")
            return 0.0

        entropies = []
        for col in numeric.columns:
            values = numeric[col].dropna().values
            if len(values) < 2:
                continue
            counts, _ = np.histogram(values, bins=self.entropy_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            max_h = math.log2(self.entropy_bins)
            entropies.append(h / max_h if max_h > 0 else 0.0)

        return float(np.mean(entropies)) if entropies else 0.0

    def _compute_symmetry(self, data: pd.DataFrame) -> float:
        """
        Measures distributional symmetry (skewness penalty).
        Score = 1 - mean(|skew| / 3) clipped to [0, 1].
        """
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.5

        skews = []
        for col in numeric.columns:
            values = numeric[col].dropna().values
            if len(values) < 3:
                continue
            skew = float(pd.Series(values).skew())
            skews.append(min(abs(skew) / 3.0, 1.0))

        if not skews:
            return 0.5

        mean_skew_penalty = float(np.mean(skews))
        return round(1.0 - mean_skew_penalty, 4)

    def _compute_information(self, data: pd.DataFrame) -> float:
        """
        Data completeness and consistency score.
        Penalises missing values and zero-variance columns.
        """
        total_cells = data.size
        if total_cells == 0:
            return 0.0

        # Completeness
        missing_ratio = data.isnull().sum().sum() / total_cells
        completeness = 1.0 - missing_ratio

        # Variance quality (zero-variance columns are uninformative)
        numeric = data.select_dtypes(include=[np.number])
        if not numeric.empty:
            zero_var_cols = (numeric.std() == 0).sum()
            var_quality = 1.0 - (zero_var_cols / len(numeric.columns))
        else:
            var_quality = 0.5

        return round((completeness * 0.6 + var_quality * 0.4), 4)

    def _compute_fairness_proxy(self, data: pd.DataFrame) -> float:
        """
        Lightweight fairness proxy: coefficient of variation penalty across
        numeric columns. Low CV spread → more equitable resource distribution.
        """
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.5

        cvs = []
        for col in numeric.columns:
            values = numeric[col].dropna().values
            if len(values) < 2:
                continue
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) if mean != 0 else 0.0
            cvs.append(min(cv, 2.0) / 2.0)  # normalise

        if not cvs:
            return 0.5

        mean_cv_penalty = float(np.mean(cvs))
        return round(1.0 - mean_cv_penalty, 4)

    def _attach_warnings(self, cos: COSScore, data: pd.DataFrame) -> None:
        """Add human-readable advisory warnings to the COS object."""
        if cos.entropy_score < 0.4:
            cos.warnings.append(
                "Low entropy detected — consider enriching your dataset diversity."
            )
        if cos.symmetry_score < 0.4:
            cos.warnings.append(
                "High skewness detected — data distribution is asymmetric; "
                "this may bias optimisation results."
            )
        if cos.information_score < 0.5:
            cos.warnings.append(
                "Poor data quality — missing values or zero-variance columns found. "
                "Imputation recommended."
            )
        if cos.fairness_score < 0.4:
            cos.warnings.append(
                "High coefficient of variation across resources — "
                "unequal distribution may indicate fairness concerns."
            )
        if not cos.passed:
            cos.warnings.append(
                f"COS below threshold ({cos.threshold}). "
                "Review data quality and fairness before deploying this model."
            )
