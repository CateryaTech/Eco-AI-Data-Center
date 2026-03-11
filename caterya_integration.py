"""
caterya_integration.py
======================
Central integration layer that wires together:
  - CATERYAEvaluator  (COS scoring)
  - QuantumFairnessEvaluator (QFE)
  - ProvenanceChain   (audit trail)
  - EthicsSwarm       (consensus validation)
  - AWS boto3         (auto-scaling)
  - Dask              (big data support)

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from evaluator import CATERYAEvaluator
from fairness import QuantumFairnessEvaluator
from provenance import ProvenanceChain
from swarm import EthicsSwarm
from scoring import COSScore

logger = logging.getLogger("eco_ai.caterya_integration")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import dask.dataframe as dd  # type: ignore
    DASK_AVAILABLE = True
except ImportError:
    dd = None
    DASK_AVAILABLE = False
    logger.warning("Dask not installed. Big data support disabled. `pip install dask`")

try:
    import boto3  # type: ignore
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. AWS auto-scaling disabled. `pip install boto3`")


# ---------------------------------------------------------------------------
# Main integration class
# ---------------------------------------------------------------------------

class EcoAIDataCenterCATERYA:
    """
    High-level integration class for Eco AI Data Center.

    Wraps optimisation functions with the full CATERYA ethical AI pipeline:
    1. Provenance recording
    2. COS evaluation (entropy, symmetry, information, fairness)
    3. Quantum fairness evaluation
    4. Ethics swarm consensus
    5. Guardrail enforcement (threshold-based blocking)

    Parameters
    ----------
    cos_threshold     : float   — COS guardrail threshold (default 0.7)
    swarm_threshold   : float   — Swarm consensus threshold (default 0.7)
    model_id          : str     — Identifier for provenance chain
    use_dask          : bool    — Enable Dask for large datasets (>500k rows)
    aws_region        : str     — AWS region for auto-scaling
    verbose           : bool    — Enable debug logging
    """

    def __init__(
        self,
        cos_threshold: float = 0.7,
        swarm_threshold: float = 0.7,
        model_id: str = "eco-ai-optimizer",
        use_dask: bool = False,
        aws_region: str = "ap-southeast-1",
        verbose: bool = False,
    ):
        self.cos_threshold = cos_threshold
        self.swarm_threshold = swarm_threshold
        self.model_id = model_id
        self.use_dask = use_dask and DASK_AVAILABLE
        self.aws_region = aws_region
        self.verbose = verbose

        # Initialise CATERYA components
        self.evaluator = CATERYAEvaluator(threshold=cos_threshold, verbose=verbose)
        self.fairness_evaluator = QuantumFairnessEvaluator(n_samples=500, seed=42)
        self.provenance = ProvenanceChain(model_id=model_id, owner="CateryaTech")
        self.swarm = EthicsSwarm(n_agents=5, threshold=swarm_threshold)

        # Session state
        self.last_cos: Optional[COSScore] = None
        self.last_swarm_result: Optional[dict] = None
        self.last_fairness_result: Optional[dict] = None

        # Record chain genesis
        self.provenance.record("system_init", {
            "model_id": model_id,
            "cos_threshold": cos_threshold,
            "swarm_threshold": swarm_threshold,
            "dask_enabled": self.use_dask,
            "aws_region": aws_region,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(
            "[EcoAI-CATERYA] Initialised | model=%s | COS_thresh=%.2f",
            model_id, cos_threshold,
        )

    # ------------------------------------------------------------------
    # Core: evaluate & wrap an optimizer
    # ------------------------------------------------------------------

    def run_with_evaluation(
        self,
        optimizer_fn: Callable,
        data: pd.DataFrame,
        sensitive_columns: Optional[list] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Run an optimizer function through the full CATERYA pipeline.

        Parameters
        ----------
        optimizer_fn      : callable — e.g. compute_pue, optimizer.run
        data              : pd.DataFrame
        sensitive_columns : columns to focus fairness evaluation on
        **kwargs          : forwarded to optimizer_fn

        Returns
        -------
        dict with keys:
            result         — optimizer output
            cos            — COSScore object
            fairness       — QFE result dict
            swarm          — EthicsSwarm verdict
            approved       — bool, overall gate pass
            audit_hash     — provenance head hash
        """
        logger.info("[EcoAI-CATERYA] Starting full evaluation pipeline …")

        # Step 1 — Data ingestion provenance
        self.provenance.record("data_ingestion", {
            "rows": len(data),
            "columns": list(data.columns),
            "memory_mb": round(data.memory_usage(deep=True).sum() / 1e6, 2),
            "nulls": int(data.isnull().sum().sum()),
        })

        # Step 2 — Optionally load via Dask for large datasets
        data = self._maybe_load_dask(data)

        # Step 3 — COS evaluation (wraps optimizer)
        opt_result, cos = self.evaluator.evaluate(optimizer_fn, data, **kwargs)
        self.last_cos = cos
        self.provenance.record("cos_evaluation", cos.to_dict())

        # Step 4 — Quantum Fairness Evaluation
        fairness_result = self.fairness_evaluator.evaluate(
            data, sensitive_columns=sensitive_columns
        )
        self.last_fairness_result = fairness_result
        self.provenance.record("fairness_evaluation", fairness_result)

        # Step 5 — Ethics Swarm consensus
        swarm_result = self.swarm.deliberate(cos)
        self.last_swarm_result = swarm_result
        self.provenance.record("swarm_deliberation", swarm_result)

        # Step 6 — Guardrail gate
        approved = cos.passed and swarm_result["approved"]
        self.provenance.record("guardrail_decision", {
            "approved": approved,
            "cos_passed": cos.passed,
            "swarm_approved": swarm_result["approved"],
            "cos_composite": cos.composite,
            "swarm_consensus": swarm_result["consensus_score"],
        })

        if not approved:
            logger.warning(
                "[EcoAI-CATERYA] ⚠️  GUARDRAIL TRIGGERED — deployment blocked. "
                "COS=%.4f, Swarm=%.4f",
                cos.composite, swarm_result["consensus_score"],
            )
        else:
            logger.info(
                "[EcoAI-CATERYA] ✅ All checks passed — deployment approved. "
                "COS=%.4f, Swarm=%.4f",
                cos.composite, swarm_result["consensus_score"],
            )

        return {
            "result": opt_result,
            "cos": cos,
            "fairness": fairness_result,
            "swarm": swarm_result,
            "approved": approved,
            "audit_hash": self.provenance._prev_hash,
        }

    def score_data(self, data: pd.DataFrame) -> COSScore:
        """
        Quick COS scoring without running an optimizer.
        Useful for live dashboard refresh.
        """
        cos = self.evaluator.score_only(data)
        self.last_cos = cos
        self.provenance.record("dashboard_score", cos.to_dict())
        return cos

    # ------------------------------------------------------------------
    # Big Data — Dask support
    # ------------------------------------------------------------------

    def load_large_csv(self, path: str, blocksize: str = "64MB") -> pd.DataFrame:
        """
        Load a large CSV using Dask and compute into a pandas DataFrame.
        Falls back to pandas.read_csv if Dask is unavailable.
        """
        if not DASK_AVAILABLE:
            logger.info("[EcoAI-CATERYA] Falling back to pandas.read_csv (Dask unavailable)")
            return pd.read_csv(path)

        logger.info("[EcoAI-CATERYA] Loading via Dask: %s (blocksize=%s)", path, blocksize)
        ddf = dd.read_csv(path, blocksize=blocksize)
        df = ddf.compute()
        self.provenance.record("dask_load", {
            "path": path, "rows": len(df), "blocksize": blocksize
        })
        return df

    def _maybe_load_dask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert to Dask DataFrame for processing if dataset is large (>500k rows)."""
        if self.use_dask and DASK_AVAILABLE and len(data) > 500_000:
            logger.info("[EcoAI-CATERYA] Large dataset (%d rows) — using Dask", len(data))
            ddf = dd.from_pandas(data, npartitions=max(4, len(data) // 100_000))
            return ddf.compute()  # Compute back for downstream compatibility
        return data

    # ------------------------------------------------------------------
    # AWS Auto-Scaling
    # ------------------------------------------------------------------

    def trigger_aws_scale_out(
        self,
        asg_name: str,
        desired_capacity: int,
        min_size: int = 1,
        max_size: int = 20,
    ) -> dict:
        """
        Trigger AWS Auto Scaling Group scale-out event.

        Parameters
        ----------
        asg_name         : str — Auto Scaling Group name
        desired_capacity : int — target number of instances
        min_size         : int — minimum instances
        max_size         : int — maximum instances

        Returns
        -------
        dict with status and response metadata
        """
        if not BOTO3_AVAILABLE:
            logger.warning("[EcoAI-CATERYA] boto3 unavailable — simulating AWS scale-out.")
            return self._simulate_aws_scale_out(asg_name, desired_capacity)

        try:
            client = boto3.client("autoscaling", region_name=self.aws_region)
            response = client.update_auto_scaling_group(
                AutoScalingGroupName=asg_name,
                MinSize=min_size,
                MaxSize=max_size,
                DesiredCapacity=desired_capacity,
            )
            self.provenance.record("aws_scale_out", {
                "asg_name": asg_name,
                "desired_capacity": desired_capacity,
                "region": self.aws_region,
                "response_metadata": response.get("ResponseMetadata", {}),
            })
            logger.info(
                "[EcoAI-CATERYA] AWS ASG '%s' scaled to %d instances.",
                asg_name, desired_capacity,
            )
            return {"status": "success", "asg_name": asg_name, "desired": desired_capacity}

        except Exception as exc:  # noqa: BLE001
            logger.error("[EcoAI-CATERYA] AWS scale-out failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _simulate_aws_scale_out(self, asg_name: str, desired_capacity: int) -> dict:
        """Return simulated response when boto3 is not available."""
        self.provenance.record("aws_scale_out_simulated", {
            "asg_name": asg_name,
            "desired_capacity": desired_capacity,
            "note": "boto3 not installed — simulation mode",
        })
        return {
            "status": "simulated",
            "asg_name": asg_name,
            "desired": desired_capacity,
            "note": "Install boto3 and configure AWS credentials for real scaling.",
        }

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_audit_report(self) -> dict:
        """Export full provenance audit report."""
        return self.provenance.export_audit_report()

    def get_audit_json(self) -> str:
        """Export audit report as JSON string."""
        return self.provenance.export_json()

    def chain_integrity_ok(self) -> bool:
        """Verify provenance chain integrity."""
        return self.provenance.verify()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cos_score(self) -> Optional[float]:
        return self.last_cos.composite if self.last_cos else None

    @property
    def cos_passed(self) -> Optional[bool]:
        return self.last_cos.passed if self.last_cos else None

    @property
    def is_dask_available(self) -> bool:
        return DASK_AVAILABLE

    @property
    def is_aws_available(self) -> bool:
        return BOTO3_AVAILABLE
