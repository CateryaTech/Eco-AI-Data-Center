"""
CATERYA Open Score (COS) Data Model
=====================================
Eco AI Data Center — CateryaTech
"""

from dataclasses import dataclass, field
from typing import Optional
import datetime


@dataclass
class COSScore:
    """
    CATERYA Open Score (COS) — composite ethical AI score.

    Attributes
    ----------
    entropy_score      : Information entropy quality (0-1)
    symmetry_score     : Distribution symmetry & bias absence (0-1)
    information_score  : Provenance & data completeness (0-1)
    fairness_score     : Quantum fairness evaluation result (0-1)
    composite          : Final weighted COS (0-1)
    passed             : True if composite >= threshold
    threshold          : Decision boundary (default 0.7)
    timestamp          : Evaluation timestamp
    warnings           : Human-readable advisory messages
    metadata           : Arbitrary audit metadata
    """

    entropy_score: float = 0.0
    symmetry_score: float = 0.0
    information_score: float = 0.0
    fairness_score: float = 0.0
    composite: float = 0.0
    passed: bool = False
    threshold: float = 0.7
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Weight configuration (must sum to 1.0)
    WEIGHTS = {
        "entropy": 0.30,
        "symmetry": 0.25,
        "information": 0.25,
        "fairness": 0.20,
    }

    def compute_composite(self) -> float:
        """Compute weighted composite COS from sub-scores."""
        w = self.WEIGHTS
        self.composite = round(
            w["entropy"] * self.entropy_score
            + w["symmetry"] * self.symmetry_score
            + w["information"] * self.information_score
            + w["fairness"] * self.fairness_score,
            4,
        )
        self.passed = self.composite >= self.threshold
        return self.composite

    def to_dict(self) -> dict:
        return {
            "entropy_score": self.entropy_score,
            "symmetry_score": self.symmetry_score,
            "information_score": self.information_score,
            "fairness_score": self.fairness_score,
            "composite": self.composite,
            "passed": self.passed,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "⚠️ FAILED"
        return (
            f"COS {status} | Composite: {self.composite:.4f} "
            f"(threshold={self.threshold}) | "
            f"Entropy={self.entropy_score:.3f}, "
            f"Symmetry={self.symmetry_score:.3f}, "
            f"Info={self.information_score:.3f}, "
            f"Fairness={self.fairness_score:.3f}"
        )
