"""
EthicsSwarm
===========
Multi-agent swarm consensus engine for ethical AI validation.
Multiple independent "ethics agents" each evaluate a COS score from
a different perspective, then vote via weighted consensus.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

import logging
import random
from dataclasses import dataclass
from typing import List
import numpy as np

from scoring import COSScore

logger = logging.getLogger("caterya.swarm")


@dataclass
class EthicsAgent:
    """A single ethics evaluation agent with a specific focus."""
    name: str
    focus: str       # "entropy" | "symmetry" | "information" | "fairness" | "holistic"
    weight: float    # influence weight in consensus vote
    vote: float = 0.0
    rationale: str = ""


class EthicsSwarm:
    """
    Runs a swarm of ethics agents that each cast a vote on a COS score,
    producing a consensus verdict.

    Parameters
    ----------
    n_agents   : int — number of agents in the swarm (default 5)
    seed       : int — random seed
    threshold  : float — consensus approval threshold (default 0.7)
    """

    DEFAULT_AGENTS = [
        EthicsAgent("Entropy Guardian",    "entropy",     0.25),
        EthicsAgent("Symmetry Warden",     "symmetry",    0.20),
        EthicsAgent("Information Auditor", "information", 0.20),
        EthicsAgent("Fairness Arbiter",    "fairness",    0.25),
        EthicsAgent("Holistic Overseer",   "holistic",    0.10),
    ]

    def __init__(
        self,
        n_agents: int = 5,
        seed: int = 42,
        threshold: float = 0.7,
    ):
        self.n_agents = n_agents
        self.seed = seed
        self.threshold = threshold
        self.rng = random.Random(seed)
        self.agents: List[EthicsAgent] = list(self.DEFAULT_AGENTS[:n_agents])

    def deliberate(self, cos: COSScore) -> dict:
        """
        Run swarm deliberation on a COS score.

        Parameters
        ----------
        cos : COSScore — the evaluated score from CATERYAEvaluator

        Returns
        -------
        dict with keys: consensus_score, approved, votes, rationales, summary
        """
        total_weight = sum(a.weight for a in self.agents)
        weighted_votes = 0.0

        for agent in self.agents:
            vote, rationale = self._agent_vote(agent, cos)
            agent.vote = vote
            agent.rationale = rationale
            weighted_votes += vote * (agent.weight / total_weight)

        consensus_score = round(weighted_votes, 4)
        approved = consensus_score >= self.threshold

        result = {
            "consensus_score": consensus_score,
            "approved": approved,
            "threshold": self.threshold,
            "votes": {
                a.name: {
                    "focus": a.focus,
                    "vote": round(a.vote, 4),
                    "weight": a.weight,
                    "rationale": a.rationale,
                }
                for a in self.agents
            },
            "summary": (
                f"{'✅ APPROVED' if approved else '⛔ REJECTED'} by ethics swarm "
                f"(consensus={consensus_score:.4f}, threshold={self.threshold})"
            ),
        }

        logger.info("[EthicsSwarm] %s", result["summary"])
        return result

    def _agent_vote(self, agent: EthicsAgent, cos: COSScore) -> tuple:
        """Determine an individual agent's vote and rationale."""
        focus_map = {
            "entropy":     cos.entropy_score,
            "symmetry":    cos.symmetry_score,
            "information": cos.information_score,
            "fairness":    cos.fairness_score,
            "holistic":    cos.composite,
        }

        base_score = focus_map.get(agent.focus, cos.composite)

        # Add slight stochastic perturbation to simulate independent perspective
        noise = self.rng.uniform(-0.03, 0.03)
        vote = float(np.clip(base_score + noise, 0.0, 1.0))

        # Build rationale
        if vote >= 0.8:
            rationale = f"Excellent {agent.focus} quality. Strongly endorses deployment."
        elif vote >= self.threshold:
            rationale = f"Acceptable {agent.focus} level. Conditional approval granted."
        elif vote >= 0.5:
            rationale = f"Marginal {agent.focus} quality. Recommends monitoring post-deployment."
        else:
            rationale = (
                f"Insufficient {agent.focus} score. Blocks deployment pending remediation."
            )

        return vote, rationale
