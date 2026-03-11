"""
ProvenanceChain
===============
Immutable audit trail for AI decision provenance tracking.
Records every evaluation step as a signed chain of events,
enabling full enterprise-grade compliance traceability.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
"""

import hashlib
import json
import logging
import datetime
from typing import Any, Optional

logger = logging.getLogger("caterya.provenance")


class ProvenanceChain:
    """
    Append-only provenance ledger for AI model evaluations.

    Each entry is content-addressed (SHA-256 hash chaining) so the
    full history can be audited and tamper detection is trivial.

    Usage
    -----
    >>> chain = ProvenanceChain(model_id="eco-optimizer-v1")
    >>> chain.record("data_ingestion", {"rows": 500, "source": "s3://bucket/data.csv"})
    >>> chain.record("cos_evaluation", cos_score.to_dict())
    >>> report = chain.export_audit_report()
    """

    def __init__(self, model_id: str = "eco-ai-data-center", owner: str = "CateryaTech"):
        self.model_id = model_id
        self.owner = owner
        self._chain: list = []
        self._prev_hash: str = "GENESIS"

        logger.info("[Provenance] Chain initialised for model: %s", model_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event_type: str, payload: Any, actor: str = "system") -> str:
        """
        Append an event to the provenance chain.

        Parameters
        ----------
        event_type : descriptive event label (e.g. "cos_evaluation")
        payload    : dict / str — data associated with the event
        actor      : who triggered this event

        Returns
        -------
        str : SHA-256 hash of the new entry
        """
        entry = {
            "seq": len(self._chain) + 1,
            "model_id": self.model_id,
            "event_type": event_type,
            "actor": actor,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "payload": payload,
            "prev_hash": self._prev_hash,
        }

        entry_hash = self._sha256(entry)
        entry["hash"] = entry_hash
        self._chain.append(entry)
        self._prev_hash = entry_hash

        logger.debug("[Provenance] Recorded: %s → %s", event_type, entry_hash[:12])
        return entry_hash

    def verify(self) -> bool:
        """
        Re-compute hashes and verify chain integrity.
        Returns True if the chain is intact, False if tampered.
        """
        prev = "GENESIS"
        for entry in self._chain:
            stored_hash = entry.get("hash", "")
            entry_copy = {k: v for k, v in entry.items() if k != "hash"}
            entry_copy["prev_hash"] = prev
            computed = self._sha256(entry_copy)
            if computed != stored_hash:
                logger.error(
                    "[Provenance] Integrity VIOLATION at seq %s!", entry.get("seq")
                )
                return False
            prev = stored_hash
        return True

    def export_audit_report(self) -> dict:
        """Export full chain as a structured audit report."""
        return {
            "model_id": self.model_id,
            "owner": self.owner,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "chain_length": len(self._chain),
            "chain_integrity": self.verify(),
            "head_hash": self._prev_hash,
            "events": self._chain,
        }

    def export_json(self, indent: int = 2) -> str:
        """Export audit report as JSON string."""
        return json.dumps(self.export_audit_report(), indent=indent, default=str)

    def last_event(self) -> Optional[dict]:
        """Return the most recent chain entry."""
        return self._chain[-1] if self._chain else None

    def __len__(self) -> int:
        return len(self._chain)

    def __repr__(self) -> str:
        return (
            f"ProvenanceChain(model_id='{self.model_id}', "
            f"entries={len(self._chain)}, "
            f"head={self._prev_hash[:12]})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(obj: Any) -> str:
        serialised = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialised).hexdigest()
