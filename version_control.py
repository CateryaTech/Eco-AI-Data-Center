"""
simulation_versions/version_control.py
=======================================
Git-inspired version control for simulation snapshots.

Features:
  - commit() — snapshot a simulation with message and author
  - checkout() — restore a previous simulation state
  - diff() — compare two simulation versions
  - branch() / merge() — lightweight branching for experiment comparison
  - tag() — mark important milestones
  - log() — view commit history
  - export_bundle() — portable JSON export of full history

All versions are content-addressed (SHA-256) and linked to
ProvenanceChain for tamper-evident history.

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from provenance import ProvenanceChain

logger = logging.getLogger("eco_ai.simulation_vc")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SimulationSnapshot:
    """A single versioned simulation state."""
    snapshot_id:  str
    parent_id:    Optional[str]
    branch:       str
    author:       str
    message:      str
    timestamp:    str
    state:        dict                 # full simulation state
    tags:         List[str] = field(default_factory=list)
    cos_score:    Optional[float] = None
    metadata:     dict = field(default_factory=dict)

    def short_id(self) -> str:
        return self.snapshot_id[:8]

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "parent_id":   self.parent_id,
            "branch":      self.branch,
            "author":      self.author,
            "message":     self.message,
            "timestamp":   self.timestamp,
            "tags":        self.tags,
            "cos_score":   self.cos_score,
            "metadata":    self.metadata,
            # State is omitted here for compact log display
        }


@dataclass
class DiffResult:
    """Result of comparing two simulation versions."""
    from_id:    str
    to_id:      str
    added:      Dict[str, Any]      # keys in 'to' not in 'from'
    removed:    Dict[str, Any]      # keys in 'from' not in 'to'
    changed:    Dict[str, tuple]    # key → (old_value, new_value)
    unchanged:  List[str]

    def summary(self) -> str:
        return (
            f"Diff {self.from_id[:8]}…{self.to_id[:8]} | "
            f"+{len(self.added)} added | "
            f"-{len(self.removed)} removed | "
            f"~{len(self.changed)} changed | "
            f"={len(self.unchanged)} unchanged"
        )


# ---------------------------------------------------------------------------
# Version control system
# ---------------------------------------------------------------------------

class SimulationVersionControl:
    """
    Git-inspired version control for Eco AI Data Center simulation states.

    Usage
    -----
    >>> vc = SimulationVersionControl(author="analyst")
    >>> snap_id = vc.commit({"pue": 1.5, "cos": 0.85}, "Initial baseline")
    >>> vc.branch("experiment-cooling")
    >>> snap_id2 = vc.commit({"pue": 1.3, "cos": 0.91}, "Improved cooling config")
    >>> diff = vc.diff(snap_id, snap_id2)
    >>> print(diff.summary())
    >>> vc.log()
    """

    def __init__(
        self,
        default_author: str = "system",
        provenance: Optional[ProvenanceChain] = None,
    ):
        self._snapshots:    Dict[str, SimulationSnapshot] = {}
        self._branches:     Dict[str, str] = {"main": ""}     # branch → HEAD snapshot_id
        self._current_branch: str = "main"
        self._tags:         Dict[str, str] = {}               # tag → snapshot_id
        self.default_author = default_author
        self.provenance = provenance or ProvenanceChain(model_id="simulation-vc")
        logger.info("[SimVC] Initialised | default_author=%s", default_author)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def commit(
        self,
        state: dict,
        message: str,
        author: Optional[str] = None,
        cos_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save current simulation state as a new snapshot.

        Parameters
        ----------
        state    : dict — full simulation state to snapshot
        message  : commit message
        author   : who made this commit (defaults to default_author)
        cos_score: optional CATERYA COS score for this snapshot
        metadata : arbitrary metadata

        Returns
        -------
        str : snapshot_id (SHA-256 of content)
        """
        author = author or self.default_author
        parent_id = self._branches.get(self._current_branch) or None

        # Content-address: hash of parent + state + timestamp
        ts = datetime.now(timezone.utc).isoformat()
        content = json.dumps({
            "parent_id": parent_id,
            "state":     state,
            "message":   message,
            "author":    author,
            "ts":        ts,
        }, sort_keys=True, default=str)
        snapshot_id = hashlib.sha256(content.encode()).hexdigest()

        snap = SimulationSnapshot(
            snapshot_id=snapshot_id,
            parent_id=parent_id,
            branch=self._current_branch,
            author=author,
            message=message,
            timestamp=ts,
            state=dict(state),
            cos_score=cos_score,
            metadata=metadata or {},
        )

        self._snapshots[snapshot_id] = snap
        self._branches[self._current_branch] = snapshot_id

        self.provenance.record("sim_commit", {
            "snapshot_id": snapshot_id[:16],
            "branch":      self._current_branch,
            "author":      author,
            "message":     message,
            "cos_score":   cos_score,
        })

        logger.info(
            "[SimVC] Committed %s on '%s' by %s: %s",
            snapshot_id[:8], self._current_branch, author, message,
        )
        return snapshot_id

    def checkout(self, snapshot_id: str) -> SimulationSnapshot:
        """
        Restore a previous simulation state.
        Returns the snapshot (does NOT modify current branch HEAD).
        """
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            # Try short ID prefix match
            matches = [s for s in self._snapshots if s.startswith(snapshot_id)]
            if len(matches) == 1:
                snap = self._snapshots[matches[0]]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous short ID '{snapshot_id}': {[m[:8] for m in matches]}")
            else:
                raise KeyError(f"Snapshot not found: '{snapshot_id}'")

        logger.info("[SimVC] Checked out %s (%s)", snap.short_id(), snap.message)
        self.provenance.record("sim_checkout", {"snapshot_id": snap.snapshot_id[:16]})
        return snap

    def branch(self, branch_name: str, from_snapshot_id: Optional[str] = None) -> str:
        """
        Create a new branch.
        If from_snapshot_id is None, branches from current HEAD.
        Returns branch HEAD snapshot_id.
        """
        if from_snapshot_id:
            head = from_snapshot_id
        else:
            head = self._branches.get(self._current_branch, "")

        self._branches[branch_name] = head
        self._current_branch = branch_name
        logger.info("[SimVC] Created branch '%s' from %s", branch_name, head[:8] if head else "empty")
        self.provenance.record("sim_branch", {"branch": branch_name, "from": head[:16] if head else ""})
        return head

    def switch_branch(self, branch_name: str) -> None:
        """Switch to an existing branch."""
        if branch_name not in self._branches:
            raise KeyError(f"Branch '{branch_name}' not found.")
        self._current_branch = branch_name
        logger.info("[SimVC] Switched to branch '%s'", branch_name)

    def merge(
        self,
        source_branch: str,
        message: Optional[str] = None,
        author: Optional[str] = None,
    ) -> str:
        """
        Merge source_branch HEAD into current branch.
        Creates a merge commit that combines both states.
        """
        if source_branch not in self._branches:
            raise KeyError(f"Branch '{source_branch}' not found.")

        source_id = self._branches[source_branch]
        target_id = self._branches.get(self._current_branch, "")

        if not source_id:
            raise ValueError(f"Source branch '{source_branch}' has no commits.")

        source_snap = self._snapshots.get(source_id)
        target_snap = self._snapshots.get(target_id) if target_id else None

        # Merged state: target overlaid with source (source wins on conflict)
        merged_state = {}
        if target_snap:
            merged_state.update(target_snap.state)
        if source_snap:
            merged_state.update(source_snap.state)

        msg = message or f"Merge '{source_branch}' into '{self._current_branch}'"
        snap_id = self.commit(
            merged_state, msg,
            author=author,
            metadata={"merge_from": source_branch, "merge_source_id": source_id[:16]},
        )
        logger.info("[SimVC] Merged '%s' into '%s'", source_branch, self._current_branch)
        return snap_id

    def tag(self, label: str, snapshot_id: Optional[str] = None) -> None:
        """Tag a snapshot with a human-readable label."""
        sid = snapshot_id or self._branches.get(self._current_branch, "")
        if not sid:
            raise ValueError("Nothing to tag — no commits on current branch.")
        self._tags[label] = sid
        snap = self._snapshots.get(sid)
        if snap:
            snap.tags.append(label)
        logger.info("[SimVC] Tagged %s as '%s'", sid[:8], label)

    # ------------------------------------------------------------------
    # History and diff
    # ------------------------------------------------------------------

    def log(
        self,
        branch: Optional[str] = None,
        n: int = 20,
    ) -> List[SimulationSnapshot]:
        """Return commit history for a branch (most recent first)."""
        branch = branch or self._current_branch
        head_id = self._branches.get(branch, "")
        if not head_id:
            return []

        history = []
        current_id = head_id
        while current_id and len(history) < n:
            snap = self._snapshots.get(current_id)
            if snap is None:
                break
            history.append(snap)
            current_id = snap.parent_id or ""

        return history

    def diff(self, from_id: str, to_id: str) -> DiffResult:
        """Compare two simulation snapshots."""
        from_snap = self.checkout(from_id)
        to_snap   = self.checkout(to_id)

        from_state = self._flatten(from_snap.state)
        to_state   = self._flatten(to_snap.state)

        all_keys = set(from_state) | set(to_state)
        added    = {k: to_state[k]   for k in all_keys if k not in from_state}
        removed  = {k: from_state[k] for k in all_keys if k not in to_state}
        changed  = {
            k: (from_state[k], to_state[k])
            for k in all_keys
            if k in from_state and k in to_state and from_state[k] != to_state[k]
        }
        unchanged = [k for k in all_keys if k in from_state and k in to_state and from_state[k] == to_state[k]]

        return DiffResult(
            from_id=from_snap.snapshot_id,
            to_id=to_snap.snapshot_id,
            added=added,
            removed=removed,
            changed=changed,
            unchanged=unchanged,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_bundle(self) -> dict:
        """Export full VC history as a portable JSON bundle."""
        return {
            "exported_at":      datetime.now(timezone.utc).isoformat(),
            "current_branch":   self._current_branch,
            "branches":         {k: v[:16] if v else "" for k, v in self._branches.items()},
            "tags":             {k: v[:16] for k, v in self._tags.items()},
            "total_snapshots":  len(self._snapshots),
            "snapshots":        [s.to_dict() for s in self._snapshots.values()],
        }

    def export_bundle_json(self) -> str:
        return json.dumps(self.export_bundle(), indent=2, default=str)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_branch(self) -> str:
        return self._current_branch

    @property
    def head(self) -> Optional[SimulationSnapshot]:
        head_id = self._branches.get(self._current_branch, "")
        return self._snapshots.get(head_id) if head_id else None

    @property
    def branches(self) -> List[str]:
        return list(self._branches.keys())

    @property
    def all_tags(self) -> Dict[str, str]:
        return dict(self._tags)

    @property
    def total_commits(self) -> int:
        return len(self._snapshots)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict:
        """Flatten nested dict to dot-notation keys."""
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(SimulationVersionControl._flatten(v, key))
            else:
                result[key] = v
        return result
