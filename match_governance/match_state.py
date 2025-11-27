# core/match_state.py

from datetime import datetime
from typing import Dict, Any, Optional, Tuple

class MatchState:
    """
    A lifecycle-aware representation of a record pair (i, j).
    Tracks evidence, decisions, timestamps, and governance status.
    """
    def __init__(self, rid_i: str, rid_j: str):
        self.pair: Tuple[str, str] = (rid_i, rid_j)

        # timestamps
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = self.created_at

        # evidence store: {"levenshtein": 0.83, "jaccard": 0.77, ...}
        self.evidence: Dict[str, Any] = {}

        # decision state: undecided / tentative / match / nonmatch / committed / deactivated
        self.state: str = "active"

    def update_evidence(self, key: str, value: Any):
        self.evidence[key] = value
        self.updated_at = datetime.now().isoformat()

    def set_state(self, new_state: str):
        self.state = new_state
        self.updated_at = datetime.now().isoformat()

    def is_committed(self) -> bool:
        return self.state == "committed"

    def is_active(self) -> bool:
        return self.state == "active"

    def deactivate(self):
        self.state = "deactivated"
        self.updated_at = datetime.now().isoformat()

    def commit(self):
        self.state = "committed"
        self.updated_at = datetime.now().isoformat()

    def __repr__(self):
        return f"MatchState[{self.pair}] state={self.state}, evidence={self.evidence}"
