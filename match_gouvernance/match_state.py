from datetime import datetime
from typing import Tuple, List, Optional, Dict


class MatchStateVersion:
    def __init__(self, version:int, timestamp:str, score:float, stage:str, decision:str):
        self.version = version
        self.timestamp = timestamp
        self.score = score
        self.stage = stage
        self.decision = decision

    def __repr__(self):
        return f"(v={self.version}, score={self.score}, stage={self.stage}, decision={self.decision}, ts={self.timestamp})"


class MatchState:
    def __init__(self, rid_i:str, rid_j:str):
        self.pair_id = (rid_i, rid_j)
        self.versions: List[MatchStateVersion] = []
        self.latest_version = 0

    def generate(self, score:float, stage:str, decision:str="tentative"):
        self.latest_version = 1
        self.versions.append(MatchStateVersion(
            version=1,
            timestamp=datetime.now().isoformat(),
            score=score, stage=stage, decision=decision
        ))

    def update(self, score=None, stage=None, decision=None):
        prev = self.versions[-1]
        self.latest_version += 1
        self.versions.append(MatchStateVersion(
            version=self.latest_version,
            timestamp=datetime.now().isoformat(),
            score=score if score is not None else prev.score,
            stage=stage if stage is not None else prev.stage,
            decision=decision if decision is not None else prev.decision
        ))

    def latest(self):
        return self.versions[-1]

    def history(self):
        return self.versions


class MatchStateRegistry:
    def __init__(self):
        self._store: Dict[str, MatchState] = {}

    def _key(self, i:str, j:str):
        return f"{i}::{j}"

    def get_or_create(self, i:str, j:str):
        k = self._key(i, j)
        if k not in self._store:
            self._store[k] = MatchState(i, j)
        return self._store[k]

    def get(self, i:str, j:str):
        return self._store[self._key(i, j)]
