# core/scr.py
from datetime import datetime
from typing import Dict, Iterable

from .match_state import MatchState
from .mri import MRI


class SCR:
    """
    State Commit & Recovery:
    - 基于策略对 MatchState 做 commit / deactivate / reactivate
    - 不直接做相似度计算，只看状态里的 evidence + 时间信息
    """

    def __init__(self, score_key: str = "score", commit_threshold: float = 0.9,
                 idle_seconds: int = 3600):
        self.score_key = score_key
        self.commit_threshold = commit_threshold
        self.idle_seconds = idle_seconds

    # ---------- Commit 策略 ----------
    def commit_by_threshold(self, mri: MRI) -> None:
        """
        如果某条 state 的 evidence[score_key] >= 阈值，则标记为 committed。
        """
        for state in mri.iter_active_states():
            score = state.evidence.get(self.score_key, None)
            if score is not None and score >= self.commit_threshold:
                state.commit()

    # ---------- Deactivate 策略 ----------
    def deactivate_idle(self, mri: MRI) -> None:
        """
        如果一个 state 长时间未更新（例如 > idle_seconds），则 deactivate。
        """
        now = datetime.now()
        for state in mri.iter_active_states():
            try:
                updated_at = datetime.fromisoformat(state.updated_at)
            except ValueError:
                continue
            if (now - updated_at).total_seconds() > self.idle_seconds:
                state.deactivate()

    # ---------- Reactivate ----------
    def reactivate_states(self, states: Iterable[MatchState]) -> None:
        """
        按照调用者的选择把部分 state 从 deactivated 变回 active。
        什么时候重激活由上层策略决定（比如：新模型上线，需要重新打分）。
        """
        for state in states:
            if state.state == "deactivated":
                state.set_state("active")
