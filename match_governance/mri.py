# core/mri.py
from typing import Dict, Iterable, Tuple, Any
import networkx as nx

from .match_state import MatchState

RecordId = str
Pair = Tuple[RecordId, RecordId]


class MRI:
    """
    Match Runtime Integration:
    - 管理一个图 G，节点是记录，边上挂 MatchState
    - 对外提供 create / update / 查询 等接口
    """
    def __init__(self):
        self.G = nx.Graph()
        # 可选：字典索引方便快速访问
        self.states: Dict[Pair, MatchState] = {}

    @staticmethod
    def _normalize_pair(i: RecordId, j: RecordId) -> Pair:
        return tuple(sorted((i, j)))

    # -------- 创建阶段：来自 Candidate Generation --------
    def create_states_from_pairs(self, pairs: Iterable[Pair]) -> None:
        """
        根据候选对创建 MatchState，并在图中加边。
        """
        for i, j in pairs:
            key = self._normalize_pair(i, j)
            if key in self.states:
                continue

            state = MatchState(i, j)
            self.states[key] = state

            # 确保节点存在
            if not self.G.has_node(i):
                self.G.add_node(i)
            if not self.G.has_node(j):
                self.G.add_node(j)

            # 边上挂属性 state=MatchState
            self.G.add_edge(i, j, match_state=state)

    # -------- 更新阶段：来自 Pairwise / Collective Matching --------
    def update_scores(self, scores: Dict[Pair, float], key: str = "score") -> None:
        """
        scores: {(i,j): score}
        key: 存在 MatchState.evidence 里的字段名，如 "levenshtein" 或 "llm_score"
        """
        for (i, j), value in scores.items():
            pair = self._normalize_pair(i, j)
            state = self.states.get(pair)
            if state is None:
                # 如果还没创建（理论上不该发生），可以选择自动创建
                state = MatchState(i, j)
                self.states[pair] = state
                if not self.G.has_node(i):
                    self.G.add_node(i)
                if not self.G.has_node(j):
                    self.G.add_node(j)
                self.G.add_edge(i, j, match_state=state)
            state.update_evidence(key, value)

    def get_state(self, i: RecordId, j: RecordId) -> MatchState | None:
        return self.states.get(self._normalize_pair(i, j))

    def iter_active_states(self):
        for state in self.states.values():
            if state.is_active():
                yield state

    def iter_all_states(self):
        return self.states.values()
