# similaritygraph.py
import networkx as nx
from typing import Dict, Tuple, Any, Iterable

from core.match_state import MatchState  # ← 你的 MatchState

RecordId = str
Pair = Tuple[RecordId, RecordId]


class SimilarityGraph:
    """
    A graph where nodes are records/entities and edges represent candidate
    relationships enriched with MatchState objects. Supports lightweight
    integration and incremental updates of similarity evidence.
    """

    def __init__(self):
        self.graph = nx.Graph()

    # -------------------------------------------------------
    # Node and Edge creation
    # -------------------------------------------------------
    def add_pair(self, rid1: RecordId, rid2: RecordId, score: float = None) -> None:
        """
        Add a pair (rid1, rid2) as an edge with optional score.
        Also create a MatchState for this pair if not existing.
        """

        # ensure nodes
        if not self.graph.has_node(rid1):
            self.graph.add_node(rid1)
        if not self.graph.has_node(rid2):
            self.graph.add_node(rid2)

        # if edge exists: do not overwrite MatchState
        if self.graph.has_edge(rid1, rid2):
            if score is not None:
                self.graph[rid1][rid2]["weight"] = score
                st = self.graph[rid1][rid2].get("state")
                if st is not None:
                    st.update_evidence("score", score)
            return

        # create new match state
        state = MatchState(rid1, rid2)

        # add edge with optional score and attached state
        edge_attr = {"state": state}
        if score is not None:
            edge_attr["weight"] = score
            state.update_evidence("score", score)

        self.graph.add_edge(rid1, rid2, **edge_attr)

    # -------------------------------------------------------
    # Updating similarity (pairwise matching output)
    # -------------------------------------------------------
    def update_weight(self, rid1: RecordId, rid2: RecordId, score: float) -> None:
        """
        Update weight and store evidence in MatchState.
        """

        if not self.graph.has_edge(rid1, rid2):
            # auto-create if missing, to avoid pipeline break
            self.add_pair(rid1, rid2, score)
            return

        # 1) update score
        self.graph[rid1][rid2]["weight"] = score

        # 2) update MatchState evidence
        st = self.graph[rid1][rid2].get("state")
        if st is not None:
            st.update_evidence("score", score)

    # -------------------------------------------------------
    # Arbitrary evidence update
    # -------------------------------------------------------
    def update_state(self, rid1: RecordId, rid2: RecordId, key: str, value: Any) -> None:
        """
        Add or update evidence on the MatchState for the edge (rid1, rid2).
        Example:
            update_state("A", "B", "jaccard", 0.72)
            update_state("A", "B", "llm_score", 0.91)
        """
        if not self.graph.has_edge(rid1, rid2):
            # auto-create edge if state doesn't exist
            self.add_pair(rid1, rid2)

        st = self.graph[rid1][rid2].get("state")
        if st is not None:
            st.update_evidence(key, value)

    # -------------------------------------------------------
    # Access methods
    # -------------------------------------------------------
    def get_match_state(self, rid1: RecordId, rid2: RecordId) -> MatchState | None:
        if self.graph.has_edge(rid1, rid2):
            return self.graph[rid1][rid2].get("state")
        return None

    def get_all_states(self) -> Iterable[MatchState]:
        """
        Iterator over all MatchState objects in the graph.
        """
        for _, _, data in self.graph.edges(data=True):
            st = data.get("state")
            if st is not None:
                yield st

    # -------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------
    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()

    def __repr__(self):
        return f"SimilarityGraph(nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"
