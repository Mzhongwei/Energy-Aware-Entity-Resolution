import os
from igraph import Graph
from datetime import datetime
from utils.utils import *
from match_gouvernance.match_state import MatchStateRegistry
state_registry = MatchStateRegistry()

class SimilarityGraph:
    def __init__(self, most_similar_num, output_format):
        self.graph = Graph(directed=False)  # directed
        self.group_map = {}  # record name -> group name
        self.vertex_map = {}  # group name -> vertex id
        self.most_similar_num = most_similar_num
        self.output_format = output_format
        self.name = ""
        self.app_logger = None
    
    def set_logger(self, logger):
        self.app_logger = logger

    def _find_or_create_group(self, name):
        group = self.group_map.get(name, name)
        try:
            self.graph.vs.find(name=group)
        except ValueError:
            self.graph.add_vertex(name=group)
        return group


    def _merge_groups(self, a, b):
        if a == b:
            return
        for k, v in self.group_map.items():
            if v == b:
                self.group_map[k] = a
        self.group_map[b] = a

        try:
            v_b = self.graph.vs.find(name=b)
            v_a = self.graph.vs.find(name=a)
        except ValueError:
            return

        id_b = v_b.index
        id_a = v_a.index
        neighbors = self.graph.neighbors(id_b, mode="ALL")

        for nid in neighbors:
            if nid == id_a:
                continue
            other = self.graph.vs[nid]["name"]
            weight = self.graph.es[self.graph.get_eid(id_b, nid)]["weight"]
            self._add_edge(a, other, weight)

        self.graph.delete_vertices(id_b)


    def _add_edge(self, group1, group2, weight):
        v1 = self.graph.vs.find(name=group1)
        v2 = self.graph.vs.find(name=group2)

        eid_1_2 = self.graph.get_eid(v1.index, v2.index, error=False)
        if eid_1_2 == -1:
            self.graph.add_edge(v1.index, v2.index, weight=weight)
        else:
            if self.graph.es[eid_1_2]["weight"] < weight:
                self.graph.es[eid_1_2]["weight"] = weight

        # eid_2_1 = self.graph.get_eid(v2.index, v1.index, error=False)
        # if eid_2_1 == -1:
        #     self.graph.add_edge(v2.index, v1.index, weight=weight)
        # else:
        #     if self.graph.es[eid_2_1]["weight"] < weight:
        #         self.graph.es[eid_2_1]["weight"] = weight

    def add_similarity(self, record, new_similarities):
        self.group_map.setdefault(record, record)
        g1 = self._find_or_create_group(record)

        for other, score in new_similarities:

            # 1) match state version logging
            state = state_registry.get_or_create(record, other)
            if len(state.history()) == 0:
                state.generate(score=score, stage="PM")
            else:
                state.update(score=score, stage="CM")

            # 2) UPDATE Similarity graph
            self.group_map.setdefault(other, other)
            g2 = self._find_or_create_group(other)

            if score == 1.0:
                self._merge_groups(g1, g2)
                g1 = self.group_map[record]
            else:
                self._add_edge(g1, g2, score)

            self._limit_edges(g2)

        self._limit_edges(g1)


    def _limit_edges(self, group):
        try:
            v = self.graph.vs.find(name=group)
        except ValueError:
            return
        edges = self.graph.incident(v.index, mode="OUT")
        if len(edges) > self.most_similar_num:
            weighted_edges = [(e, self.graph.es[e]["weight"]) for e in edges]
            to_delete = sorted(weighted_edges, key=lambda x: x[1], reverse=True)[self.most_similar_num:]
            self.graph.delete_edges([e for e, _ in to_delete])


    def get_group_members(self, group_name):
        return [k for k, v in self.group_map.items() if v == group_name]

    def add_vertex_members(self):
        for v in self.graph.vs:
            group = v["name"]
            members = self.get_group_members(group)
            v["members"] = ",".join(members)


    def display(self):
        for v in self.graph.vs:
            name = v["name"]
            members = self.get_group_members(name)
            neighbors = self.graph.neighbors(v.index, mode="OUT")
            neighbor_names = [self.graph.vs[n]["name"] for n in neighbors]
            print(f"Group {name}: members={members}, connected to={neighbor_names}")

    
    def check_output_path(self, name):
        self.name = name

    def update_file(self):
        filepath = ""
        if self.output_format == "graphml":
            filepath = f"pipeline/similarity/{self.name}.graphml"
            self.export_graphml(f"pipeline/similarity/{self.name}.graphml")
        return filepath

    def export_graphml(self, path):
        """
        Export graph to GraphML format (with weights)
        """
        self.add_vertex_members()
        self.graph.write_graphml(path)
        print(OUTPUT_FORMAT.format(f'[OK] Graph exported to GraphML: {path}.', datetime.now().strftime(TIME_FORMAT)))
        

    def export_edgelist(self, path):
        """
        Export edge list: node1 node2 weight
        """
        with open(path, "w") as f:
            for e in self.graph.es:
                source = self.graph.vs[e.source]["name"]
                target = self.graph.vs[e.target]["name"]
                weight = e["weight"]
                f.write(f"{source} {target} {weight}\n")
        print(f"[OK] Graph exported to edge list: {path}")

    def export_pickle(self, path):
        """
        Save entire igraph.Graph object as pickle
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"[OK] Graph saved as pickle: {path}")


if __name__ == '__main__':
    sim = SimilarityGraph(3, 'graphml')
    sim.add_similarity("apple", [("banana", 0.8), ("cherry", 0.9), ("grape", 0.85)])
    sim.add_similarity("apple", [("orange", 0.7), ("pear", 0.95), ("watermelon", 0.92)])
    sim.add_similarity("apple", [("pomme", 1)])

    print(sim.display())  