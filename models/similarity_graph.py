from igraph import Graph
from datetime import datetime
from utils.utils import OUTPUT_FORMAT, TIME_FORMAT

class SimilarityGraph:
    def __init__(self, most_similar_num, output_format):
        self.graph = Graph(directed=False)  # directed
        self.group_map = {}  # record name -> group name
        self.vertex_map = {}  # group name -> vertex id
        self.most_similar_num = most_similar_num
        self.output_format = output_format

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


    def export_graphml(self, path):
        """
        Export graph to GraphML format (with weights)
        """
        self.add_vertex_members()
        self.graph.write_graphml(path)
        print(OUTPUT_FORMAT.format(f'[OK] Graph exported to GraphML: {path}.', datetime.now().strftime(TIME_FORMAT)))
