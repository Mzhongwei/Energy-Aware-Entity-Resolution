import numpy as np
import math

class NodeSampler:
    '''Summary:
    This is a sampler for vertex in graph. If the number of neighbors is above certain threshold (50 bu default), 
    sampler will automatically chooses the aliased method to simplifier the random sample. 
    If not, simple sample method will be applied by np structure

    Attributes:
        threshold(int): threshold for sampler selection
        n(int): number of neighbors
        method(str): sample method. "simple-s"--> no weight, n<=threshold; "simple-s"--> no weight, n>threshold; 
                                    "weighted"--> weight, n<=threshold; "alias"--> weight, n>threshold
        neighbors(np array): list of neighbor indexes for a vertex
        is_first(np array): list of boolean values which describe weather the neighbor corresponding can be the first node of a sentence
                            *exists only when the vertex can not be the first node*
        weights(np array): list of weights corresponding to each neighbor in the list neighbors
                            *exists when the edges are weighted*
        alias_prob(np zeros): alias probability after normalization
                            *exists only for alias method*
        alias_idx(np zeros): alias id
                            *exists only for alias method*
    '''
    def __init__(self, neighbors, weighted, threshold=50, weights=None):
        self.threshold = threshold
        if weighted:
            self._build_weighted(neighbors, weights)
        else:
            self._build_simple(neighbors)

    def _build_weighted(self, neighbors, weights):
        self.neighbors = np.array(neighbors, dtype=int)
        self.weights = np.array(weights, dtype=float)
        self.n = len(self.neighbors)

        if self.n <= self.threshold:
            self.method = "weighted"
        else:
            self.method = "alias"
            self._build_alias_table()
    
    def _build_simple(self, neighbors):
        self.neighbors = np.array(neighbors, dtype=int)
        self.n = len(self.neighbors)
        if self.n <= self.threshold:
            self.method = "simple-s"
        else:
            self.method = "simple-l"

    def _build_alias_table(self):
        norm_w = self.weights * self.n / self.weights.sum()
        self.alias_prob = np.zeros(self.n)
        self.alias_idx = np.zeros(self.n, dtype=int)

        small, large = [], []
        for i, w in enumerate(norm_w):
            (small if w < 1 else large).append(i)

        while small and large:
            s, l = small.pop(), large.pop()
            self.alias_prob[s] = norm_w[s]
            self.alias_idx[s] = l
            norm_w[l] -= 1 - norm_w[s]
            (small if norm_w[l] < 1 else large).append(l)

        for i in small + large:
            self.alias_prob[i] = 1
            self.alias_idx[i] = i

    def sample(self):
        # perform a random sample
        if self.n == 0:
            return None  # no neighbors

        if self.method == "simple-s":
            return np.random.choice(self.neighbors)
        elif self.method == "simple-l":
            i = np.random.randint(self.n)
            return self.neighbors[i]
        elif self.method == "weighted":
            return np.random.choice(self.neighbors, p=self.weights / self.weights.sum())
        elif self.method == "alias":
            i = np.random.randint(self.n)
            if np.random.rand() < self.alias_prob[i]:
                return self.neighbors[i]
            else:
                return self.neighbors[self.alias_idx[i]]
    
    def sample_firstnode(self):
        candidate_indices = self.neighbors[self.is_first]
        if self.method == "simple-s" or self.method == "simple-l":
            return np.random.choice(candidate_indices)
        else:
            candidate_weights = self.weights[self.is_first]
            return np.random.choice(candidate_indices, p=candidate_weights / candidate_weights.sum())

    def update(self, neighbors, weights):
        self._build_weighted(neighbors, weights)
    
    def update_firstnode_list(self, isfirst):
        self.is_first = np.array(isfirst, dtype=bool)
