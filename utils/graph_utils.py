import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import networkx as nx
random.seed(123)
np.random.seed(123)


class Graph:
    ''' graph class '''
    def __init__(self, nodes, edges, children, parents): 
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight

def load_graph(graph_file):
    G = nx.DiGraph()
    with open(graph_file, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(int(u), int(v))  
    return G


def read_graph(path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not(directed):
                # regard as undirectional
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    # change the probability to 1/indegree
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
            
    return Graph(nodes, edges, children, parents)





