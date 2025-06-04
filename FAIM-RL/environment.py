import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import copy
import utils.graph_utils as graph_utils
from collections import deque,Counter
random.seed(123)
np.random.seed(123)


class Environment:
    ''' environment that the agents run in '''
    def __init__(self, name, graphs,graph_realizations, budget, training=True):
        '''
            method: 'RR' or 'MC'
            use_cache: use cache to speed up
        '''
        self.name = name
        self.graphs = graphs
        self.graph_realizations=graph_realizations
        # IM
        self.budget = budget
        self.training = training 

    def reset_graphs(self, num_graphs=10):
        # generate new graph
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        ''' restart '''
        if idx is None:
            idx = random.randint(0, 4)
            self.graph =  copy.deepcopy(self.graphs[idx])
            self.graph_realization=self.graph_realizations[idx]
            self.idx=idx
        else:
            self.idx=idx
            self.graph = copy.deepcopy(self.graphs[idx])
            self.graph_realization=self.graph_realizations[idx]
        self.action = [i for i in range(self.graph.num_nodes)]
        self.state = [0 for _ in range(self.graph.num_nodes)]
        # IM
        self.prev_inf = 0 # previous influence score
        self.states = []
        self.actions_set = set()
        self.visited=set()
        self.actions=[]
        self.rewards = []
        self.store_graph=[]
        self.training = training

    
    def bfs_multi_source_with_mask(self,G, seeds):
        visited = set()
        queue = deque()

        for s in seeds:
            visited.add(s)
            queue.append(s)

        while queue:
            node = queue.popleft()
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited
    
    def compute_reward_realization(self, S):
        G = self.graph_realization
        visited = set()
        queue = deque()
        visited.add(S)
        queue.append(S)
        while queue:
            current = queue.popleft()
            for neighbor in G.successors(current):  # 只走出边
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        reward = len(visited)
        if not self.visited:
            before=0
        else:
            before=len(self.visited)
        self.visited.update(visited)
        reward=len(self.visited)-before
        self.rewards.append(reward)
        return reward,visited

    def step(self, action, time_reward=None):
        ''' change state and get reward '''
        # node has already been selected
        # store state and action
        self.store_graph.append(copy.deepcopy(self.graph))
        reward ,visited= self.compute_reward_realization(action)
        self.rewards.append(reward)
        self.graph.edges = {
            edge: weight for edge, weight in self.graph.edges.items()
            if edge[0] not in visited
        }
        self.actions_set.add(action)
        self.actions.append(action)
