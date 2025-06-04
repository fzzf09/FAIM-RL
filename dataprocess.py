import numpy as np
import scipy.sparse as sp
import torch
import world
import random
from collections import Counter, deque
random.seed(123)
np.random.seed(123)
class BasicDataset:
    """基本数据集基类，供继承使用"""
    def __init__(self):
        pass

class SpaGraphDataset(BasicDataset):
    def __init__(self, graph, k=5):
        """
        使用图构建用户-物品数据集
        
        参数:
            graph: NetworkX 图实例
            k: 用户的数量 (前 k 个度数最高的节点被视为用户)
        """
        super(SpaGraphDataset, self).__init__()
        self.graph = graph
        self.k = k
        self.R = 1000
        device_id = 7
        self.device=torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        degree_dict = self.graph.degree  # 直接使用 graph 里的 degree 属性
        sorted_nodes = sorted(degree_dict, key=lambda x: x[1], reverse=True)
        self.users = [node for node, _ in sorted_nodes[:self.k]]
        self.items = [node for node, _ in sorted_nodes[self.k:]]
        
        self.n_users = len(self.users)
        self.m_items = len(self.items)
        self.Graph = None
        self.allPos=self.bfs()


    def _build_sparse_graph(self):
        edges = list(self.graph.edges.keys())  # 从字典中获取边
        num_nodes = len(self.graph.nodes)  # 使用 self.graph.nodes 获取节点数量

        row = [edge[0] for edge in edges]
        col = [edge[1] for edge in edges]
        data = [1] * len(edges)

        adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj


    def getSparseGraph(self):
        return self.adj_matrix 
    
    def bfs(self):
        influenced = []
        for node in self.users:
            visited = set()
            queue = deque()
            visited.add(node)
            queue.append(node)
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.successors(current):  # 只走出边
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            influenced.append(list(visited))
        return  influenced
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def getSparseGraph(self):
        if self.Graph is None:
                norm_adj = self._build_sparse_graph()
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph