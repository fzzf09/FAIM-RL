import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import  scatter_add, scatter_softmax
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
import utils.graph_utils as graph_utils
from collections import deque
from tqdm import tqdm
from torch import optim
from torch_geometric.nn import GCNConv
import dataprocess
import world

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15

class GCN_MLP_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_mlp_layers=5):
        super(GCN_MLP_Model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # MLP Layers
        mlp_layers = []
        mlp_in = hidden_channels
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(mlp_in, hidden_channels))
            mlp_in = hidden_channels
        self.mlp = nn.ModuleList(mlp_layers)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x= self.conv3(x, edge_index)
        x=  F.relu(x)

        for layer in self.mlp:
            x = layer(x)
            x = F.relu(x)
        x = self.final_layer(x)
        if data.y is not None:
            return x[data.y]
        else:
            return x


def UniformSample_original_python(dataset):
    dataset : dataprocess.BasicDataset
    users = dataset.users
    allPos = dataset.allPos #self._allPos = [array([0, 2]), array([2, 3]), array([0, 1, 4])]表示用户0正向交互物品0，2
    S = []
    for i, user in enumerate(users):#每次遍历一个用户
        posForUser = allPos[i]#得该改用户影响到的节点
        if len(posForUser) == 0:
            continue
        while True:
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex] #从被影响到的节点中随机选一个
            if positem in  users:
                continue
            else:
                break
        while True:
            negitem = np.random.randint(0, dataset.m_items) #随机选择一个不被影响的节点
            if negitem in posForUser or negitem in users:
                continue
            else:
                break
        S.append([user, positem, negitem])#每个用户有u,被影响节点，没影响节点进入S
    return np.array(S)



def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    #print(batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)



def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    
    S = UniformSample_original_python(dataset) #每个epoch，为每个用户生成一个u,node,未影响node
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    #users, posItems, negItems = utils.shuffle(users, posItems, negItems) #对索引打乱
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(minibatch(users, #比如我们有4096个用户，batch size是2048,每次只取2048个节点，分两次
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return f"loss{aver_loss:.3f}"


def get_init_node_embed(graph, device):
    dataset = dataprocess.SpaGraphDataset(graph, k=10)
    lightmodel = LightGCN(world.config, dataset)
    bpr = BPRLoss(lightmodel, world.config)
    for epoch in range(world.config['train_epochs']):
        output_information = BPR_train_original(dataset, lightmodel, bpr, epoch)
        if epoch%20==0:
            print(output_information)
    lightmodel.eval()

    users_emb, items_emb = lightmodel.computer()
    num_users = len(dataset.users)  
    num_items = len(dataset.items) 

    embedding_dim = users_emb.size(1)  
    max_index = num_users + num_items  
    merged_emb = torch.zeros((max_index, embedding_dim), device=users_emb.device)  # 保持张量在相同的设备上

    for idx, user_id in enumerate(dataset.users):
        merged_emb[user_id] = users_emb[idx]
    for idx, item_id in enumerate(dataset.items):
        merged_emb[item_id] = items_emb[idx]

    return merged_emb



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:dataprocess.BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataprocess.BasicDataset = dataset #应该是一个继承了 BasicDataset 的实例，包含用户和物品的信息、图的邻接矩阵等。
        self.__init_weight()

    def __init_weight(self):
        device_id = 7
        self.device=torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.item_list= self.dataset.items
        self.latent_dim = self.config['latent_dim_rec'] #每个用户和物品嵌入的维度
        self.n_layers = self.config['lightGCN_n_layers'] #LightGCN 的层数
        self.keep_prob = self.config['keep_prob'] #节点 dropout 的保留概率
        self.A_split = self.config['A_split'] #表示邻接矩阵是否分块存储（用于大图优化计算）
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.device) # 使用 PyTorch 的 Embedding 模块初始化嵌入矩阵
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.device) 
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('use NORMAL distribution initilizer')  #如果没有预训练，使用正态分布 (N(0, 0.1)) 初始化用户和物品嵌入权重。
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph() #Graph 是从数据集中加载的稀疏图邻接矩阵
        print(f"lgn is already to go(dropout:{self.config['dropout']})")


    def __dropout_x(self, x, keep_prob): #随机丢弃部分图的边，起到正则化和数据增强的作用 x就是图邻接矩阵 这个可能用不上
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)#邻居聚合公式
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1) #聚合每一层的嵌入
        light_out = torch.mean(embs, dim=1)#取平均值
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users
        tmp_pos_emb = []
        tmp_neg_emb = []
        tmp_pos_emb_ego = []
        tmp_neg_emb_ego = []

        for i in range(len(pos_items)):
            index_pos = self.item_list.index(pos_items[i])
            index_neg = self.item_list.index(neg_items[i])
            tmp_pos_emb.append(all_items[index_pos])
            tmp_neg_emb.append(all_items[index_neg])
            tmp_pos_emb_ego.append(self.embedding_item(torch.tensor(index_pos, device=self.embedding_item.weight.device)))
            tmp_neg_emb_ego.append(self.embedding_item(torch.tensor(index_neg, device=self.embedding_item.weight.device)))

        pos_emb = torch.stack(tmp_pos_emb)
        neg_emb = torch.stack(tmp_neg_emb)
        pos_emb_ego = torch.stack(tmp_pos_emb_ego)
        neg_emb_ego = torch.stack(tmp_neg_emb_ego)

        users_emb_ego = self.embedding_user.weight
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    
    def bpr_loss(self, users, pos, neg):#待修改  users: 用户(种子节点)索引。pos: 正样本（用户实际交互过的物品，被种子节点影响的节点）索引。neg: 负样本（用户未交互过的物品，未被种子节点影响的节点）索引。
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss #正则化损失，用于防止过拟合。
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


