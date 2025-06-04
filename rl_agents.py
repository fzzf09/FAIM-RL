import random
import time
import os
from collections import namedtuple, deque
import numpy as np
import models
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQAgent:
    ''' deep Q agent '''
    def __init__(self, args):
        '''
        lr: learning rate
        n_step: (s_t-n,a_t-n,r,s_t)
        '''
        self.model_name = args.model
        self.gamma = 1 # discount factor of future rewards
        self.n_step = args.n_step # num of steps to accumulate rewards

        self.training = not(args.test)
        self.T = args.T

        self.memory = ReplayMemory(args.memory_size)
        self.batch_size = args.bs # batch size for experience replay

        self.double_dqn = args.double_dqn
        self.device = args.device

        self.node_dim = 2
        self.edge_dim = 4
        self.reg_hidden = args.reg_hidden
        self.embed_dim = args.embed_dim
        self.graph_node_embed = {}

        self.model = models.GCN_MLP_Model(in_channels=64,hidden_channels=128,out_channels=1,num_mlp_layers=5).to(self.device)
        self.target = models.GCN_MLP_Model(in_channels=64,hidden_channels=128,out_channels=1,num_mlp_layers=5).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.setup_graph_input = self.setup_graph_input

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if not self.training:
            # load pretrained model for testing
            cwd = os.getcwd()
            #self.model.load_state_dict(torch.load(os.path.join(cwd, args.model_file)))

            self.model.load_state_dict(torch.load(os.path.join(cwd, args.model_file), map_location=torch.device('cpu')))
            self.model.eval()

    def reset(self):
        ''' restart '''
        pass

    def get_graph_node_emb(self,graph_r,graphs):
        for i in range(len(graph_r)):
            if id(graphs[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graph_r[i])] = models.get_init_node_embed(graph_r[i], self.device) # epochs for initial embedding
                embed = self.graph_node_embed[id(graph_r[i])]
                #np.savetxt(f"embed_graph_{i}_train.csv", embed.detach().cpu().numpy(), delimiter=",")


    def setup_graph_input(self, graph_r,graphs, actions=None):
        sample_size = len(graphs)
        data = []
        if actions is not None:
            actions = actions.to(self.device)
        for i in range(sample_size):
            with torch.no_grad():
                # copy node embedding as node feature
                x = self.graph_node_embed[id(graph_r[i])].detach().clone()
                #print(graphs[i].from_to_edges())
                edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
                y = actions[i].detach().clone() if actions is not None else None
                data.append(Data(x=x, edge_index=edge_index, y=y))

        with torch.no_grad():
            loader = DataLoader(data, pin_memory=False, num_workers=0, batch_size=sample_size, shuffle=False)
            for batch in loader:
                # adjust y if applicable
                if actions is not None:
                    total_num = 0
                    for i in range(1, sample_size):
                        total_num += batch[i - 1].num_nodes
                        batch[i].y += total_num
                return batch.to(self.device)


    def select_action(self, graph_r,graph, actions,budget):
        ''' act upon state '''
        graph_input = self.setup_graph_input([graph_r],[graph])
        with torch.no_grad():
                q_a = self.model(graph_input)
                if len(actions)+1>=budget:
                    top_q_values, _ = torch.topk(q_a.squeeze(), k=5)
                    #print(top_q_values.tolist())
                res=torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone().tolist()
                for index in res:
                    if index not in actions:
                        return index
        #if budget is None:
        #        return torch.argmax(q_a).detach().clone()
        #else:
        #        return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()

    def memorize(self, env):
        '''n step for stability'''
        returns = [0.0] * len(env.rewards)
        for i in reversed(range(len(env.rewards))):
            returns[i] = env.rewards[i] + (self.gamma * returns[i + 1] if i + 1 < len(env.rewards) else 0)       

        for i in range(len(env.actions)):
            if i + self.n_step < len(env.actions):
                self.memory.push(
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    torch.tensor([env.actions[i + self.n_step]], dtype=torch.long),
                    torch.tensor([returns[i] - (self.gamma ** self.n_step) * returns[i + self.n_step]], dtype=torch.float),
                    env.store_graph[i],
                    env.graph_realization,
                    env.store_graph[i+self.n_step])


    def fit(self):
        '''fit on a batch sampled from replay memory'''
        # optimize model
        sample_size = self.batch_size if len(self.memory) >= self.batch_size else len(self.memory)
        # need to fix dimension and restrict action space
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))

        actions = torch.cat(batch.action)               # a_t
        next_actions = torch.cat(batch.next_action)     # a_{t+n}
        rewards = torch.cat(batch.reward)               # r_t^{(n)}
        graph_r = batch.graph_r                       # graph realization (共享)
        graphs = batch.graph                            # s_t
        next_graphs_r = batch.next_graph_r                  # s_{t+n}                          # 状态 s_{t+n}

        current_inputs = self.setup_graph_input(graphs, graph_r, actions)
        next_inputs = self.setup_graph_input(graphs, next_graphs_r, next_actions)

        with torch.no_grad():
            next_q_values = self.model(next_inputs).squeeze(dim=1)

        expected_q_values = rewards.to(self.device) + (self.gamma ** self.n_step) * next_q_values
        current_q_values = self.model(current_inputs).squeeze(dim=1)
        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # if double dqn, update target network if needed
        if self.double_dqn:
            self.target.load_state_dict(self.model.state_dict())
            return True
        return False

    def save_model(self, file_name):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), os.path.join(cwd, file_name))


Transition = namedtuple('Transition',
                        ('action','next_action','reward','graph_r','graph','next_graph_r'))

class ReplayMemory(object):
    '''random replay memory'''
    def __init__(self, capacity):
        # temporily save 1-step snapshot
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


Agent = DQAgent
