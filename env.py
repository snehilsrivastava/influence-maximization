import torch
import dgl
from collections import namedtuple
import dgl.function as fn
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader

class IM(object):
    def __init__(
        self, 
        max_budget, 
        p,
        num_nodes,
        cost
        ):
        self.max_budget = max_budget
        assert(p <= 1 and p >= 0)
        self.p = p
        self.num_nodes = num_nodes
        self.cost = cost

    def compute_reward(self, state):
        # reward is the number of additional nodes influenced
        reward = 0
        # each node has one chance to influence each neighbour
        new_influenced = state
        tot_influenced = state
        while((new_influenced == 1).sum() >= 1):
            next = torch.full(
                (self.num_nodes, 1),
                0, 
                dtype = torch.long
                )
            for e in range(self.g.number_of_edges()):
                if((new_influenced[self.g.edges()[0][e]] == 1) and 
                   not(tot_influenced[self.g.edges()[1][e]] == 1 or new_influenced[self.g.edges()[1][e]] == 1)):
                    r = random.random()
                    if(r < self.p):
                        # node influenced
                        next[self.g.edges()[1][e]] = 1
                        reward += 1
            tot_influenced = torch.bitwise_or(tot_influenced, new_influenced)
            new_influenced = next
        return reward
     
    def step(self, action):
        reward, sol, done = self._take_action(action)
        
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info
    
    def _take_action(self, action):
        r1, r2 = 0, 0
        num_iter = 1000
        for i in range(num_iter):
            r1 += self.compute_reward(self.x)
        if(self.cost[action] < self.max_budget):
            self.x[action] = 1
            self.max_budget = self.max_budget - self.cost[action]
        next_sol = 0
        for i in range(num_iter):
            r2 += self.compute_reward(self.x)
        done = self._check_done()
        return (r2 - r1)/num_iter, next_sol, done

    def _check_done(self): 
        inactive = (self.x == 0).float()
        self.g.ndata['h'] = inactive
        not_selected = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')
        done = (not_selected == 0) or (self.max_budget <= 0)
        return done
                
    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).float()
        ob = torch.cat([ob_x], dim = 2)
        return ob
    
    # using num_samples = 1 as of now 
    def register(self, g, num_samples = 1):
        self.g = g
        self.g.set_n_initializer(dgl.init.zero_initializer)
        # self.num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (self.num_nodes, num_samples),
            0, 
            dtype = torch.long
            )
        ob = self._build_ob()
        return ob


# temporary code to show working of environment
cost = torch.tensor([300, 300, 300, 300, 300])
env = IM(1000, 0.6, 5, cost)

g = dgl.DGLGraph()
g.set_n_initializer(dgl.init.zero_initializer)

src_ids = torch.tensor([0, 1, 2, 3])
dst_ids = torch.tensor([1, 2, 3, 4])
g = dgl.graph((src_ids, dst_ids), num_nodes=5)
ob = env.register(g)

# print(env.g.edges())
# print(env.max_budget)
# print(env.p)
# print(env.x)
# print(ob)
# print(env._check_done())

obnext, reward, done, info = env.step(3)
print(obnext, reward, done, info)
obnext, reward, done, info = env.step(0)
print(obnext, reward, done, info)