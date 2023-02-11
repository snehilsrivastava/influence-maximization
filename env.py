import torch
import dgl
from collections import namedtuple
import dgl.function as fn
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader
from dqn_agent import Agent
import numpy as np
from collections import deque

class IM(object):
    def __init__(self, max_budget, p, num_nodes, cost):
        self.max_budget = max_budget
        assert(p <= 1 and p >= 0)
        self.p = p
        self.num_nodes = num_nodes
        self.cost = cost

    def compute_reward(self, state):
        # reward is the number of additional nodes influenced
        reward = 0
        # each node has one chance to influence each neighbour
        # print(state)
        new_influenced = state.detach().cpu().numpy().ravel()
        # print(new_influenced)
        tot_influenced = state.detach().cpu().numpy().ravel()
        while((new_influenced == 1).sum() >= 1):
            # next = torch.full(
            #     (self.num_nodes, 1),
            #     0, 
            #     dtype = torch.long
            #     )
            next = np.zeros(self.num_nodes)
            for e in range(self.g.number_of_edges()):
                # print(new_influenced[self.g.edges()[0][e]])
                if((new_influenced[self.g.edges()[0][e]] == 1) and 
                   not(tot_influenced[self.g.edges()[1][e]] == 1 or new_influenced[self.g.edges()[1][e]] == 1)):
                    r = random.random()
                    if(r < self.p):
                        # node influenced
                        next[self.g.edges()[1][e]] = 1
                        reward += 1
            # print(new_influenced)
            tot_influenced = tot_influenced + new_influenced
            new_influenced = next
        return reward
     
    def step(self, action):
        reward, sol, done = self._take_action(action)
        
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        # need to convert ob to ndarray from tensor
        ob = ob.detach().cpu().numpy().ravel()
        return ob, reward, done, info
    
    def _take_action(self, action):
        r1, r2 = 0, 0
        num_iter = 100
        for i in range(num_iter):
            r1 += self.compute_reward(self.x[:-1])
        if(self.cost[action] < self.max_budget):
            self.x[action] = 1
            self.x[-1] -= self.cost[action]
            self.max_budget -= self.cost[action]
        # write code for else case 
        next_sol = 0
        for i in range(num_iter):
            r2 += self.compute_reward(self.x[:-1])
        done = self._check_done()
        return (r2 - r1)/num_iter, next_sol, done

    def _check_done(self): 
        inactive = (self.x[:-1] == 0).type(torch.float)
        # print(inactive)
        self.g.ndata['h'] = inactive
        not_selected = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')
        done = (not_selected == 0) or (self.max_budget <= 0)
        return done
                
    def _build_ob(self):
        ob_x = self.x
        # ob = torch.cat([ob_x], dim = 2)
        # return ob
        return ob_x
    
    # using num_samples = 1 as of now 
    def register(self, g, num_samples = 1):
        self.g = g
        self.g.set_n_initializer(dgl.init.zero_initializer)
        t = torch.full((self.num_nodes, 1), 0, dtype=torch.float16)
        # torch.full(
            #     (self.num_nodes, 1),
            #     0, 
            #     dtype = torch.long
            #     )
        self.x = torch.cat((t, torch.tensor([[self.max_budget]])), 0)
        ob = self._build_ob()
        return ob


cost = torch.tensor([300, 300, 300, 300, 300])
maxb = 1000
env = IM(maxb, 0.6, 5, cost)
src_ids = torch.tensor([0, 1, 2, 3])
dst_ids = torch.tensor([1, 2, 3, 4])
g = dgl.graph((src_ids, dst_ids), num_nodes=5)
ob = env.register(g)

# agent = Agent(state_size=env.num_nodes + 1, action_size=1, seed=0)
# # watch an untrained agent
# state = np.array([0, 0, 0, 0, 0, 1000])
# for j in range(200):
#     action = agent.act(state)
#     # print(action)
#     state, reward, done, _ = env.step(action)
    
#     if done:
#         break 
        

agent = Agent(state_size=env.num_nodes + 1, action_size=1, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = np.array([0, 0, 0, 0, 0, maxb])
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()