from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import random
from collections import deque, namedtuple

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

class AC(nn.Module):
    def __init__(self, states = 19, actions = 3, lr=1e-4, hidden_layers=128, batch_size=10, action_limits=12.):
        super().__init__()
        self.q = Qfunc(states=states,actions=actions)
        self.pi=Pifunc(states=states,actions=actions,action_limits=action_limits)
        self.action_len=actions
        self.pi_optimizer = torch.optim.AdamW(self.pi.parameters(), lr=lr)
        self.q_optimizer = torch.optim.AdamW(self.q.parameters(), lr=lr)
        
    def act(self, obs):
        with torch.no_grad():
            return np.clip(self.pi(obs).numpy(),-10,10)

    def get_action(self,o, noise_scale):
        a = self.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.action_len)
        return np.clip(a, -10, 10)



class Qfunc(nn.Module):
    def __init__(self, states = 19, actions = 5, lr=1e-2, hidden_layers=128, batch_size=10):
        super().__init__()
        self.batch_size=batch_size
        self.states = states
        self.q = nn.Sequential(
            nn.Linear(states+actions, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, 1),
        )

    def forward(self,obs,act):
        q=self.q(torch.cat([obs, act],dim=-1))
        return torch.squeeze(q,-1)
    
class Pifunc(nn.Module):
    def __init__(self, states = 19, actions = 5, lr=1e-4, hidden_layers=128, batch_size=10,action_limits=12.):
        super().__init__()
        self.batch_size=batch_size
        self.states = states
        self.action_limits=action_limits
        self.pi = nn.Sequential(
            nn.Linear(states, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, 2*hidden_layers),
            nn.Tanh(),
            nn.Linear(2*hidden_layers, actions),
            nn.Tanh(),        
        )

    def forward(self,obs):
        pi=self.action_limits*self.pi(obs)
        return torch.squeeze(pi,-1)

# class RL_A(nn.Module):
#     def __init__(self, states = 19, actions = 5, lr=1e-4, hidden_layers=128, batch_size=10):
#         super(RL_A, self).__init__()
#         self.batch_size=batch_size
#         self.states = states
#         self.action_inputs=np.array([[-0.3,-0.15],[-0.3,0.15],[0.2,-0.15],[0.2,0.15]])

#         self.target_network = nn.Sequential(
#             nn.Linear(states, 2*hidden_layers),
#             nn.ReLU(),
#             nn.Linear(2*hidden_layers, 2*hidden_layers),
#             nn.ReLU(),
#             nn.Linear(2*hidden_layers, actions)
#         )


#         self.optimizer=self.configure_optimizers(lr=lr)
#         self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

#     def forward(self, x, train=True):
#         action = torch.clamp(self.target_network(x), min=-10., max=10.)
#         if train==True:
#             action=action+np.random.normal(0,1.)
#         return action


#     def configure_optimizers(self,lr=1e-4):
#         return torch.optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

    # def optimize_model(self,memory):
    #     if len(memory) < self.batch_size:
    #         return
    #     transitions = memory.sample(self.batch_size)
    #     batch = Transition(*zip(*transitions))

    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), dtype=torch.bool)
    #     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    #     state_batch = torch.cat(batch.state).reshape((self.batch_size,self.states))
    #     action_batch = torch.cat(batch.action).reshape((self.batch_size,1))
    #     reward_batch = torch.cat(batch.reward).reshape((self.batch_size,1))

    #     state_action_values = self.forward(state_batch)

    #     return 