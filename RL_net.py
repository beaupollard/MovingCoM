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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class RL_NN(nn.Module):
    def __init__(self, states = 4, actions = 2, lr=1e-4, hidden_layers=64, batch_size=10):
        super(RL_NN, self).__init__()
        self.batch_size=batch_size
        self.states = states
        self.action_inputs=[-10.,10.]
        self.model_predict = nn.Sequential(
            nn.Linear(states+1, 2*hidden_layers),
            nn.ReLU(),
            nn.Linear(2*hidden_layers, 2*hidden_layers),
            nn.ReLU(),
            nn.Linear(2*hidden_layers, states),        
        )

        self.target_network = nn.Sequential(
            nn.Linear(states, 2*hidden_layers),
            nn.ReLU(),
            nn.Linear(2*hidden_layers, 2*hidden_layers),
            nn.ReLU(),
            nn.Linear(2*hidden_layers, actions)
        )


        self.optimizer=self.configure_optimizers(lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def forward(self, x):
 
        return self.target_network(x)


    def configure_optimizers(self,lr=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr, amsgrad=True)


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