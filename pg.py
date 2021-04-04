# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 2021

@author: George Yiasemis
"""

import torch.nn as nn

class PolicyNet(nn.Module):
    '''
    Policy Gradient Network consisted of two feedfowrard neural networks.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super(PolicyNet, self).__init__()
        # input_dim = state space dim
        # output_dim = action space dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.fc1,
                                self.dropout,
                                self.relu,
                                self.fc2)

    def forward(self, x):

        x = self.net(x)
        return x
