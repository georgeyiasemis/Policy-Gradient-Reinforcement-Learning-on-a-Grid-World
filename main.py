# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 2021

@author: George Yiasemis
"""
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from environment import Environment
from agent import Agent
from pg import PolicyNet
from utils import *


# Define Environments and Agents
gamma = 0.8
train_environment = Environment(display=False, magnification=500, id=' Train')
train_agent = Agent(train_environment, gamma=gamma)

test_environment = Environment(display=False, magnification=500, id=' Test')
test_agent = Agent(test_environment, gamma=gamma)

policy = PolicyNet(2, 256, train_agent.environment.num_actions)

num_episodes = 200
num_episode_steps = 20

# Model Parameters

lr = 0.002
optimiser = optim.Adam(policy.parameters(), lr=lr)

print_every_episode = 1

lookback = 5
train_rewards = []
test_rewards = []


for episode in range(1, num_episodes + 1):

    # Training
    loss, train_reward, train_trace = train(train_agent, policy, optimiser, gamma, num_episode_steps)
    train_agent.environment.plot_trace(train_trace)

    # Validation 
    test_reward, test_trace = evaluate(test_agent, policy, num_episode_steps)
    test_agent.environment.plot_trace(test_trace)

    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    mean_train_rewards = np.mean(train_rewards[-lookback:])
    mean_test_rewards = np.mean(test_rewards[-lookback:])

    if episode % print_every_episode == 0:

        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')



plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.legend(loc='lower right')
plt.grid()
plt.savefig('rewards.png')