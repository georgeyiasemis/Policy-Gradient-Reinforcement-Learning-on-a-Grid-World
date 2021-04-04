import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np

from environment import Environment
from agent import Agent
from pg import PolicyNet


# Define Environments and Agents
gamma = 0.9
train_environment = Environment(display=False, magnification=500, id=' Train')
train_agent = Agent(train_environment, gamma=gamma)

test_environment = Environment(display=False, magnification=500, id=' Test')
test_agent = Agent(test_environment, gamma=gamma)

policy = PolicyNet(2, 128, train_agent.environment.num_actions)

num_episodes = 200
num_episode_steps = 20

# Model Parameters

lr = 0.01
optimiser = optim.Adam(policy.parameters(), lr=lr)

print_every_episode = 1

def train(agent, policy, optimiser, gamma, num_episode_steps):

    policy.train()

    action_log_probs = []
    rewards = []
    episode_reward = 0

    agent.reset()
    state = agent.state
    trace = [state]
    end_episode = False
    for i in range(num_episode_steps):

        state = torch.tensor(state).float()

        # Make a prediction using the Policy network
        pred = policy(state)

        # Calculate \pi_{\theta}( a | s_t) for all a \in {0, 1, ..., .environment.num_actions-1}
        action_probs = nn.Softmax(-1)(pred)

        # Categorical Distribution
        p = distributions.Categorical(probs=action_probs)

        # Choose an action a_t \in {0, 1, ..., .environment.num_actions-1}
        # with probability action_probs
        action = p.sample()

        # Calculate \pi_{\theta}( a_t | s_t)
        action_log_prob = p.log_prob(action)

        # Take a step using the policy; observe reward and next state
        (_, _, reward, state), end_episode = agent.step(action.item())

        action_log_probs.append(action_log_prob)
        rewards.append(reward)
        trace.append(state)
        episode_reward += reward

        if end_episode:
            break

    # Tensor of shape (min(num_episode_steps,steps_to_goal), 1)
    action_log_probs = torch.stack(action_log_probs)

    R = calculate_returns(rewards, gamma)

    loss = update_policy(R, action_log_probs, optimiser)

    return loss, episode_reward, trace

def calculate_returns(rewards, discount_factor, normalise = True):

    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalise:
        returns = (returns - returns.mean()) / returns.std()

    return returns

def update_policy(returns, action_log_probs, optimiser):

    returns = returns.detach()

    loss = - (returns * action_log_probs).sum()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    return loss.item()

def evaluate(agent, policy, num_episode_steps):

    policy.eval()

    reached_goal = False
    episode_reward = 0
    agent.reset()

    state = agent.state
    trace = [state]
    for i in range(num_episode_steps):

        state = torch.tensor(state).float()

        with torch.no_grad():

            pred = policy(state)

            action_prob = nn.Softmax(-1)(pred)

            action = action_prob.argmax(-1)

        (_, _, reward, state), reached_goal = agent.step(action.item())
        trace.append(state)
        episode_reward += reward

        if reached_goal:
            break

    return episode_reward, trace

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
