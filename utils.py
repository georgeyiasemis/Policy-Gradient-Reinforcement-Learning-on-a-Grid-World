import torch
import torch.nn as nn
import torch.distributions as distributions


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


