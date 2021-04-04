import numpy as np


# The Agent class allows the agent to interact with the environment.
class Agent():

    # The class initialisation function.
    def __init__(self, environment, gamma=0.9, reward_fun=None):
        # Set the agent's environment.
        self.environment = environment
        # Set gamma for the Bellman Eq.
        self.gamma = gamma
        # Create the agent's current state
        self.state = None
        # Set reward function
        if reward_fun == None:
            self.reward_fun = reward_fun_a(a=1)
        else:
            self.reward_fun = reward_fun
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, discrete_action):# epsilon=None, greedy=False):

        assert discrete_action in range(self.environment.num_actions)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, self.distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(self.distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # print(self.distance_to_goal)
        has_reached_goal = np.all(self.state.round(2) == self.environment.goal_state.round(2))
        # Return the transition and
        return transition, has_reached_goal


    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):

        return self.reward_fun(distance_to_goal)
    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        # Up, Right, Down, Left
        actions = {0: np.array([0.1, 0]), 1: np.array([0, 0.1]),
                   2: np.array([-0.1, 0]), 3: np.array([0, -0.1])}

        return actions[discrete_action].astype('float32')

def reward_fun_a(a=1):

    return lambda dist: np.power(1 - dist, a)

def step_reward_fun(goal_reward=1):

    return lambda dist: goal_reward if dist <= 0.05 else 0
