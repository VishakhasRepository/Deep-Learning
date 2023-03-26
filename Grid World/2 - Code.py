import numpy as np

# Define the Gridworld environment
class Gridworld:
    def __init__(self):
        self.grid = np.zeros((4,4))
        self.grid[0, 0] = -1
        self.grid[3, 3] = -1
        self.current_state = [0, 0]
        
    def step(self, action):
        i, j = self.current_state
        if action == "up":
            next_state = [max(i-1, 0), j]
        elif action == "down":
            next_state = [min(i+1, 3), j]
        elif action == "left":
            next_state = [i, max(j-1, 0)]
        elif action == "right":
            next_state = [i, min(j+1, 3)]
        else:
            raise ValueError("Invalid action.")
        reward = self.grid[next_state[0], next_state[1]]
        self.current_state = next_state
        return next_state, reward

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.Q = np.zeros((4, 4, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(["up", "down", "left", "right"])
        else:
            return ["up", "down", "left", "right"][np.argmax(self.Q[state[0], state[1]])]
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = [0, 0]
            while state != [3, 3]:
                action = self.choose_action(state)
                next_state, reward = self.env.step(action)
                self.Q[state[0], state[1], ["up", "down", "left", "right"].index(action)] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state[0], next_state[1]]) - self.Q[state[0], state[1], ["up", "down", "left", "right"].index(action)])
                state = next_state

# Create Gridworld environment
env = Gridworld()

# Create Q-learning agent and train it
agent = QLearningAgent(env)
agent.train(1000)

# Evaluate the trained agent
state = [0, 0]
while state != [3, 3]:
    action = agent.choose_action(state)
    next_state, _ = env.step(action)
    state = next_state
    print(state)

import matplotlib.pyplot as plt

# 5. Visualize the learned Q-values

# Define a function to plot the learned Q-values for each state and action
def plot_q_values(Q):
    # Create a plot with subplots for each action
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    
    # Iterate over each state and plot the Q-values for each action
    for s in range(env.observation_space.n):
        q_values = [Q[s, a] for a in range(env.action_space.n)]
        axs[s].bar(range(env.action_space.n), q_values)
        axs[s].set_title(f"State {s}")
        axs[s].set_xticks(range(env.action_space.n))
        axs[s].set_xlabel("Action")
        axs[s].set_ylabel("Q-Value")
    
    # Set the title for the plot
    fig.suptitle("Learned Q-Values")
    
    # Show the plot
    plt.show()

# Plot the learned Q-values
plot_q_values(Q)
