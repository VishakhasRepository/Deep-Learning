# Import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

# Define the Cliff Walking environment
env = gym.make("CliffWalking-v0")

# Set the hyperparameters
num_episodes = 500
num_steps = 200
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
exploration_decay_rate = 0.99

# Initialize the Q-learning agent
num_actions = env.action_space.n
num_states = env.observation_space.n
q_table = np.zeros((num_states, num_actions))

# Define the epsilon-greedy policy
def epsilon_greedy_policy(state, q_table, epsilon):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    return action

# Train the Q-learning agent
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    epsilon = exploration_rate * (exploration_decay_rate ** episode)
    for step in range(num_steps):
        action = epsilon_greedy_policy(state, q_table, epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
            learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))
        state = next_state
        if done:
            break
    rewards.append(total_reward)

# Evaluate the trained agent
total_reward = 0
for episode in range(10):
    state = env.reset()
    for step in range(num_steps):
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
print("Average reward over 10 evaluation episodes: ", total_reward/10)

# Plot the learning curve
plt.plot(np.arange(num_episodes), rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()

# Plot the Q-value convergence
q_values = np.max(q_table, axis=1).reshape(env.observation_space.shape)
plt.imshow(q_values, cmap="cool")
plt.colorbar()
plt.show()

# Plot the policy heatmap
policy = np.argmax(q_table, axis=1).reshape(env.observation_space.shape)
plt.imshow(policy, cmap="cool")
plt.colorbar()
plt.show()

# Hyperparameter Tuning
# Vary the hyperparameters and repeat the training and evaluation steps to improve the performance of the agent.

# Code motivation: This code provides a solution to the classic reinforcement learning problem of Cliff Walking using Q-learning algorithm. The agent learns to navigate a gridworld environment with a cliff at the edge, while maximizing the total reward it receives over time. The Q-learning algorithm updates the Q-values based on the rewards received and the maximum Q-value of the next state. The agent's policy is learned through an epsilon-greedy exploration strategy. The code also includes results analysis and hyperparameter tuning to improve the performance of the agent.
