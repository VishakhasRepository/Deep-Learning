import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from Kaggle using pandas
data = pd.read_csv('https://www.kaggle.com/mojocolors/900000-hands-of-blackjack-results')

# Preprocess the data to extract the relevant features and labels
X = data.drop(['result', 'player_hand', 'dealer_hand'], axis=1)
y = data['result']

# Split the data into training and testing sets using scikit-learn's train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import gym

# Define the state space, action space, and rewards for the game
state_space = 2  # player's hand and dealer's upcard
action_space = 2  # hit or stand
rewards = {'win': 1, 'draw': 0, 'lose': -1}

# Implement the game environment using the OpenAI Gym framework
class BlackjackEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(action_space)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(10),  # player's hand value
            gym.spaces.Discrete(10)   # dealer's upcard value
        ))
        self.reward_range = (-1, 1)
        self._reset()

    def _step(self, action):
        # Implement the game logic for a single step
        ...
        return observation, reward, done, {}

    def _reset(self):
        # Reset the game to the initial state
        ...
        return observation

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the Q-learning or SARSA algorithm for the agent
class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            Dense(32, input_shape=(self.state_space,), activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_space, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        # Implement epsilon-greedy policy
        ...

    def update_model(self, state, action, reward, next_state, done):
        # Implement the Q-learning or SARSA algorithm
        ...

import numpy as np
import matplotlib.pyplot as plt

# Train the agent using the training data
agent = Agent(state_space, action_space)
episodes = 1000
steps_per_episode = 100
history = {'episode_reward': [], 'epsilon': []}

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(steps_per_episode):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.update_model(state, action, reward, next_state, done)
        state = next_state

        if done:
            break

    history['episode_reward'].append(episode_reward)
    history['epsilon'].append(agent.epsilon)
    agent.epsilon *= agent.epsilon_decay

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

# Plot the learning curve for the agent
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history['episode_reward'])
ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward')
ax.set_title('Learning Curve')
plt.show()

from sklearn.model_selection import ParameterGrid

# Define a grid of hyperparameters to search over
param_grid = {'alpha': [0.1, 0.2, 0.3], 'gamma': [0.9, 0.95, 0.99], 'epsilon_decay': [0.95, 0.99]}

# Perform a grid search over the hyperparameters
best_reward = -np.inf
best_params = None
for params in ParameterGrid(param_grid):
    agent = Agent(state_space, action_space, alpha=params['alpha'], gamma=params['gamma'], epsilon_decay=params['epsilon_decay'])
    history = train(agent, env, episodes=1000, steps_per_episode=100, verbose=False)
    mean_reward = np.mean(history['episode_reward'])
    if mean_reward > best_reward:
        best_reward = mean_reward
        best_params = params

print(f"Best hyperparameters: {best_params}")

# Test the agent using the testing data
history = test(agent, env, episodes=100, steps_per_episode=100, verbose=True)

# Calculate and print the mean episode reward and win rate
mean_reward = np.mean(history['episode_reward'])
win_rate = np.mean([reward == 1 for reward in history['episode_reward']])
print(f"Mean episode reward: {mean_reward:.2f}")
print(f"Win rate: {win_rate:.2%}")

import pickle

# Save the trained agent to a file
with open('agent.pkl', 'wb') as f:
    pickle.dump(agent, f)

# Load the trained agent from a file
with open('agent.pkl', 'rb') as f:
    agent = pickle.load(f)

# Use the trained agent to play a game of blackjack
state = env.reset()
done = False
while not done:
    action = agent.get_action(state, train=False)
    state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
