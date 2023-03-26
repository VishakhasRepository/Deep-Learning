import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

class DQNAgent:
    ...
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

batch_size = 32
n_episodes = 1000
output_dir = 'model_output/lunar_lander/'

agent = DQNAgent(state_size, action_size)
done = False
for e in range(n_episodes):
    state = env.reset().reshape(1, state_size)
    for time in range(1000):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(1, state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, n_episodes, time, agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
    if e % 50 == 0:
        agent.model.save_weights(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

# e. evaluating the trained agent
def evaluate_agent(env, agent, episodes=10):
    rewards = []
    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print("Episode ", i+1, " Reward: ", total_reward)
    avg_reward = sum(rewards) / episodes
    print("Average reward: ", avg_reward)
