import numpy as np
import pandas as pd
import gym
from gym.envs.classic_control import rendering

env = gym.make('Snake-v0')
env.reset()

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # build and return a neural network model
        pass

    def act(self, state):
        # return an action based on the current state
        pass

    def replay(self, batch_size):
        # replay memory to update Q-values
        pass

    def remember(self, state, action, reward, next_state, done):
        # store experience in replay memory
        pass

def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

agent = Agent(state_size, action_size)

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(max_time):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

scores = []
for e in range(test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    for time in range(max_time):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = np.reshape(next_state, [1, state_size])
        if done:
            break
    scores.append(score)
print("Average score: ", np.mean(scores))
