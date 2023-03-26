import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
env.render()

def q_learning_algorithm(alpha, gamma, epsilon, episodes, max_steps):
    
    # initialize Q table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    
    for episode in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # epsilon-greedy strategy for exploration/exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
                
            next_state, reward, done, info = env.step(action)
            
            # Q-table update
            q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state, :]))
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        epsilon = 1 / episode
        rewards.append(total_reward)
        
    return q_table, rewards

alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 10000
max_steps = 100

q_table, rewards = q_learning_algorithm(alpha, gamma, epsilon, episodes, max_steps)

# plot rewards over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning on Taxi-v3')
plt.show()

# test the agent
state = env.reset()
env.render()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

print("Total reward: ", total_reward)

# evaluate the agent with different hyperparameters
alphas = [0.1, 0.2, 0.3]
gammas = [0.5, 0.6, 0.7]
epsilons = [0.05, 0.1, 0.15]
episodes = 10000
max_steps = 100

results = {}

for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:
            q_table, rewards = q_learning_algorithm(alpha, gamma, epsilon, episodes, max_steps)
            results[(alpha, gamma, epsilon)] = np.mean(rewards[-100:])
            
# find the best hyperparameters
best_params = max(results, key=results.get)
print("Best hyperparameters: ", best_params)
