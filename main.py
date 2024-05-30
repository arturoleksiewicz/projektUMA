import numpy as np
import random
import matplotlib.pyplot as plt

class TransportEnvironment:
    def __init__(self, num_locations, num_time_steps):
        self.num_locations = num_locations
        self.num_time_steps = num_time_steps
        self.state = (0, 0)  # początkowy stan (lokalizacja, czas)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        location, time_step = self.state
        new_location = (location + action) % self.num_locations
        reward = -abs(new_location - time_step)  # przykładowa funkcja nagrody
        self.state = (new_location, time_step + 1)
        done = (time_step + 1 == self.num_time_steps)
        return self.state, reward, done
class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.q_table.shape[1]))
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
def simulate_day(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    return total_reward

def run_experiments(num_episodes, env, agent):
    rewards = []
    for episode in range(num_episodes):
        total_reward = simulate_day(env, agent)
        rewards.append(total_reward)
    return rewards
num_locations = 5
num_time_steps = 10
num_states = num_locations * num_time_steps
num_actions = num_locations  # zakładając, że akcje to zmiana lokalizacji

env = TransportEnvironment(num_locations, num_time_steps)
agent = QLearningAgent(num_states, num_actions)

num_episodes = 1000
rewards = run_experiments(num_episodes, env, agent)

# Wizualizacja wyników
plt.plot(rewards)
plt.xlabel('Epizod')
plt.ylabel('Całkowita nagroda')
plt.title('Postęp uczenia się agenta Q-Learning')
plt.show()
