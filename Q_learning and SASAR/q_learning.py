import numpy as np
import gym

class QLearning:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = self._greedy_action(state)
        return action
    
    def _greedy_action(self, state):
        q_values = self.q_table[state, :]
        action = np.random.choice(np.where(q_values == np.max(q_values))[0])
        return action
    
    def learn(self, state, action, reward, next_state, done):
        q_predict = self.q_table[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (q_target - q_predict)

def train(env, agent, episodes=100000):
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        while True:
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            action = agent.choose_action(state)
            if done:
                break
            if episode % 5000 == 0:
                env.render()
    return agent.q_table

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    agent = QLearning(env)
    q_table = train(env, agent)
    print(q_table)
    # test(env, agent)
    env.close()