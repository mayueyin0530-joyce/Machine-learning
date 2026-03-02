import numpy as np

class QLearningAgent:
    """Implementation of Q-learning algorithm"""
    def __init__(self, state_num, action_num):
        #Obtain the number of states and actions from the environment
        self.state_num = state_num
        self.action_num = action_num

        self.lr = 0.1  #Learning rate α
        self.gamma = 0.9  #Discount factor γ
        self.epsilon = 0.9  #Initial exploration rate ε
        self.epsilon_min = 0.1  #Minimum exploration rate
        self.epsilon_decay_rate = 0.995  #Exploration rate attenuation coefficient

        #Initialize the Q-table
        self.q_table = np.zeros((self.state_num, self.action_num))

    #ε-greedy strategy for selecting actions
    def choose_action(self, state):
        #Random number is less than ε: Random exploration; greater than ε: Select the optimal action
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_num)
        else:
            #Select the action with the maximum Q value in the current state
            state_q_values = self.q_table[state, :]
            action = np.random.choice(np.where(state_q_values == np.max(state_q_values))[0])
        return action

    #Update Q-table
    def update_q_table(self, state, action, reward, next_state):
        #Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    #Exploration rate decay function
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def choose_best_action(self, state):
        state_q_values = self.q_table[state, :]
        action = np.random.choice(np.where(state_q_values == np.max(state_q_values))[0])
        return action