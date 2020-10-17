import numpy as np
from abc import ABCMeta, abstractmethod
from utils.Memory import Transition


class Agent(metaclass=ABCMeta):
    def __init__(self, n_state, n_action, batch_size, learning_rate, epsilon, gamma, optimizer, memory):
        self.n_state = n_state
        self.n_action = n_action
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = optimizer
        self.memory = memory

    def store_transition(self, s, a, r, s_, done):
        self.memory.push(s, a, r, s_, done)

    def sample_transition(self, batch_size):
        b_memory = Transition(*zip(*self.memory.sample(batch_size)))
        return np.array(b_memory.state), np.array([b_memory.action]).T, np.array([b_memory.reward]).T, \
               np.array(b_memory.next_state), np.array(b_memory.done)

    def ready_to_learn(self):
        return self.memory.is_full()
        # return len(self.memory) >= self.batch_size

    @abstractmethod
    def select_action(self, x):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError
