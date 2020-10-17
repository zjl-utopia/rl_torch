from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """save transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return len(self.memory) == self.capacity

    def __len__(self):
        return len(self.memory)


# import random
# import numpy as np
# from collections import deque
#
#
# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         state = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)
#
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done
#
#     def is_full(self):
#         return len(self.buffer) == self.capacity
#
#     def __len__(self):
#         return len(self.buffer)
