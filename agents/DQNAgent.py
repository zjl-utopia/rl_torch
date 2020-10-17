import numpy as np

import torch
import torch.nn as nn
from torch import optim

from .Agent import Agent
from utils.Memory import ReplayBuffer


class DQNAgent(Agent):
    def __init__(self, n_state, n_action, action_shape, batch_size=32, learning_rate=10e-3, epsilon=0.9, gamma=0.9,
                 network=None, optimizer=optim.Adam, memory=ReplayBuffer, capacity=2000,
                 use_target_net=True, target_update_freq=100, dueling=False):

        super(DQNAgent, self).__init__(n_state=n_state, n_action=n_action, batch_size=batch_size,
                                       learning_rate=learning_rate, epsilon=epsilon, gamma=gamma,
                                       optimizer=optimizer, memory=memory(capacity))

        self.action_shape = action_shape

        # double dqn
        self.use_target_net = use_target_net
        self.target_update_freq = target_update_freq

        # dueling dqn
        self.dueling = dueling

        # -----------Define 2 networks (target and training)------#
        self.eval_net = network()
        if self.use_target_net:
            self.target_net = network()

        # Define counter and loss function
        self.learn_step_counter = 0

        # ------- Define the optimizer------#
        self.optimizer = optimizer(self.eval_net.parameters(), lr=learning_rate)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def select_action(self, x, deterministic=False):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < self.epsilon or deterministic:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.action_shape == 0 else action.reshape(self.action_shape)
        else:  # random
            action = np.random.randint(0, self.n_action)
            action = action if self.action_shape == 0 else action.reshape(self.action_shape)
        return action

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.use_target_net and self.learn_step_counter % self.target_update_freq == 0:
            # Assign the parameters of eval_net to target_net
            self.update_target_net()
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        state, action, reward, next_state, done = self.sample_transition(self.batch_size)

        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = torch.FloatTensor(state)
        b_a = torch.LongTensor(action.astype(int))
        b_r = torch.FloatTensor(reward)
        b_s_ = torch.FloatTensor(next_state)

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)

        # calculate the q value of next state
        if self.use_target_net:
            q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        else:
            q_next = self.eval_net(b_s_).detach()
        # select the maximum q value

        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

    def update_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
