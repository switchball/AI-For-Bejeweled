#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File DQN.py created on 23:05 2017/10/19 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import cv2
import random
import tensorflow as tf
from collections import deque

from Agent import Agent
from BejeweledEnvironment import *

REPLAY_SIZE = 128000
BATCH_SIZE = 32
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.05
GAMMA = 0.9


class DQN(Agent):
    def __init__(self):
        super(DQN, self).__init__()
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_dim = 2*7*8 + 1

        self.create_Q_network()
        self.create_training_method()

        assert self.Q_value.graph is tf.get_default_graph()
        init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.Q_value.graph)
        self.session.run(init)

    def create_Q_network(self):
        input_layer = tf.placeholder(tf.float32, [None, 8, 8, 9])
        action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        labels_input = tf.placeholder(tf.float32, [None])
        self.s_input = input_layer
        self.a_input = action_input
        self.y_label = labels_input

        with tf.variable_scope("dqn", reuse=False):
            # Convolutional Layer #1
            # Computes 16 features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 8, 8, 9]
            # Output Tensor Shape: [batch_size, 8, 8, 16]
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
            print(conv1.name)

            # Convolutional Layer #2
            # Computes 64 features using a 3x3 filter.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 8, 8, 16]
            # Output Tensor Shape: [batch_size, 8, 8, 64]
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            # Flatten tensor into a batch of vectors
            # Input Tensor Shape: [batch_size, 8, 8, 64]
            # Output Tensor Shape: [batch_size, 8 * 8 * 64]
            conv2_flat = tf.reshape(conv2, [-1, 8 * 8 * 64])

            # Dense Layer
            # Densely connected layer with 256 neurons
            # Input Tensor Shape: [batch_size, 8 * 8 * 64]
            # Output Tensor Shape: [batch_size, 113]
            dense = tf.layers.dense(inputs=conv2_flat, units=self.action_dim, activation=tf.nn.relu)

            self.Q_value = dense


    def create_training_method(self):
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.a_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_label - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random mini_batch from replay memory
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(session=self.session,
                                          feed_dict={self.s_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # Step 3: optimize
        self.optimizer.run(session=self.session, feed_dict={
            self.y_label: y_batch,
            self.a_input: action_batch,
            self.s_input: state_batch
        })

    def greedy_action(self, state):
        Q_value = self.Q_value.eval(session=self.session, feed_dict={
            self.s_input: [state]
        })[0]

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1), 1
        else:
            return np.argmax(Q_value), 0

    def action(self, state):
        Q_value = self.Q_value.eval(session=self.session, feed_dict={
            self.s_input: state
        })
        return np.argmax(Q_value)

if __name__ == '__main__':
    env = BejeweledEnvironment()
    agent = DQN()
    STEP_NUM = 500
    TEST_ROUND = 10

    for episode in range(10000):
        _state, initial_score = env.reset()
        state = _state.state

        print("Episode {} start.".format(episode))

        for step in range(STEP_NUM):
            action, flag_greedy = agent.greedy_action(state)
            _next_state, reward, done = env.step(action)
            next_state = _next_state.state
            print("{}#{} Step Action: {}, Reward: {} Greedy: {} eps={}".
                  format(episode, step, BejeweledAction().action_space[action], reward, flag_greedy, agent.epsilon))

            env.render()

            if np.count_nonzero(_next_state.prediction == 0) > 40:
                print("No detection, sleep for 3 seconds.")
                time.sleep(3)

            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # test per 10 episode
        if episode % 10 == 0 and episode > 0:
            total_reward = 0
            for i in range(TEST_ROUND):
                state = env.reset()
                for j in range(STEP_NUM):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done = env.step(action)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST_ROUND
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)
            if avg_reward >= 10000:
                break

