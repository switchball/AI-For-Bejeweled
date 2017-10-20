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
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.05
GAMMA = 0.7


class DQN(Agent):
    def __init__(self):
        super(DQN, self).__init__()
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_dim = 2*7*8 + 1

        self.checkpointDir = './model/dqn_model/'

        self.create_Q_network()
        self.create_training_method()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dqn')
        self.saver = tf.train.Saver(var_list=variables)

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
            # Output Tensor Shape: [batch_size, 6, 6, 16]
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name='conv1')
            print(conv1.name)

            self.conv1 = conv1

            # Convolutional Layer #2
            # Computes 64 features using a 3x3 filter.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 8, 8, 16]
            # Output Tensor Shape: [batch_size, 6, 6, 64]
            # conv2 = tf.layers.conv2d(
            #     inputs=conv1,
            #     filters=16,
            #     kernel_size=[3, 3],
            #     padding="valid",
            #     activation=tf.nn.relu)

            # Flatten tensor into a batch of vectors
            # Input Tensor Shape: [batch_size, 8, 8, 64]
            # Output Tensor Shape: [batch_size, 4 * 4 * 8]
            conv1_flat = tf.reshape(conv1, [-1, 6 * 6 * 16])

            # Dense Layer
            # Densely connected layer with 256 neurons
            # Input Tensor Shape: [batch_size, 8 * 8 * 64]
            # Output Tensor Shape: [batch_size, 113]
            dense = tf.layers.dense(inputs=conv1_flat, units=self.action_dim, activation=tf.nn.relu)

            self.Q_value = dense


    def create_training_method(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.a_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_label - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost, global_step=self.global_step)

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpointDir)
        if ckpt:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("[DQN Prepare] Model " + ckpt.model_checkpoint_path + " restored.")
        else:
            print("[DQN Prepare] Model not found at", self.checkpointDir)

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

        if self.time_step % 200 == 0:
            self.saver.save(self.session, self.checkpointDir + 'model.ckpt', global_step=self.global_step)
            print('[DQN Model] model saved:', self.global_step)

    def eval_result(self, state):
        v = self.conv1.eval(session=self.session, feed_dict={
            self.s_input: [state]
        })[0]
        return v

    def greedy_action(self, state):
        Q_value = self.Q_value.eval(session=self.session, feed_dict={
            self.s_input: [state]
        })[0]

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 50000


        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1), Q_value, 1
        else:
            return np.argmax(Q_value), Q_value, 0

    def action(self, state):
        Q_value = self.Q_value.eval(session=self.session, feed_dict={
            self.s_input: [state]
        })[0]
        return np.argmax(Q_value)

def tag(img, q_values, action, reward, action_space):
    step_h, step_w = int(img.shape[0] / 8), int(img.shape[1] / 8)
    def p(c, r):
        return (int((c+0.5)*step_w), int((r+0.5)*step_h))

    maximum = np.max(q_values)
    minimum = np.min(q_values)
    average = np.average(q_values)
    nonzero = np.count_nonzero(q_values==0)
    print('qv, max={}, min={}, avg={}, #Zeros={}'.format(maximum, minimum, average, nonzero))
    print(sorted(-q_values))
    for idx, qv in enumerate(q_values):
        a, b, c = action_space[idx]
        if c == 'H':
            row1, row2 = a, a
            col1, col2 = b, b + 1
        elif c == 'V':
            col1, col2 = a, a
            row1, row2 = b, b + 1
        else:
            continue
        if qv >= 0:
            color = (0, 0, int(qv/maximum*255))
            thickness = int(qv/maximum*5)
        else:
            color = (0, int(qv/minimum*255), 0)
            thickness = int(qv/maximum*5)

        cv2.line(img, p(col1, row1), p(col2, row2), color, thickness=thickness)
        if idx == action:
            cv2.rectangle(img, p(col1 - 0.4, row1 - 0.4), p(col2 + 0.4, row2 + 0.4), (255, 0, 0), 3)

    if reward > 0:
        cv2.rectangle(img, (0, 0), p(2, 1), (0, 255, 0), -1)
        cv2.putText(img, '%s' % int(reward*100), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)


    cv2.imshow('Sprites', img)
    cv2.waitKey(1)

def check_board(prediction):
    p = np.array(prediction)
    for idx in indices:
        x = p[idx]
        if (x==x[0]).all():
            return True
    return False

def default_solution(prediction):
    solution = []
    for idx, action in enumerate(action_space):
        a, b, c = action
        p = prediction
        if c == 'H':
            row1, row2 = a, a
            col1, col2 = b, b + 1
        elif c == 'V':
            col1, col2 = a, a
            row1, row2 = b, b + 1
        x = row1 * 8 + col1
        y = row2 * 8 + col2
        p[x], p[y] = p[y], p[x]
        if check_board(p):
            solution.append(idx)
        p[x], p[y] = p[y], p[x]


    return random.choice(solution) if len(solution) > 0 else len(action_space)-1

indices = [[0, 1, 2], [0, 8, 16], [1, 2, 3], [8, 16, 24], [2, 3, 4], [16, 24, 32], [3, 4, 5], [24, 32, 40], [4, 5, 6], [32, 40, 48], [5, 6, 7], [40, 48, 56], [8, 9, 10], [1, 9, 17], [9, 10, 11], [9, 17, 25], [10, 11, 12], [17, 25, 33], [11, 12, 13], [25, 33, 41], [12, 13, 14], [33, 41, 49], [13, 14, 15], [41, 49, 57], [16, 17, 18], [2, 10, 18], [17, 18, 19], [10, 18, 26], [18, 19, 20], [18, 26, 34], [19, 20, 21], [26, 34, 42], [20, 21, 22], [34, 42, 50], [21, 22, 23], [42, 50, 58], [24, 25, 26], [3, 11, 19], [25, 26, 27], [11, 19, 27], [26, 27, 28], [19, 27, 35], [27, 28, 29], [27, 35, 43], [28, 29, 30], [35, 43, 51], [29, 30, 31], [43, 51, 59], [32, 33, 34], [4, 12, 20], [33, 34, 35], [12, 20, 28], [34, 35, 36], [20, 28, 36], [35, 36, 37], [28, 36, 44], [36, 37, 38], [36, 44, 52], [37, 38, 39], [44, 52, 60], [40, 41, 42], [5, 13, 21], [41, 42, 43], [13, 21, 29], [42, 43, 44], [21, 29, 37], [43, 44, 45], [29, 37, 45], [44, 45, 46], [37, 45, 53], [45, 46, 47], [45, 53, 61], [48, 49, 50], [6, 14, 22], [49, 50, 51], [14, 22, 30], [50, 51, 52], [22, 30, 38], [51, 52, 53], [30, 38, 46], [52, 53, 54], [38, 46, 54], [53, 54, 55], [46, 54, 62], [56, 57, 58], [7, 15, 23], [57, 58, 59], [15, 23, 31], [58, 59, 60], [23, 31, 39], [59, 60, 61], [31, 39, 47], [60, 61, 62], [39, 47, 55], [61, 62, 63], [47, 55, 63]]


if __name__ == '__main__':
    env = BejeweledEnvironment()
    agent = DQN()
    STEP_NUM = 50
    TEST_ROUND = 1

    action_space = BejeweledAction().action_space
    agent.restore_model()

    for episode in range(1000):
        _state, initial_score = env.reset()
        print("Initial score", initial_score)
        state = _state.state

        print("Episode {} start.".format(episode))

        for step in range(STEP_NUM):
            action, q_values, flag_greedy = agent.greedy_action(state)

            if (flag_greedy):
                action = default_solution(_state.prediction)

                print("Faking action:", action)

            _next_state, reward, done = env.step(action, wait=0.6)
            next_state = _next_state.state
            print("{}#{} Step Action: {}, Reward: {} Greedy: {} eps={}".
                  format(episode, step, BejeweledAction().action_space[action], reward, flag_greedy, agent.epsilon))

            result = env.render()
            tag(result, q_values, action, reward, action_space)

            print(agent.eval_result(state).shape)

            if np.count_nonzero(_next_state.prediction == 0) > 40:
                print("No detection, sleep for 3 seconds.")
                time.sleep(3)

            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            _state = _next_state
            if done:
                break

        # test per 50 episode
        if episode % 50 == 0 and episode > 0:
            total_reward = 0
            for i in range(TEST_ROUND):
                _state, initial_score = env.reset()
                state = _state.state
                for j in range(STEP_NUM):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    _next_state, reward, done = env.step(action)
                    next_state = _next_state.state
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST_ROUND
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)
            if avg_reward >= 10000:
                break

