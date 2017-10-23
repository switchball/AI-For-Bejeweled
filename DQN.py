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
import pickle

from Agent import Agent
from BejeweledEnvironment import *

REPLAY_SIZE = 128000


class DQN(Agent):
    def __init__(
            self,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=100000,
            batch_size=32,
            e_greedy_increment=None,
            double_q=True,
            dueling=True,
            output_graph=False):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        self.dueling = dueling

        # total learning step
        self.learn_step_counter = 0

        # replay memory
        self.replay_buffer = deque()
        self.load_replay()
        self.time_step = 0
        self.action_dim = 2*7*8 + 1

        self.checkpointDir = './model/dqn_model/'

        self.create_Q_network()
        self.create_training_method()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_dqn')
        self.saver = tf.train.Saver(var_list=variables)

        assert self.q_value.graph is tf.get_default_graph()
        init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.q_value.graph)
        self.session.run(init)

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.session.graph)

        self.loss_his = []

    def load_replay(self):
        try:
            with open('replay.dat', 'rb') as f:
                self.replay_buffer = pickle.load(f)
                print('Load Replay buffer size =', len(self.replay_buffer))
        except:
            print('Could not load replay buffer.')

    def create_Q_network(self):
        def build_layers(input_layer, c_name, filter_num=32, fc_num=1024):
            with tf.variable_scope(c_name, reuse=False):
                with tf.variable_scope('Conv'):
                    part_A = tf.slice(input_layer, [0, 0, 0, 0], [-1, 8, 8, 1])
                    part_B = tf.slice(input_layer, [0, 0, 0, 1], [-1, 8, 8, 7])
                    part_C = tf.slice(input_layer, [0, 0, 0, 8], [-1, 8, 8, 1])

                    conv1_A = tf.layers.conv2d(
                        inputs=part_A,
                        filters=filter_num,
                        kernel_size=[3, 3],
                        padding='valid',
                        activation=tf.nn.relu,
                        use_bias=False,
                        name='conv1_A'
                    )
                    conv1_BB = tf.layers.conv3d(
                        inputs=tf.expand_dims(part_B, -1),
                        filters=filter_num,
                        kernel_size=[3, 3, 1],
                        padding='valid',
                        activation=tf.nn.relu,
                        use_bias=True,
                        name='conv1_BB'
                    )
                    conv1_C = tf.layers.conv2d(
                        inputs=part_C,
                        filters=filter_num,
                        kernel_size=[3, 3],
                        padding='valid',
                        activation=tf.nn.relu,
                        use_bias=False,
                        name='conv1_C'
                    )
                    conv1_B = tf.tensordot(conv1_BB, tf.ones(7, tf.float32),
                                           axes=[[3], [0]], name='conv1_B')

                    conv1 = tf.add(tf.add(conv1_A, conv1_C), conv1_B, name='conv1')

                    conv1_flat = tf.reshape(conv1, [-1, 6 * 6 * filter_num])

                if self.dueling:
                    # Dueling DQN
                    with tf.variable_scope('Value'):
                        _fc = tf.layers.dense(inputs=conv1_flat, units=fc_num, activation=tf.nn.relu)
                        self.V = tf.layers.dense(inputs=_fc, units=1)

                    with tf.variable_scope('Advantage'):
                        _fc = tf.layers.dense(inputs=conv1_flat, units=fc_num, activation=tf.nn.relu)
                        self.A = tf.layers.dense(inputs=_fc, units=self.action_dim)

                    with tf.variable_scope('Q'):
                        out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
                else:
                    with tf.variable_scope('Q'):
                        fc = tf.layers.dense(inputs=conv1_flat, units=fc_num)
                        out = tf.layers.dense(inputs=fc, units=self.action_dim)

                q_value = out
                return q_value

        input_layer = tf.placeholder(tf.float32, [None, 8, 8, 9], name='s')
        input_layer_ = tf.placeholder(tf.float32, [None, 8, 8, 9], name='s_')
        action_input = tf.placeholder(tf.int32, [None,], name='a')
        reward_input = tf.placeholder(tf.float32, [None], name='r')
        self.s = input_layer
        self.s_ = input_layer_
        self.a = action_input
        self.r = reward_input

        filter_num = 32
        fc_num = 1024

        self.q_value = build_layers(input_layer, 'eval_dqn')
        self.q_next = build_layers(input_layer_, 'target_dqn')

    def create_training_method(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('q_target'):
            if self.double_q:
                ind = tf.argmax(self.q_value, axis=1)
                inds = tf.stack([tf.range(tf.shape(ind)[0], dtype=tf.int32), tf.cast(ind, tf.int32)], axis=1)
                selected_q_next = tf.gather_nd(params=self.q_next, indices=inds, name='Q_maxQ_s_')  # shape=(None, )
            else:
                selected_q_next = tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            q_target = self.r + self.gamma * selected_q_next  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_value, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpointDir)
        if ckpt:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("[DQN Prepare] Model " + ckpt.model_checkpoint_path + " restored.")
        else:
            print("[DQN Prepare] Model not found at", self.checkpointDir)

    def perceive(self, state, action, reward, next_state, done):
        #one_hot_action = np.zeros(self.action_dim)
        #one_hot_action[action] = 1

        ### Permutation on state and next_state
        _ind = [0]+list(np.random.permutation([1,2,3,4,5,6,7]))+[8]
        state_p = np.swapaxes(np.swapaxes(state, 0, 2)[_ind], 0, 2)
        next_state_p = np.swapaxes(np.swapaxes(next_state, 0, 2)[_ind], 0, 2)
        ### Permutation finished

        self.replay_buffer.append((state_p, action, reward, next_state_p, done))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

        if len(self.replay_buffer) % 20 == 0:
            with open('replay.dat', 'wb') as f:
                pickle.dump(self.replay_buffer, f, True)
                print("Replay save. ", len(self.replay_buffer))

    def train_Q_network(self):
        # Step 1: check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            print('\ntarget_params_replaced:', self.learn_step_counter)
            if self.learn_step_counter > 0:
                self.saver.save(self.session, self.checkpointDir + 'model.ckpt', global_step=self.global_step)
                print('[DQN Model] model saved. last batch loss =', np.average(self.loss_his[-self.replace_target_iter:-1]))
            else:
                print('[DQN Model] training start!')
        # Step 2: obtain random mini_batch from replay memory
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]

        # Step 3: calculate y
        y_batch = []
        Q_value_batch = self.q_value.eval(session=self.session,
                                          feed_dict={self.s: next_state_batch})
        for i in range(0, self.batch_size):
            done = mini_batch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))

        # Step 3: optimize (ignore done, assume done=False)
        _, loss = self.session.run([self._train_op, self.loss], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.r: reward_batch,
            self.s_: next_state_batch
        })

        self.learn_step_counter += 1

        self.loss_his.append(loss)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def eval_conv_result(self, state):
        v = self.conv1.eval(session=self.session, feed_dict={
            self.s: [state]
        })[0]
        return v

    def eval_kernel_result(self):
        with tf.variable_scope('dqn', reuse=True):
            kernel = tf.get_variable('conv1_BB/kernel')
            bias = tf.get_variable('conv1_BB/bias')
            ks, bs = self.session.run([tf.squeeze(kernel), bias])
            return ks, bs

    def greedy_action(self, state):
        Q_value = self.q_value.eval(session=self.session, feed_dict={
            self.s: [state]
        })[0]

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1), Q_value, 1
        else:
            return np.argmax(Q_value), Q_value, 0

    def action(self, state):
        Q_value = self.q_value.eval(session=self.session, feed_dict={
            self.s: [state]
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
    print('qv, max={}, min={}, avg={}, #Zeros={}, action={}'.
          format(maximum, minimum, average, nonzero, action_space[action]))
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
            color = (0, 0, int(qv/maximum*255)) if maximum > 0 else (0,0,0)
            thickness = int(qv/maximum*5) if maximum > 0 else 0
        else:
            color = (0, int(qv/minimum*255), 0) if minimum < 0 else (0,0,0)
            thickness = max(int(qv/minimum*5),0) if minimum < 0 else 0

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

def tag_conv(conv):
    size = 40
    img = np.zeros((size*6+60, size*6, 3), np.uint8)
    # conv.shape = (6, 6, 16)
    tm = np.max(conv)
    try:
        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                for k in range(conv.shape[2]):
                    base_x, base_y = j*size, i*size
                    pad_x, pad_y = 11+(k%4)*5, 11+int(k/4)*5
                    maximum = np.max(conv[i,j])
                    color = (0,0,int(conv[i,j,k]/maximum*255))
                    cv2.rectangle(img, (base_x + pad_x, base_y + pad_y),
                                  (base_x + pad_x + 3, base_y + pad_y + 3),
                                  color, -1)
                    thickness = int(3*maximum/tm)
                    if thickness > 0:
                        cv2.rectangle(img, (base_x+4, base_y+4),
                                      (base_x + size-4, base_y + size-4),
                                      (0, 255, 0), thickness)
    except Exception as e:
        print(e)

    cv2.putText(img, '%s' % tm, (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.imshow('conv1', img)
    cv2.moveWindow('conv1', 1024+168, 0)
    cv2.waitKey(1)

def tag_kernel(kernel_v, bias_v):
    size = 64
    img = np.zeros(( 2*(size+60), size*8, 3), np.uint8)
    tm = np.max(kernel_v)
    try:
        for k in range(kernel_v.shape[2]):
            maximum = np.max(kernel_v[:, :, k])
            minimum = np.min(kernel_v[:, :, k])
            mm = max(maximum, -minimum) + 0.001
            thickness = int(3 * maximum / tm)
            base_x, base_y = (k % 8) * size, int(k / 8) * (size + 60)
            for i in range(kernel_v.shape[0]):
                for j in range(kernel_v.shape[1]):
                    pad_x, pad_y = 13+j*14, 13+i*14
                    v = kernel_v[i,j,k]
                    if v >= 0:
                        color = (0, 0, int(v / mm * 255))
                    else:
                        color = (int(-v / mm * 255), 0, 0)
                    cv2.rectangle(img, (base_x + pad_x, base_y + pad_y),
                                  (base_x + pad_x + 10, base_y + pad_y + 10),
                                  color, -1)
                    if thickness > 0:
                        cv2.rectangle(img, (base_x+7, base_y+9),
                                      (base_x + size-9, base_y + size-9),
                                      (0, 255, 0), thickness)
            cv2.putText(img, '%.3f' % maximum, (10+base_x, 80+base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), thickness)
            cv2.putText(img, '%.3f' % bias_v[k], (10+base_x, 100+base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
    except Exception as e:
        print(e)

    cv2.imshow('kernel', img)
    cv2.moveWindow('kernel', 550, 512+20)
    cv2.waitKey(1)

if __name__ == '__main__':
    env = BejeweledEnvironment()
    agent = DQN(n_actions=2*7*8+1, e_greedy=0.8, output_graph=True)
    STEP_NUM = 200
    TEST_ROUND = 0

    action_space = BejeweledAction().action_space
    agent.restore_model()

    from AdaptiveRec import AdaptiveRec
    AR = AdaptiveRec()

    wait_round = 0

    for episode in range(500):
        _state, initial_score = env.reset()
        state = _state.state

        print("Episode {} start.".format(episode))
        total_reward = 0

        for step in range(STEP_NUM):
            result = env.render(show=False)

            action, q_values, flag_greedy = agent.greedy_action(state)

            if ((flag_greedy and random.random() < 0.6) or total_reward < 2):
                action = default_solution(_state.prediction)

            _next_state, reward, done = env.step(action, wait=1.3)
            next_state = _next_state.state
            total_reward += reward
            #print("{}#{} Step Action: {}, Reward: {} Greedy: {} eps={}".
            #      format(episode, step, BejeweledAction().action_space[action], reward, flag_greedy, agent.epsilon))

            # render before
            tag(result, q_values, action, reward, action_space)

            #conv = agent.eval_conv_result(state)
            #tag_conv(conv)

            #k_v, b_v = agent.eval_kernel_result()
            #tag_kernel(k_v, b_v)

            #AR.append(env.last_image, _state.prediction)
            #AR.show()
            # _ = env.last_image # get sprite img

            if np.count_nonzero(_next_state.prediction == 0) > 30:
                print("No detection, sleep for 4 seconds.")
                time.sleep(4)
                if wait_round > 2:
                    env.mouse_click_on_sprite(7, 2)
                    env.reset()
                    wait_round = 0
                    total_reward = 0
                else:
                    wait_round += 1
            else:
                wait_round = 0

            if total_reward > 2:
                agent.perceive(state, action, reward, next_state, done)
            state = next_state
            _state = _next_state
            if done:
                break

        avg_reward = total_reward
        print('episode: ', episode, 'Evaluation Average Reward:', avg_reward, "Greedy:", agent.epsilon)

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

