import os
import sys
import random as rnd
from collections import deque

import numpy as np
import cv2
import tensorflow as tf
import cytomatrix as cym


class DQN(object):
    def __init__(self, game, num_actions, gamma, observe_len,
        explore_len, init_epsilon, final_epsilon, memory_size,
        batch_size, frames_per_action
        ):
        self.game = game
        self.num_actions = num_actions
        self.gamma = gamma
        self.observe_len = observe_len
        self.explore_len = explore_len
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.frames_per_action = frames_per_action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_network(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
        W_fc2 = self.weight_variable([512, self.num_actions])
        b_fc2 = self.bias_variable([self.num_actions])

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, readout, h_fc1

    def train_network(self, game_state, s, readout, h_fc1, sess):
        # define the cost function
        a = tf.placeholder("float", [None, self.num_actions])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # open up a game state to communicate with emulator
        game_state.new()

        # store the previous observations in replay memory
        D = deque()

        # printing
        # os.makedirs(os.path.join("logs_" + GAME), exist_ok=True)
        # a_file = open("logs_" + GAME + "/readout.txt", 'w')
        # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(self.num_actions)
        do_nothing[0] = 1
        x_t, _, r_0, playing = game_state.run(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # start training
        epsilon = self.init_epsilon
        t = 0
        while True:
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            if t % self.frames_per_action == 0:
                a_t = np.zeros([self.num_actions])
                action_index = 0
                if rnd.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = rnd.randrange(self.num_actions)
                    a_t[rnd.randrange(self.num_actions)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1

            # run the selected action and observe next state and reward
            x_t1_colored, _, r_t, playing = game_state.run(a_t)
            if not playing:
                game_state.new()

            # process the new state
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            if t % self.frames_per_action == 0:
                # store the transition in D
                D.append((s_t, a_t, r_t, s_t1, playing))
                if len(D) > self.memory_size:
                    D.popleft()

            # only train if done observing
            if t > self.observe_len:
                # sample a minibatch to train on
                minibatch = rnd.sample(D, self.batch_size)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
                for i in range(0, len(minibatch)):
                    playing = minibatch[i][4]
                    # if terminal, only equals reward
                    if not playing:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + self.gamma * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch
                })

            # scale down epsilon
            if epsilon > self.final_epsilon and t > self.observe_len:
                epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore_len

            # take timestep forward
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + self.game + '-dqn', global_step=t)

            # print info
            state = ""
            if t <= self.observe_len:
                state = "observe"
            elif t > self.observe_len and t <= self.observe_len + self.explore_len:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP:", t, "| STATE:", state, "| EPSILON:", np.around(epsilon, 4),
                "| ACTION:", action_index, "| REWARD:", np.around(r_t, 3), "| Q_AVE:", np.around(np.average(readout_t), 4)
                )
            # write info to files
            # if t % 10000 <= 100:
            #     a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #     h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #     cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)


def run():
    sess = tf.InteractiveSession()
    env = cym.Game()
    dqn = DQN(game='cytomatrix', num_actions=5, gamma=0.99, observe_len=50000,
        explore_len=2000000, init_epsilon=1.0, final_epsilon=0.02, memory_size=50000,
        batch_size=32, frames_per_action=1)
    s, readout, h_fc1 = dqn.create_network()
    dqn.train_network(env, s, readout, h_fc1, sess)
