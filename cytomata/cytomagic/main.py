from collections import deque
import numpy as np
import random as rnd
import tensorflow as tf
import cv2
import cytomatrix

# HYPERPARAMETERS--------------------
# UP, DOWN, LEFT, RIGHT, NO_OP
ACTIONS = 5
# Learning rate
GAMMA = 0.99
# Epsilon greedy exploration
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# Number of frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
# Batch size
BATCH = 100

class Director(object):
    def __init__(self):
        pass

    def create_dqn(self):
        # Overall network structure
        W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
        b_conv1 = tf.Variable(tf.zeros([32]))
        W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
        b_conv2 = tf.Variable(tf.zeros([64]))
        W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
        b_conv3 = tf.Variable(tf.zeros([64]))
        W_fc4 = tf.Variable(tf.zeros([1600, 512]))
        b_fc4 = tf.Variable(tf.zeros([512]))
        W_fc5 = tf.Variable(tf.zeros([512, ACTIONS]))
        b_fc5 = tf.Variable(tf.zeros([ACTIONS]))
        # Input: preprocessed game pixel data
        s = tf.placeholder("float", [None, 80, 80, 4])
        # Forward propagation calculations
        conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "SAME") + b_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1], padding = "SAME") + b_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "SAME") + b_conv3)
        conv3_flat = tf.reshape(conv3, [-1, 1600])
        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
        # Output: vector of Q-values for each action
        fc5 = tf.matmul(fc4, W_fc5) + b_fc5
        return s, fc5

    def train_dqn(self, input_layer, output_layer, session):
        # Init recommended actions vector
        q_vals = tf.placeholder("float", [None, ACTIONS])
        # Ground truth value
        gt = tf.placeholder("float", [None])
        # Reduce to single action from q_vals
        action = tf.reduce_sum(tf.multiply(output_layer, q_vals), reduction_indices=1)
        # Cost function for backprop
        cost = tf.reduce_mean(tf.square(action - gt))
        # Minimize cost function
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # Open up a new game state
        game = cytomatrix.Game()
        game.new()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        frame, _, reward, terminal = game.run(do_nothing)
        # Preprocess raw pixel inputs
        # RGB to grayscale
        frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        # Binarize image
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        # Form a stack of 4 images as input
        input_tensor = np.stack((frame, frame, frame, frame), axis=2)

        # Init relevant objects
        # Experience replay buffer
        replay_buffer = deque()
        # For saving training progress
        saver = tf.train.Saver()
        # Init session vars
        session.run(tf.global_variables_initializer())
        epsilon = INITIAL_EPSILON
        t = 0

        # Exploration and training loop
        while True:
            output_tensor = output_layer.eval(feed_dict={input_layer : [input_tensor]})[0]
            argmax_tensor = np.zeros([ACTIONS])
            # Epsilon greedy exploration
            if(rnd.random() <= epsilon):
                # Make random actions with probability epsilon
                action_index = rnd.randrange(ACTIONS)
            else:
                # Follow recommended action with probability 1-epsilon
                action_index = np.argmax(output_tensor)
            argmax_tensor[action_index] = 1
            # Slowly decrease random exploration over time
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # Preprocess the next frame
            frame, _, reward, terminal = game.run(argmax_tensor)
            frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
            frame = np.reshape(frame, (80, 80, 1))
            input_tensor1 = np.append(frame, input_tensor[:, :, 0:3], axis=2)

            # Add memory to replay buffer
            replay_buffer.append((input_tensor, argmax_tensor, reward, input_tensor1))

            # Erase oldest memory sequence if replay buffer is full
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            # Once enough memories collected in replay buffer - start to train on it
            if t > OBSERVE:
                # Get batches of data from the replay buffer
                minibatch = rnd.sample(replay_buffer, BATCH)
                # Sort batch data into respective types
                input_batch = [d[0] for d in minibatch]
                argmax_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                input1_batch = [d[3] for d in minibatch]
                gt_batch = []
                out_batch = out.eval(feed_dict={input_layer : input1_batch})
                # Calculate ground truth as expected reward (bellman equation)
                for i in range(0, len(minibatch)):
                    gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))
                # Train network
                train_step.run(feed_dict={gt : gt_batch, argmax : argmax_batch, input_layer : input_batch})

            # Transition to the next state
            input_tensor_t = input_tensor1
            t += 1

            # Save and print training progress/stats to screen
            if t % 10000 == 0:
                saver.save(session, './cytomatrix-dqn', global_step=t)
            print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, "/ Q_MAX %e" % np.max(output_tensor))
        game.quit()


def run():
    # Create session
    sess = tf.InteractiveSession()
    # Input and output layer
    director = Director()
    inp, out = director.create_dqn()
    director.train_dqn(inp, out, sess)
