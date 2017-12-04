from collections import deque
import numpy as np
import random as rnd
import tensorflow as tf
import cv2
from ..cytomatrix import main


# HYPERPARAMETERS #
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

def create_graph():
    # Convolutional layers and bias vectors
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # Input for game pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Computes RELU activation on a 2D convolution given 4D input and filter tensors
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "VALID") + b_conv3)
    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

def train_graph(inp, outp, session):
    # Argmax to later select action with highest Q value
    argmax = tf.placeholder("float", [None, ACTIONS])
    # Ground truth
    gt = tf.placeholder("float", [None])
    # Pick a single action based on max Q value
    action = tf.reduce_sum(tf.mul(output, argmax), reduction_indices=1)
    # Cost function for backprop
    cost = tf.reduce_mean(tf.square(action - gt))
    # Optimization step taken to reduce the cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Init game
    game = main.Game()
    while True:
        game.new()
        game.run()
        # Init experience replay buffer
        deq = deque()
        # Intial frame
        game_data = game.get_rl_input()
        frame = game_data[0]
        # RGB to grayscale
        frame = cv2.cvtColor(cv2.resize(frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        # Binarize
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        # Stack frames to form input tensor
        inp_tensor = np.stack((frame, frame, frame, frame), axis = 2)

        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        t = 0
        epsilon = INITIAL_EPSILON

    while True:
        out_tensor = outp.eval(feed_dict = {inp : [inp_tensor]})[0]
        argmax_tensor = np.zeros([ACTIONS])
        # Pick a random action with probability epsilon
        if rnd.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        # Else go with the prescribed action based on max Q value
        else:
            maxIndex = np.argmax(out_tensor)
        argmax_tensor[maxIndex] = 1
        # Slowly decrease this random exploration over time
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Get the next observation and reward
        frame, _, reward = game.get_rl_input(argmax_tensor)

        # Preprocess
        frame = cv2.cvtColor(cv2.resize(frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (100, 100, 1))
        # New input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)

        # Add input tensor, argmax tensor, reward and updated input tensor to replay buffer
        deq.append((inp_t, argmax_t, reward, inp_t1))

        # If replay buffer is full, erase the oldest memory
        if len(deq) > REPLAY_MEMORY:
            deq.popleft()

        if t > OBSERVE:
            # Sample random batches of memories
            minibatch = rnd.sample(deq, BATCH)

            # Group the components of the batch
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp : inp_t1_batch})

            # Add values to the batch based on outcomes
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))



            # Train on that sequence
            train_step.run(feed_dict={
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })

        # Update input tensor to the next frame
        inp_t = inp_t1
        t += 1

        # Print training progress
        if t % 10000 == 0:
            saver.save(session, './' + 'cytomata' + '-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))

def run():
    # Create session
    sess = tf.InteractiveSession()
    # Input and output layer
    inp, out = create_graph()
    train_graph(inp, out, sess)
