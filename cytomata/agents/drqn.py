import os
import csv
import random as rnd

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cytomatrix as cym


def process_state(state):
    x = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(x, (-1, 84, 84, 1))


def prep_update_target(tf_vars, tau):
    """Prep operations to copy parameters from primary to target network"""
    num_vars = len(tf_vars)
    update_ops = []
    for idx, var in enumerate(tf_vars[0: num_vars // 2]):
        update_ops.append(tf_vars[idx + num_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tf_vars[idx + num_vars // 2].value())))
    return update_ops


def apply_update_target(update_ops, sess):
    """Run operations to copy parameters from primary to target network
    and check to see if it worked"""
    for op in update_ops:
        sess.run(op)
    num_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[num_vars // 2].eval(session=sess)
    if a.all() == b.all():
        return True
    else:
        return False


class DRQN():
    def __init__(self, h_size, rnn_cell, my_scope):
        """Create the DRQN tensorflow graph"""
        # Format input
        self.image_in = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32)
        # Construct the convolution layers
        self.conv1 = slim.convolution2d(
            inputs=self.image_in, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None, scope=my_scope + '_conv1')
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None, scope=my_scope + '_conv2')
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=my_scope + '_conv3')
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=h_size,
            kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=my_scope + '_conv4')
        # Send conv4 output into recurrent layer
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        # The RNN requires input as [batch x trace x units]
        self.conv_flat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.train_length, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.conv_flat, cell=rnn_cell, dtype=tf.float32,
            initial_state=self.state_in, scope=my_scope + '_rnn')
        # Return back to [batch x units] after RNN processing
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        self.AW = tf.Variable(tf.random_normal([h_size // 2, 4]))
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        self.salience = tf.gradients(self.Advantage, self.image_in)
        # Combine A and V to get final Q-values.
        self.Q_out = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_out, 1)
        # SSE loss function of target and estimated Q values.
        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)
        self.est_Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.target_Q - self.est_Q)
        # Lample & Chatlot 2016: Propagate only accurate gradients through the network
        # Mask the first half of the losses for each trace
        self.maskA = tf.zeros([self.batch_size, self.train_length // 2])
        self.maskB = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        # Optimize model
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.load = 0
        self.buffer_size = buffer_size

    def add(self, experience):
        # Erase oldest exp sequence if buffer is full
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0: (1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)
        self.load += 1

    def sample(self, batch_size, trace_length):
        sampled_episodes = rnd.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point: point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 5])


def train_network(env, num_actions=5, batch_size=4, trace_length=8, update_freq=5, gamma=0.99,
    start_E=1.0, end_E=0.05, annealing_steps=1000000, num_episodes=100000, pretrain_steps=100000, load_model=True,
    save_path='./drqn', h_size=512, max_ep_length=5000, time_per_step=1, summary_length=1, tau=0.001):
    # batch_size: How many experience traces to use for each training step.
    # trace_length: How long each experience trace will be when training
    # update_freq: How often to perform a training step.
    # gamma: Discount factor on the target Q-values
    # start_E: Starting chance of random action
    # end_E: Final chance of random action
    # annealing_steps: How many steps of training to reduce start_E to end_E.
    # num_episodes: How many episodes of game environment to train network with.
    # pretrain_steps: How many steps of random actions before training begins.
    # load_model: Whether to load a saved model.
    # path: The path to save our model to.
    # h_size: The size of the final convolutional layer before splitting it into Advantage and Value streams.
    # max_ep_length: The max allowed length of our episode.
    # time_per_step: Length of each step used in gif creation
    # summary_length: Number of episodes to periodically save for analysis
    # tau: Update the target network by small steps
    tf.reset_default_graph()
    # Define cells for primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    main_QN = DRQN(h_size, cell, 'main')
    target_QN = DRQN(h_size, cellT, 'target')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)
    trainables = tf.trainable_variables()
    target_ops = prep_update_target(trainables, tau)
    exp_buffer = experience_buffer()

    # Set the rate of random action decrease
    epsilon = start_E
    step_drop = (start_E - end_E) / annealing_steps

    # Steps per episode
    jlist = []
    # Reward per episode
    rlist = []
    # Ave Q-value per episode
    qlist = []
    total_steps = 0

    # Make a path for our model to be saved in.
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs('./center', exist_ok=True)
    # Write the first line of the master log-file for the Control Center
    # with open('./center/log.csv', 'w') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

    # New session
    with tf.Session() as sess:
        if load_model:
            ckpt = tf.train.get_checkpoint_state(save_path)
            print('Loading Model: ' + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)
        # Set the target network to be equal to the primary network.
        apply_update_target(target_ops, sess)
        for i in range(num_episodes):
            episode_buffer = []
            # Reset environment and get first observation
            gstate = env.reset()
            s = process_state(gstate)
            # Reset the recurrent layer's hidden state
            state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            # env episode has not terminated
            d = True
            # Total rewards
            r = 0
            # Q-values
            q_ep = []
            # Episode steps
            j = 0
            # Training loop
            while j < max_ep_length:
                j += 1
                # Choose an action by epsilon-greedy from the Q-network
                if np.random.rand(1) < epsilon or total_steps < pretrain_steps:
                    state1 = sess.run(main_QN.rnn_state,
                        feed_dict={main_QN.image_in: s, main_QN.train_length: 1,
                        main_QN.state_in: state, main_QN.batch_size: 1})
                    a = np.random.randint(0, num_actions)
                else:
                    a, state1 = sess.run([main_QN.predict, main_QN.rnn_state],
                        feed_dict={main_QN.image_in: s, main_QN.train_length: 1,
                        main_QN.state_in: state, main_QN.batch_size: 1})
                    a = a[0]
                # Act on environment and fetch new state
                gstate1, r, d = env.step(a)
                s1 = process_state(gstate1)
                total_steps += 1
                # Add sequence to experience replay buffer
                episode_buffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                # Slowly decrease random actions
                if total_steps > pretrain_steps:
                    if epsilon > end_E:
                        epsilon -= step_drop

                    if total_steps % update_freq == 0:
                        apply_update_target(target_ops, sess)
                        # Reset the recurrent layer's hidden state
                        state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
                        # Get a random batch of experiences.
                        train_batch = exp_buffer.sample(batch_size, trace_length)
                        # Perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(main_QN.predict, feed_dict={
                            main_QN.image_in: np.vstack(train_batch[:, 3]),
                            main_QN.train_length: trace_length, main_QN.state_in: state_train,
                            main_QN.batch_size: batch_size})
                        Q2 = sess.run(target_QN.Q_out, feed_dict={
                            target_QN.image_in: np.vstack(train_batch[:, 3]),
                            target_QN.train_length: trace_length, target_QN.state_in: state_train,
                            target_QN.batch_size: batch_size})
                        end_multiplier = -(train_batch[:, 4] - 1)
                        double_Q = Q2[range(batch_size * trace_length), Q1]
                        target_Q = train_batch[:, 2] + (gamma * double_Q * end_multiplier)
                        # Update the network with our target values.
                        sess.run(main_QN.update_model,
                            feed_dict={main_QN.image_in: np.vstack(train_batch[:, 0]),
                            main_QN.target_Q: target_Q, main_QN.actions: train_batch[:, 1],
                            main_QN.train_length: trace_length, main_QN.state_in: state_train,
                            main_QN.batch_size: batch_size})
                        q_ep.append(Q1)
                # Move forward to new state
                s = s1
                gstate = gstate1
                state = state1
                if not d:
                    break

            # Add the episode to the experience buffer
            buffer_array = np.array(episode_buffer)
            episode_buffer = list(zip(buffer_array))
            exp_buffer.add(episode_buffer)
            jlist.append(j)
            rlist.append(r)
            qlist.append(np.mean(q_ep))

            # Periodically save the model.
            if i % 5 == 0 and i != 0:
                saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
                print ("Saved Model")
            if len(rlist) % summary_length == 0 and len(rlist) != 0:
                print("Totals Steps:", total_steps, "| Buffer Load:", exp_buffer.load,
                    "| EPSILON:", np.around(epsilon, 4),
                    "| REWARD:", np.around(np.mean(rlist[-summary_length:]), 3),
                    "| Q_AVE:", np.around(np.mean(qlist[-summary_length:]), 3))
                # save_to_center(i, rlist, jlist, np.reshape(np.array(episode_buffer), [len(episode_buffer), 5]),
                #     summary_length, h_size, sess, main_QN, time_per_step)
        saver.save(sess, save_path + '/model-' + str(i) + '.cptk')


def use_network(env, epsilon=0.01, num_episodes=10000, load_model=True, save_path='./drqn',
    h_size=512, max_ep_length=50, time_per_step=1, summary_length=100):
    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    main_QN = DRQN(h_size, cell, 'main')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)

    # Total steps per episode
    jlist = []
    # Total rewards per episode
    rlist = []
    total_steps = 0

    # Make a path for our model to be saved in.
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./center', exist_ok=True)

    # Write the first line of the master log-file for the Control Center
    with open('./center/log.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

    with tf.Session() as sess:
        if load_model:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        for i in range(num_episodes):
            episode_buffer = []
            # Reset environment and get first new observation
            gstate = env.reset()
            s = process_state(gstate)
            d = True
            r_all = 0
            j = 0
            state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            # The Q-Network
            while j < max_ep_length:
                j += 1
                # Choose an action by greedy epsilon policy
                if np.random.rand(1) < epsilon:
                    state1 = sess.run(main_QN.rnn_state,
                        feed_dict={main_QN.image_in: s,
                        main_QN.train_length: 1, main_QN.state_in: state, main_QN.batch_size: 1})
                    a = np.random.randint(0, 4)
                else:
                    a, state1 = sess.run([main_QN.predict, main_QN.rnn_state],
                        feed_dict={main_QN.image_in: s, main_QN.train_length: 1,
                        main_QN.state_in: state, main_QN.batch_size: 1})
                    a = a[0]
                gstate1, r, d = env.step(a)
                s1 = process_state(gstate1)
                total_steps += 1
                episode_buffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                r_all += r
                s = s1
                gstate = gstate1
                state = state1
                if not d:
                    break

            buffer_array = np.array(episode_buffer)
            jlist.append(j)
            rlist.append(r_all)

            # Periodically save the model
            if len(rlist) % summary_length == 0 and len(rlist) != 0:
                print(total_steps, np.mean(rlist[-summary_length:]), epsilon)
                # save_to_center(i, rlist, jlist, np.reshape(buffer_array, [len(episode_buffer), 5]),
                #     summary_length, h_size, sess, main_QN, time_per_step)
    print("Percent of succesful episodes: " + str(sum(rlist) / num_episodes) + "%")


# Record performance metrics and episode logs for the Control Center.
def save_to_center(i, rlist, jlist, buffer_array, summary_length, h_size, sess, main_QN, time_per_step):
    with open('./center/log.csv', 'a') as myfile:
        state_display = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        images_s = []
        for idx, z in enumerate(np.vstack(buffer_array[:, 0])):
            img, state_display = sess.run([main_QN.salience, main_QN.rnn_state],
                feed_dict={main_QN.image_in: np.reshape(buffer_array[idx, 0], [1, 7056]),
                main_QN.train_length: 1, main_QN.state_in: state_display, main_QN.batch_size: 1})
            images_s.append(img)
        images_s = (images_s - np.min(images_s)) / (np.max(images_s) - np.min(images_s))
        images_s = np.vstack(images_s)
        images_s = np.resize(images_s, [len(images_s), 84, 84, 3])
        luminance = np.max(images_s, 3)
        images_s = np.multiply(np.ones([len(images_s), 84, 84, 3]), np.reshape(luminance, [len(images_s), 84, 84, 1]))
        make_gif(np.ones([len(images_s), 84, 84, 3]), './center/frames/sal' + str(i) + '.gif',
            duration=len(images_s) * time_per_step, true_image=False, salience=True, salIMGS=luminance
            )
        images = zip(buffer_array[:, 0])
        images.append(buffer_array[-1, 3])
        images = np.vstack(images)
        images = np.resize(images, [len(images), 84, 84, 3])
        make_gif(images, './center/frames/image' + str(i) + '.gif',
            duration=len(images) * time_per_step, true_image=True, salience=False
            )
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i, np.mean(jlist[-100:]), np.mean(rlist[-summary_length:]),
            './frames/image' + str(i) + '.gif', './frames/log' + str(i) + '.csv', './frames/sal' + str(i) + '.gif']
            )
        myfile.close()
    with open('./center/frames/log' + str(i) + '.csv', 'w') as myfile:
        state_train = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['ACTION', 'REWARD', 'A0', 'A1', 'A2', 'A3', 'V'])
        a, v = sess.run([main_QN.Advantage, main_QN.Value],
            feed_dict={main_QN.image_in: np.vstack(buffer_array[:, 0]),
            main_QN.train_length: len(buffer_array), main_QN.state_in: state_train, main_QN.batch_size: 1
            })
        wr.writerows(zip(buffer_array[:, 1], buffer_array[:, 2], a[:, 0], a[:, 1], a[:, 2], a[:, 3], v[:, 0]))


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except Exception:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except Exception:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        # clipB = clip.set_mask(mask).set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
        # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False)


def run():
    env = cym.Game()
    train_network(env)
