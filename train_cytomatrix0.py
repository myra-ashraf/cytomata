import os
import sys
import time
import random
import datetime
import threading
from absl import flags
import numpy as np

import gym
from baselines import deepq
# from pysc2.env import sc2_env # TODO
# from pysc2.lib import actions # TODO
# from pysc2.env import environment # TODO
from baselines.a2c import a2c
from baselines.a2c.policies import CnnPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv # TODO
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import cytomata
from cytomata.wrappers import wrap_cytomatrix
# import deepq_cytomatrix_shards # TODO


# _MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
# _SELECT_ARMY = actions.FUNCTIONS.select_army.id
# _SELECT_ALL = [0]
# _NOT_QUEUED = [0]
ACTIONS = range(0, 5)

FLAGS = flags.FLAGS
# flags.DEFINE_string("map", "CollectcytomatrixShards",
#                     "Name of a map to use to play.")
flags.DEFINE_string('log', 'stdout', 'Log via stdout or tensorboard.')
flags.DEFINE_string('algorithm', 'deepq', 'RL algorithm to use.')
flags.DEFINE_integer('timesteps', 2000000, 'Steps to train.')
flags.DEFINE_float('exploration_fraction', 0.2, 'Fraction of training time to act randomly.')
flags.DEFINE_boolean('prioritized', True, 'Use prioritized replay buffer.')
flags.DEFINE_boolean('dueling', True, 'Use dueling DQN variant.')
flags.DEFINE_float('lr', 0.0005, 'Learning rate.')
flags.DEFINE_integer('num_agents', 4, 'Number of RL agents for A2C.')
flags.DEFINE_integer('num_scripts', 4, 'Number of script agents for A2C.')
flags.DEFINE_integer('nsteps', 20, 'Number of batch steps for A2C.')

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = -100
last_filename = ''

start_time = datetime.datetime.now().strftime('%Y%m%d%H%M')


def main():
    FLAGS(sys.argv)
    print('algorithm: %s' % FLAGS.algorithm)
    print('timesteps: %s' % FLAGS.timesteps)
    print('exploration_fraction: %s' % FLAGS.exploration_fraction)
    print('prioritized: %s' % FLAGS.prioritized)
    print('dueling: %s' % FLAGS.dueling)
    print('num_agents: %s' % FLAGS.num_agents)
    print('lr: %s' % FLAGS.lr)

    if (FLAGS.lr == 0):
        FLAGS.lr = random.uniform(0.00001, 0.001)
        print('random lr: %s' % FLAGS.lr)

    lr_round = round(FLAGS.lr, 8)

    logdir = 'tensorboard'

    if (FLAGS.algorithm == 'deepq'):
        logdir = 'tensorboard/cytomatrix/%s/%s_%s_prio%s_duel%s_lr%s/%s' % (
        FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
        FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)

    if (FLAGS.log == 'tensorboard'):
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=None,
            output_formats=[TensorBoardOutputFormat(logdir)])
    elif (FLAGS.log == 'stdout'):
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=None,
            output_formats=[HumanOutputFormat(sys.stdout)])

    if (FLAGS.algorithm == 'deepq'):
        env = gym.make('Cytomatrix-v0')
        env = wrap_cytomatrix(env)

        model = deepq.models.cnn_to_mlp(
            convs=[(16, 8, 4), (32, 4, 2)],
            hiddens=[256],
            dueling=True)

        act = deepq.learn( # TODO
            env,
            q_func=model,
            # num_actions=len(ACTIONS),
            lr=FLAGS.lr,
            max_timesteps=FLAGS.timesteps,
            buffer_size=10000,
            exploration_fraction=FLAGS.exploration_fraction,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=True,
            print_freq=1,
            callback=deepq_callback)
        act.save('cytomatrix_model.pkl')


def deepq_callback(locals, globals):
    global max_mean_reward, last_filename
    if ('done' in locals and locals['done'] == True):
        if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 1
            and locals['mean_100ep_reward'] > max_mean_reward):

            # print('mean_100ep_reward : %s max_mean_reward : %s' % (
            #     locals['mean_100ep_reward'], max_mean_reward))

            os.makedirs(os.path.join(PROJ_DIR, 'models/deepq/'), exist_ok=True)

            if (last_filename != ''):
                os.remove(last_filename)
                # print('Deleted last model file: %s' % last_filename)

            max_mean_reward = locals['mean_100ep_reward']

            act = deepq.simple.ActWrapper(locals['act'], locals['act_params'])
            filename = os.path.join(PROJ_DIR,
                'models/deepq/cytomatrix_%s.pkl' % locals['mean_100ep_reward'])
            act.save(filename)

            print('Save best mean_100ep_reward model to %s' % filename)
            last_filename = filename
            max_mean_reward = locals['mean_100ep_reward']

if __name__ == '__main__':
    main()
