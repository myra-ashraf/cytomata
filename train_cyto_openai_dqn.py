import re
import os
import glob
import argparse
import time

import gym
from baselines import deepq, bench, logger
from baselines.common import set_global_seeds

import cytomata
from cytomata.wrappers import wrap_cytomatrix
from cytomata.agents import dqn


max_mean_reward = -100
last_filename = ''


def main():
    game = 'Cytomatrix-v0'
    now = time.strftime('_%Y-%m-%d-%H-%M-%S')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(proj_dir, 'models/deepq/')
    log_dir = os.path.join(proj_dir, 'models/deepq/logs/', game + now)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(proj_dir, 'models', 'deepq', game + '_*.pkl')
    try:
        last_model = sorted(glob.glob(model_path))[-1]
        last_filename = last_model
        max_mean_reward = float(re.findall(r"[-+]?\d*\.\d+|\d+", last_model)[-1])
    except IndexError:
        last_model = None

    logger.configure(log_dir)
    # set_global_seeds(0)

    env = gym.make(game)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_cytomatrix(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True

    act = dqn.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        print_freq=1,
        callback=callback,
        load_state=last_model)

    model_fname = game + '_' + now + '.pkl'
    act.save(os.path.join(model_dir, model_fname))


def callback(locals, globals):
    global max_mean_reward, last_filename
    if ('done' in locals and locals['done'] == True):
        if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 1
            and locals['mean_100ep_reward'] > max_mean_reward):

            os.makedirs(os.path.join(proj_dir, 'models/deepq/'), exist_ok=True)
            if (last_filename != ''):
                os.remove(last_filename)

            act = deepq.simple.ActWrapper(locals['act'], locals['act_params'])
            filename = os.path.join(proj_dir,
                'models/deepq/' + game + '_%s.pkl' % locals['mean_100ep_reward'])
            act.save(filename)

            print('Save best mean_100ep_reward model to %s' % filename)
            last_filename = filename
            max_mean_reward = locals['mean_100ep_reward']


if __name__ == '__main__':
    main()
