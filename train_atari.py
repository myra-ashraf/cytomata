import os
import gym
import cytomata
import argparse

from baselines import deepq
from baselines.common.atari_wrappers import make_atari
from baselines.common import set_global_seeds
from baselines import bench
from baselines import logger
from cytomata.wrappers import wrap_cytomatrix


PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
max_mean_reward = -100
last_filename = ''


def callback(locals, globals):
    global max_mean_reward, last_filename
    if ('done' in locals and locals['done'] == True):
        if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 1
            and locals['mean_100ep_reward'] > max_mean_reward):

            os.makedirs(os.path.join(PROJ_DIR, 'models/deepq/'), exist_ok=True)
            if (last_filename != ''):
                os.remove(last_filename)

            act = deepq.simple.ActWrapper(locals['act'], locals['act_params'])
            filename = os.path.join(PROJ_DIR,
                'models/deepq/cytomatrix_%s.pkl' % locals['mean_100ep_reward'])
            act.save(filename)

            print('Save best mean_100ep_reward model to %s' % filename)
            last_filename = filename
            max_mean_reward = locals['mean_100ep_reward']


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(2e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)

    # env = gym.make('PongNoFrameskip-v4')
    # env = bench.Monitor(env, logger.get_dir())
    # env = wrap_cytomatrix(env)

    env = make_atari("PongNoFrameskip-v4")
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling))

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        print_freq=1,
        callback=callback)
    act.save('pong_model.pkl')


if __name__ == '__main__':
    main()
