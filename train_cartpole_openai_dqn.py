import os
import time

import gym

from baselines import deepq, bench, logger
from baselines.common import set_global_seeds


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    game = 'CartPole-v0'
    now = time.strftime('_%Y-%m-%d-%H-%M-%S')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(proj_dir, 'models/deepq/')
    log_dir = os.path.join(proj_dir, 'models/deepq/logs/', game + now)
    os.makedirs(log_dir, exist_ok=True)

    logger.configure(log_dir)
    # set_global_seeds(0)

    env = gym.make(game)
    env = bench.Monitor(env, logger.get_dir())

    model = deepq.models.mlp([64])

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )

    model_fname = game + '_' + now + '.pkl'
    act.save(os.path.join(model_dir, model_fname))


if __name__ == '__main__':
    main()
