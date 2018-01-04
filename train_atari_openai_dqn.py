import os
import time

import gym
from baselines import deepq, bench, logger
from baselines.common import set_global_seeds

from cytomata.wrappers import wrap_atari


def main():
    game = 'PongNoFrameskip-v4'
    now = time.strftime('_%Y-%m-%d-%H-%M-%S')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(proj_dir, 'models/deepq/')
    log_dir = os.path.join(proj_dir, 'models/deepq/logs/', game + now)
    os.makedirs(log_dir, exist_ok=True)

    logger.configure(log_dir)
    # set_global_seeds(0)

    env = gym.make(game)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
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
    )

    model_fname = game + '_' + now + '.pkl'
    act.save(os.path.join(model_dir, model_fname))
    env.close()


if __name__ == '__main__':
    main()
