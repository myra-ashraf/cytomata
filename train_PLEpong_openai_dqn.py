import os
import time

import gym
import gym_ple
from baselines import deepq, bench, logger
# from baselines.common import set_global_seeds

from cytomata.wrappers import wrap_atari


game = 'PygamePong-v0'
extra_label = ''
target_score = 19


def callback(lcl, glb):
    # stop training if ave reward exceeds target_score
    is_solved = (lcl['t'] > 100
        and sum(lcl['episode_rewards'][-100:]) / 100 >= target_score)
    return is_solved


def main():
    now = time.strftime('_%Y-%m-%d-%H-%M-%S')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(proj_dir, 'models/deepq/')
    log_dir = os.path.join(proj_dir, 'models/deepq/logs/', game + now + extra_label)
    os.makedirs(log_dir, exist_ok=True)

    logger.configure(log_dir)
    # set_global_seeds(0)

    env = gym.make(game)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari(env)

    # model = deepq.models.cnn_to_mlp(
    #     convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    #     hiddens=[256],
    #     dueling=True
    # )
    #
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     lr=1e-4,
    #     max_timesteps=2000000,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     train_freq=4,
    #     learning_starts=10000,
    #     target_network_update_freq=1000,
    #     gamma=0.99,
    #     prioritized_replay=True,
    #     print_freq=1,
    #     callback=callback
    # )
    #
    # model_fname = game + '_' + now + extra_label + '.pkl'
    # act.save(os.path.join(model_dir, model_fname))
    # env.close()


if __name__ == '__main__':
    main()
