import numpy as np
import gym
from baselines import deepq
from cytomata.wrappers import wrap_atari


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_atari(env)
    act = deepq.load("pong_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(np.array(obs)[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
