import gym
import numpy as np
import matplotlib.pyplot as plt
import cytomata
from cytomata.wrappers import wrap_atari

if __name__ == '__main__':
    env = gym.make("PygamePong-v0")
    env = wrap_atari(env)
    obs1 = env.reset()
    plt.figure(1)
    plt.imshow(obs1)
    for i in range(4):
        obs2, _, _, _ = env.step(1)
    plt.figure(2)
    plt.imshow(obs2)
    for i in range(2):
        obs3, _, _, _ = env.step(2)
    plt.figure(3)
    plt.imshow(obs3)
    plt.show()
