import gym
import numpy as np
import matplotlib.pyplot as plt
import cytomata
from cytomata.wrappers import wrap_cytomatrix

if __name__ == '__main__':
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_cytomatrix(env)
    obs1 = env.reset()
    plt.figure(1)
    plt.imshow(obs1)
    for i in range(12):
        obs2, _, _, _ = env.step(3)
    plt.figure(2)
    plt.imshow(obs2)
    plt.show()
