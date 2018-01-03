import gym
import numpy as np
import matplotlib.pyplot as plt
import cytomata
from cytomata import envs
from cytomata.wrappers import wrap_cytomatrix

if __name__ == '__main__':
    env = gym.make("Cytomatrix-v0")
    env = wrap_cytomatrix(env)
    obs1 = env.reset()
    plt.figure(1)
    plt.imshow(obs1)
    for i in range(1):
        obs2, _, _, _ = env.step(1)
    for i in range(1):
        obs3, _, _, _ = env.step(2)
    print(np.array_equal(np.array(obs2), np.array(obs1)))
    plt.figure(2)
    plt.imshow(obs2)
    plt.figure(3)
    plt.imshow(obs3)
    plt.show()
