import gym
import numpy as np
import matplotlib.pyplot as plt
import cytomata
from cytomata.wrappers import wrap_atari, wrap_cytomatrix
from cytomata.wrappers import ScaledFloatFrame
from baselines.common.atari_wrappers_deprecated import wrap_dqn
# plt.gray()

if __name__ == '__main__':
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    obs = env.reset()
    print(np.array(obs).shape)
    plt.figure(1)
    plt.imshow(obs)
    for i in range(10):
        obs, _, _, _ = env.step(0)
    plt.figure(2)
    plt.imshow(obs)
    plt.show()
