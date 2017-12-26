import gym
import matplotlib.pyplot as plt
import cytomata
from cytomata.wrappers import wrap_cytomatrix
plt.gray()

if __name__ == '__main__':
    env = gym.make("Cytomatrix-v0")
    env = wrap_cytomatrix(env)
    obs = env.reset()
    plt.figure(1)
    plt.imshow(obs)
    for i in range(16):
        obs, _, _, _ = env.step(1)
    plt.figure(2)
    plt.imshow(obs)
    plt.show()
