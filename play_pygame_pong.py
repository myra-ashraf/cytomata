from cytomata.envs import PygamePong


if __name__ == '__main__':
    env = PygamePong()
    while True:
        env.reset()
        while not env.terminal:
            obs, rew, done, _ = env.step(0)
