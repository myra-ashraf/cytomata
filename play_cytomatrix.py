from cytomata.envs import Cytomatrix

if __name__ == '__main__':
    env = Cytomatrix()
    while True:
        env._reset()
        while not env.terminal:
            env._step(None)
