from cytomata.envs import Cytomatrix

if __name__ == '__main__':
    env = Cytomatrix()
    while True:
        env.reset()
        while not env.terminal:
            env.step(None)
