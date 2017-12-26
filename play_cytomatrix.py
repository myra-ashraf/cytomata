from cytomata.envs import CytomatrixEnv

if __name__ == '__main__':
    env = CytomatrixEnv()
    while True:
        env.reset()
        while not env.terminal:
            env.step(None)
