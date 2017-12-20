from envs.cytomatrix import CytomatrixEnv

if __name__ == '__main__':
    env = CytomatrixEnv()
    while True:
        env.new()
        while not env.terminal:
            env.step()
