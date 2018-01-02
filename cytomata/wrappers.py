from baselines.common import atari_wrappers as wrap


def wrap_cytomatrix(env):
    env = wrap.NoopResetEnv(env, noop_max=10)
    env = wrap.MaxAndSkipEnv(env, skip=4)
    # env = wrap.EpisodicLifeEnv(env)
    # env = wrap.FireResetEnv(env)
    env = wrap.WarpFrame(env)
    # env = wrap.ScaledFloatFrame(env)
    # env = wrap.ClipRewardEnv(env)
    env = wrap.FrameStack(env, 4)
    return env
