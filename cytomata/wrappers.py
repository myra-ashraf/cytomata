from baselines.common import atari_wrappers as wrap


def wrap_cytomatrix(env):
    # env = wrap.NoopResetEnv(env, noop_max=30)
    env = wrap.MaxAndSkipEnv(env, skip=8)
    env = wrap.WarpFrame(env)
    env = wrap.ScaledFloatFrame(env)
    env = wrap.ClipRewardEnv(env)
    env = wrap.FrameStack(env, 4)
    return env
