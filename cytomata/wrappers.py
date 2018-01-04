from baselines.common import atari_wrappers as wrap


def wrap_atari(env_id):
    env = wrap.NoopResetEnv(env_id, noop_max=30)
    env = wrap.MaxAndSkipEnv(env, skip=4)
    env = wrap.EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = wrap.FireResetEnv(env)
    env = wrap.WarpFrame(env)
    env = wrap.ScaledFloatFrame(env)
    env = wrap.ClipRewardEnv(env)
    env = wrap.FrameStack(env, 4)
    return env


def wrap_cytomatrix(env):
    # env = wrap.NoopResetEnv(env, noop_max=10)
    env = wrap.MaxAndSkipEnv(env, skip=4)
    # env = wrap.EpisodicLifeEnv(env)
    # env = wrap.FireResetEnv(env)
    env = wrap.WarpFrame(env)
    # env = wrap.ScaledFloatFrame(env)
    # env = wrap.ClipRewardEnv(env)
    env = wrap.FrameStack(env, 4)
    return env
