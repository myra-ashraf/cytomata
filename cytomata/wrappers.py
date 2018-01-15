from baselines.common import atari_wrappers as wrap


def wrap_atari(env_id, noop_reset=True, max_skip=True, episodic_life=True,
    scale=False, clip_rewards=True, frame_stack=False):
    env = env_id
    if noop_reset:
        env = wrap.NoopResetEnv(env, noop_max=30)
    if max_skip:
        env = wrap.MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = wrap.EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = wrap.FireResetEnv(env)
    env = wrap.WarpFrame(env)
    if scale:
        env = wrap.ScaledFloatFrame(env)
    if clip_rewards:
        env = wrap.ClipRewardEnv(env)
    if frame_stack:
        env = wrap.FrameStack(env, 4)
    return env


def wrap_cytomatrix(env):
    env = wrap.MaxAndSkipEnv(env, skip=4)
    env = wrap.WarpFrame(env)
    env = wrap.FrameStack(env, 4)
    return env
