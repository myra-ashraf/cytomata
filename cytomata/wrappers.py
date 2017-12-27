from baselines.common import atari_wrappers as wrap
import matplotlib.pyplot as plt
plt.jet()

def wrap_cytomatrix(env):
    env = wrap.NoopResetEnv(env, noop_max=30)
    env = wrap.MaxAndSkipEnv(env, skip=16)
    # env = wrap.EpisodicLifeEnv(env)
    # env = wrap.FireResetEnv(env)
    env = wrap.WarpFrame(env)
    # env = wrap.ScaledFloatFrame(env)
    env = wrap.ClipRewardEnv(env)
    env = wrap.FrameStack(env, 4)
    return env
