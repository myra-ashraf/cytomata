import os
import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.acktr.acktr_disc import learn
from baselines.ppo2.policies import CnnPolicy

import cytomata
from cytomata.wrappers import wrap_cytomatrix


def make_cyto_env(env_id, num_env, seed, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_cytomatrix(env)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def train(env_id, num_timesteps, seed, num_cpu):
    env = VecFrameStack(make_cyto_env(env_id, num_cpu, seed), 4)
    learn(CnnPolicy, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)


if __name__ == '__main__':
    args = {
        'env': 'Cytomatrix-v0',
        'num_timesteps': 2000000,
        'seed': 23,
    }
    logger.configure()
    train(args['env'], num_timesteps=args['num_timesteps'], seed=args['seed'], num_cpu=4)
