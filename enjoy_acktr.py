""" Use a pre-trained acktr model to play Breakout.
    To train: python3 ./run_atari.py
        You'll need to add "logger.configure(<some dir>)" to run_atari.py so it will save checkpoint files
    Then run this script with a checkpoint file as the argument
    A running average of the past 100 rewards will be printed
"""
import os
from optparse import OptionParser
from collections import deque
import time

import gym
import cloudpickle
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.acktr_disc import Model
from baselines.acktr.policies import CnnPolicy
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def displayFrames(state):
    frame1 = np.squeeze(state[:,:,:,0])
    frame2 = np.squeeze(state[:,:,:,1])
    frame3 = np.squeeze(state[:,:,:,2])
    frame4 = np.squeeze(state[:,:,:,3])


    plt.figure(1,figsize=(16, 18), dpi=80)
    ax1=plt.subplot(411)
    plt.imshow(frame1,cmap='gray')
    ax1=plt.subplot(412)
    plt.imshow(frame2,cmap='gray')
    ax1=plt.subplot(413)
    plt.imshow(frame3,cmap='gray')
    ax1=plt.subplot(414)
    plt.imshow(frame4,cmap='gray')
    plt.show()

def testOneCheckpoint(env, fname, options, verbose=False):
    tf.reset_default_graph()

    total_timesteps=int(40e6)
    nprocs = 2
    nenvs = 1
    nstack = 4
    nsteps = 1
    nenvs = 3 # This has to match what was used to train the model. I don't see how to get it out of the model/checkpoint files

    ob_space = env.observation_space
    ac_space = env.action_space
    nh, nw, nc = ob_space.shape
    batch_ob_shape = (nenvs*nsteps, nh, nw, nc*nstack)

    policy=CnnPolicy

    mf_name = os.path.join( os.path.dirname(fname), 'make_model.pkl' )
    with open( mf_name, 'rb') as fh:
        make_model = cloudpickle.load(fh)
    model = make_model()

    #model = Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nstack=nstack)
    model.load(fname)
    act = model.step_model
    if not act:
        print( "Unable to load model" )
        return

    episode = 1
    reward_100 = deque(maxlen=100)

    while options.max_episodes == 0 or episode <= options.max_episodes:
        print( "Starting episode {}".format(episode) )
        game_reward = np.zeros([nenvs])
        for point in range(5):
            print( "    Point {}".format(point+1) )
            state = np.zeros(batch_ob_shape, dtype=np.uint8)
            states = model.initial_state

            obs, dones = env.reset(), np.array([False])
            episode_reward = np.zeros([nenvs])
            while not dones[0]:
                if options.render:
                    env.render()
                state = update_obs(state,obs)

                actions, values, states = act.step(state, states, dones)
                obs, rew, dones, _ = env.step(actions)
                for n, done in enumerate(dones):
                    if done:
                        obs[n] = obs[n]*0

                episode_reward += rew
            if verbose:
                print( "   Point Reward: {0}".format( episode_reward ) )
            game_reward += episode_reward
        reward_100.append(game_reward[0])
        episode += 1
        if verbose:
            print( "Game Reward/Avg100: {0}  {1:.3f}".format( game_reward, np.mean(reward_100) ) )

    rmin = np.min(reward_100)
    rmax = np.max(reward_100)
    rmean = np.mean(reward_100)
    rstd = np.std(reward_100)
    if verbose:
        print( "Minimum: {0:.3f}".format( rmin ) )
        print( "Maximum: {0:.3f}".format( rmax ) )
        print( "Avg/std: {0:.3f}/{1:.3f}".format( rmean, rstd  ) )
    return rmin, rmax, rmean, rstd

def update_obs(state, obs):
    obs = np.reshape( obs, state.shape[0:3] )
    state = np.roll(state, shift=-1, axis=3)
    state[:, :, :, -1] = obs
    return state

def getOptions():
    usage = "Usage: python3 enjoy_breakout.py [options] <checkpoint>"
    parser = OptionParser( usage=usage )
    parser.add_option("-r","--render", action="store_true", default=False, help="Render gym environment. Will greatly reduce speed.");
    parser.add_option("-m","--max_episodes", default="0", type="int", help="Maximum number of episodes to play.");
    parser.add_option('--seed', help='RNG seed', type=int, default=0)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        print( usage )
        exit()

    return (options, args)

def main():
    options, args = getOptions()

    #env = gym.make("BreakoutNoFrameskip-v4")
    #env = wrap_deepmind(env)

    seed = options.seed
    num_cpu = 3
    def make_env(rank):
        def _thunk():
            env = gym.make("BreakoutNoFrameskip-v4")
            env.seed(seed + rank)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    tf.reset_default_graph()
    set_global_seeds(0)

    if not os.path.isdir(args[0]):
        rmin, rmax, rmean, rstd = testOneCheckpoint( env, args[0], options, verbose=True )
    else:
        out_fname = os.path.join(args[0],"checkpoint_tests.txt")
        with open(out_fname, 'w') as outf:
            for fname in os.listdir(args[0]):
                if fname.startswith("checkpoint") and not fname.endswith(".txt"):
                    rmin, rmax, rmean, rstd = testOneCheckpoint( env, os.path.join(args[0], fname), options, verbose=False )
                    msg = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format( fname, options.max_episodes, rmin, rmax, rmean, rstd )
                    print( msg )
                    outf.write(msg+"\n")

    print( 'Closing environments' )
    env.close()

if __name__ == '__main__':
    main() 
