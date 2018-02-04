import os
import numpy as np

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import baselines.common.tf_util as U
from baselines import bench, deepq
from baselines.deepq.utils import Uint8Input, load_state

import cytomata
from cytomata.wrappers import wrap_cytomatrix


def make_env(game_name):
    env = gym.make(game_name)
    env = bench.Monitor(env, None)
    env = wrap_cytomatrix(env)
    return env


def play(env, act, stochastic, video_path):
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    ep_rewards = [0.0]
    while True:
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]
        obs, rew, done, _ = env.step(action)
        ep_rewards[-1] += rew
        if len(ep_rewards) > num_episodes:
            if len(ep_rewards) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(ep_rewards[-1])
            num_episodes = len(ep_rewards)
        if done:
            obs = env.reset()
            ep_rewards.append(0.0)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'Cytomatrix-v0_2018-01-29-10-09-34/model-1600000'
    model_dir = os.path.join(current_dir, 'experiments', 'deepq', model_name)
    env = 'Cytomatrix-v0'
    stochastic = True
    video_path = os.path.join(model_dir, 'enjoy.mp4')
    with U.make_session(4) as sess:
        env = make_env(env)
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[512],
            dueling=True
        )
        act = deepq.build_act(
            make_obs_ph=lambda name: Uint8Input(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n)
        load_state(os.path.join(model_dir, "saved"))
        play(env, act, stochastic, video_path)
