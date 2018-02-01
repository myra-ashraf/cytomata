import os
import numpy as np

import gym
from gym.monitoring import VideoRecorder
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
    while True:
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)
        if done:
            obs = env.reset()
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'Cytomatrix-v0_2018-01-31-08-32-44/model-1800000'
    model_dir = os.path.join(current_dir, 'experiments', 'deepq', model_name)
    env = 'Cytomatrix-v0'
    stochastic = True
    video = True
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
        play(env, act, stochastic, video)
