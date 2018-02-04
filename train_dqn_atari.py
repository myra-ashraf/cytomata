import re
import os
import time
import json

import gym
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import bench, deepq, logger
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import Uint8Input, load_state, save_state
from baselines.common.misc_util import pickle_load, relatively_safe_pickle_dump, set_global_seeds
from baselines.common import atari_wrappers as wrap


def wrap_atari(env, noop_reset=True, max_skip=True, episodic_life=True,
    scale=False, clip_rewards=True, frame_stack=True):
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


def save_model(save_dir, state):
    try:
        model_dir = 'model-{}'.format(state['steps'])
        state_path = os.path.join(save_dir, 'training_state.pkl.zip')
        save_state(os.path.join(save_dir, model_dir, 'saved'))
        relatively_safe_pickle_dump(state, state_path, compression=True)
        logger.log('\nSaved model to {}\n'.format(model_dir))
    except Exception as err:
        logger.log('\nFailed to save model\n')
        logger.log(str(err) + '\n')

def load_model(save_dir):
    try:
        state_path = os.path.join(os.path.join(save_dir, 'training_state.pkl.zip'))
        if os.path.exists(state_path):
            state = pickle_load(state_path, compression=True)
            model_dir = 'model-{}'.format(state['steps'])
            load_state(os.path.join(save_dir, model_dir, 'saved'))
            logger.log('Loaded model {} at {} steps'.format(save_dir, state['steps']))
            return state
        else:
            logger.log('\nFailed to load model: path not found\n')
    except Exception as err:
        logger.log('\nFailed to load model\n')
        logger.log('Error: ' + str(err) + '\n')


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    timestamp = time.strftime('_%Y-%m-%d-%H-%M-%S')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, 'experiments', 'deepq', env_name + timestamp)
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(log_dir)

    args = {
        'env': env_name,
        'save_dir': save_dir,
        'timestamp': timestamp,
        'save_freq': 200000,  # how often (steps) to save
        'load_model': False,  # load existing model and continue training
        'load_dir': os.path.join(current_dir, 'experiments', 'deepq', 'PongNoFrameskip-v4_2018-01-31-08-32-44'),
        'reset_state': True,  # throw away the saved replay buffer and current step from loaded state
        'seed': 23,  # random seed for reproducibility
        'total_steps': 2000000,  # training iterations
        'pw_eps': False,  # Use a piecewise epsilon schedule otherwise use linear
        'pw_eps_sched': [(0, 1.0), (800000, 0.4), (1000000, 0.1), (1200000, 0.01)],  # (step, epsilon value at that step)
        'linear_eps_steps': 100000,  # number of steps to decrease epsilon from initial to final
        'linear_eps_init': 1.0,  # initial epsilon
        'linear_eps_final': 0.01,  # final epsilon
        'buffer_size': 100000,  # replay buffer size; not too high or run out of RAM
        'learning_rate': 1e-4,  # lower = more stable training but slower
        'batch_size': 32,  # num of transitions to optimize at same time
        'train_freq': 4,  # number of steps between optimizations
        'target_update_freq': 1000,  # number of steps between updating target network
        'hiddens': [512],  # Fully connected layers
        'double_q': True,  # double Q learning (https://arxiv.org/abs/1509.06461)
        'dueling': True,  # dueling model (https://arxiv.org/abs/1511.06581)
        'prioritized_replay': True,  # prioritized replay buffer (https://arxiv.org/abs/1511.05952)
        'prioritized_alpha': 0.6,  # prioritized replay alpha parameter
        'prioritized_beta0': 0.4,  # prioritized replay initial beta parameter
        'prioritized_epsilon': 1e-6,  # prioritized replay epsilon parameter
        'param_noise': False,  # parameter space noise for exploration (https://arxiv.org/abs/1706.01905)
        'layer_norm': False,  # layer norm - needs to be True if using param noise
        'pn_update_freq': 50,  # number of steps between rescaling of param noise
        'pn_reset_freq': 10000,  # maximum number of steps to take per episode before re-perturbing the exploration policy
    }
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    env = gym.make(args['env'])
    monitored_env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari(monitored_env)

    if args['seed'] > 0 and args['seed'] is not None:
        set_global_seeds(args['seed'])
        env.unwrapped.seed(args['seed'])

    # if args.gym_monitor and savedir:
    #     env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    with U.make_session(8) as sess:
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=args['hiddens'],
            dueling=args['dueling']
        )

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: Uint8Input(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args['learning_rate']),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args['double_q'],
            param_noise=args['param_noise']
        )

        if args['pw_eps']:
            exploration = PiecewiseSchedule(args['pw_eps_sched'], outside_value=0.01)
        else:
            exploration = LinearSchedule(
                schedule_timesteps=args['linear_eps_steps'],
                initial_p=args['linear_eps_init'], final_p=args['linear_eps_final']
            )

        if args['prioritized_replay']:
            replay_buffer = PrioritizedReplayBuffer(args['buffer_size'], args['prioritized_alpha'])
            beta_schedule = LinearSchedule(args['total_steps'], initial_p=args['prioritized_beta0'], final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args['buffer_size'])

        U.initialize()
        update_target()
        steps = 0
        epsilon_step = 0

        # Load the model
        if args['load_model']:
            state = load_model(args['load_dir'])
            if state is not None and not args['reset_state']:
                steps, replay_buffer = state['steps'], state['replay_buffer']
                epsilon_step = steps
                # monitored_env.set_state(state['monitor_state'])

        episode_rewards = [0.0]
        obs = env.reset()
        steps_since_reset = 0
        reset = True

        # Main training loop
        time0 = time.time()
        step_durations = []
        while True:
            steps += 1
            epsilon_step += 1
            steps_since_reset += 1

            # Take action and store transition in the replay buffer.
            kwargs = {}
            if not args['param_noise']:
                update_eps = exploration.value(epsilon_step)
                update_param_noise_threshold = 0.
            else:
                if args['pn_reset_freq'] > 0 and steps_since_reset > args['pn_reset_freq']:
                    # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                    reset = True

                update_eps = 0.01  # ensures that we cannot get stuck completely
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(
                    1. - exploration.value(epsilon_step) + exploration.value(epsilon_step) / float(env.action_space.n)
                )
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = (steps % args['pn_update_freq'] == 0)

            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            new_obs, rew, done, _ = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            episode_rewards[-1] += rew

            if done:
                steps_left = args['total_steps'] - steps
                if step_durations:
                    loop_seconds = np.mean(step_durations)
                    logger.record_tabular('step duration', loop_seconds)
                    logger.record_tabular('ETA (hrs)', steps_left * loop_seconds / 3600)
                completion = np.round(steps / args['total_steps'] * 100, 3)
                logger.record_tabular("percent complete", completion)
                logger.record_tabular("steps", steps)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("reward (100 ep mean)", round(np.mean(episode_rewards[-100:]), 3))
                logger.record_tabular("epsilon", exploration.value(epsilon_step))
                logger.dump_tabular()
                steps_since_reset = 0
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if (steps > max(5 * args['batch_size'], args['buffer_size'] // 20) and
                    steps % args['train_freq'] == 0):
                # Sample a bunch of transitions from replay buffer
                if args['prioritized_replay']:
                    experience = replay_buffer.sample(args['batch_size'], beta=beta_schedule.value(steps))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args['batch_size'])
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                # Update the priorities in the replay buffer
                if args['prioritized_replay']:
                    new_priorities = np.abs(td_errors) + args['prioritized_epsilon']
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if steps % args['target_update_freq'] == 0:
                update_target()

            # Save the model and training state.
            if steps > 0 and (steps % args['save_freq'] == 0 or steps > args['total_steps']):
                save_model(args['save_dir'], {
                    'replay_buffer': replay_buffer,
                    'steps': steps,
                    # 'monitor_state': monitored_env.get_state(),
                })

            if steps > args['total_steps']:
                break

            time1 = time.time()
            step_durations.append(time1 - time0)
            time0 = time1
            if len(step_durations) > 100:
                step_durations.pop(0)
