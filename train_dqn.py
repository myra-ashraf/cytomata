import re
import os
import time
import json

import gym
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import bench, deepq, logger
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.misc_util import (
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
)

import cytomata
from cytomata.wrappers import wrap_cytomatrix


args = {
    'env': 'PongNoFrameskip-v0',  # name of gym environment
    'save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'deepq'),
    'save_freq': 10000,  # how often (steps) to save
    'load_model': True,  # load existing model and continue training
    'seed': 23,  # random seed for increased reproducibility
    'monitor': False,  # Record video and stats of training; slightly slows down training
    'total_steps': 2000000,  # At least 1M steps for decent results
    'buffer_size': 100000,  # replay buffer size; not too high or run out of RAM
    'learning_rate': 1e-4,  # lower = more stable training but slower
    'batch_size': 32,  # num of transitions to optimize at same time
    'train_freq': 4,  # number of steps between optimizations; stability
    'target_update_freq': 1000,  # number of steps between updating target network; stability
    'double_q': True,  # double Q learning (https://arxiv.org/abs/1509.06461)
    'dueling': True,  # dueling model (https://arxiv.org/abs/1511.06581)
    'prioritized_replay': True,  # prioritized replay buffer (https://arxiv.org/abs/1511.05952)
    'prioritized_alpha': 0.6,  # prioritized replay alpha parameter
    'prioritized_beta0': 0.4,  # prioritized replay initial beta parameter
    'prioritized_epsilon': 1e-6,  # prioritized replay epsilon parameter
    'param_noise': True,  # parameter space noise for exploration (https://arxiv.org/abs/1706.01905)
    'layer_norm': True,  # layer norm - needs to be True if using param noise
    'pn_update_freq': 50,  # number of steps between rescaling of param noise
    'pn_reset_freq': 10000,  # maximum number of steps to take per episode before re-perturbing the exploration policy
}


def maybe_save_model(save_dir, state):
    """Save model if path is specified"""
    if save_dir is None:
        return
    model_dir = 'model-{}'.format(state['num_iters'])
    U.save_state(os.path.join(save_dir, model_dir, 'saved'))
    relatively_safe_pickle_dump(state, os.path.join(save_dir, 'training_state.pkl.zip'), compression=True)
    relatively_safe_pickle_dump(state['monitor_state'], os.path.join(save_dir, 'monitor_state.pkl'))
    logger.log('Saved model\n')

def maybe_load_model(save_dir):
    """Load model if present at the specified path."""
    if save_dir is None:
        return
    state_path = os.path.join(os.path.join(save_dir, 'training_state.pkl.zip'))
    if os.path.exists(state_path):
        state = pickle_load(state_path, compression=True)
        model_dir = 'model-{}'.format(state['steps'])
        U.load_state(os.path.join(save_dir, model_dir, 'saved'))
        logger.log('Loaded models checkpoint at {} steps'.format(state['steps']))
        return state


if __name__ == '__main__':
    timestamp = time.strftime('_%Y-%m-%d-%H-%M-%S_')
    log_dir = os.path.join(args['save_dir'], 'logs', args['env'] + timestamp)
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(log_dir)

    env = gym.make(args['env'])
    monitored_env = bench.Monitor(env, logger.get_dir())
    env = wrap_cytomatrix(monitored_env)

    if args['seed'] > 0 and args['seed'] is not None:
        set_global_seeds(args['seed'])
        env.unwrapped.seed(args['seed'])

    if args['monitor']:
        env = gym.wrappers.Monitor(env, log_dir, force=True)

    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    with U.make_session(4) as sess:
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[512],
            dueling=True
        )

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args['learning_rate'], epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args['double_q'],
            param_noise=args['param_noise']
        )

        exploration = PiecewiseSchedule([
            (0, 1.0),
            (args['total_steps'] / 50, 0.1),
            (args['total_steps'] / 5, 0.01)
        ], outside_value=0.01)

        if args['prioritized_replay']:
            replay_buffer = PrioritizedReplayBuffer(args['buffer_size'], args['prioritized_alpha'])
            beta_schedule = LinearSchedule(args['total_steps'], initial_p=args['prioritized_beta0'], final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args['buffer_size'])

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(args['save_dir'])
        if state is not None:
            steps, replay_buffer = state['steps'], state['replay_buffer'],
            monitored_env.set_state(state['monitor_state'])

        start_time, start_steps = None, None
        episode_rewards = [0.0]
        obs = env.reset()
        steps = 0
        steps_since_reset = 0
        reset = True

        # Main training loop
        while True:
            steps += 1
            steps_since_reset += 1

            # Take action and store transition in the replay buffer.
            kwargs = {}
            if not args['param_noise']:
                update_eps = exploration.value(steps)
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
                update_param_noise_threshold = -np.log(1. - exploration.value(steps) + exploration.value(steps) / float(env.action_space.n))
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

            # if start_time is not None:
            #     steps_per_iter.update(info['steps'] - start_steps)
            #     iteration_time_est.update(time.time() - start_time)
            # start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if steps > 0 and (steps % args['save_freq'] == 0 or steps > args['total_steps']):
                maybe_save_model(savedir, {
                    'replay_buffer': replay_buffer,
                    'steps': steps,
                    'monitor_state': monitored_env.get_state(),
                })

            if steps > args['total_steps']:
                break

            if done:
                steps_left = args['total_steps'] - steps
                completion = np.round(steps / args['total_steps'], 1)

                logger.record_tabular("completion", completion)
                logger.record_tabular("steps", steps)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("reward (100 ep mean)", round(np.mean(episode_rewards[-100:]), 1))
                logger.record_tabular("exploration", exploration.value(steps))
                if args['prioritized_replay']:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                # fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                #                 if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                # logger.log()
                # logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                # logger.log()
