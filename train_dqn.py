import os
import shutil
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

import cytomata
from cytomata.wrappers import wrap_cytomatrix, wrap_atari


def main(env_name, tags=None, load_dir=None, fresh_state=False, rnd_seed=23,
    total_steps=1000000, save_freq=10000, gamma=0.99, buffer_size=100000,
    buffer_append_freq=1, epsilon_sched=[(0, 1.0), (400000, 0.6), (800000, 0.01)],
    learning_rate=1e-4, learning_starts=100000, batch_size=32, train_freq=4,
    target_update_freq=1000, hiddens=[512], double_q=True, dueling_q=True,
    prioritized_replay=True, pr_alpha=0.6, pr_beta0=0.4, pr_epsilon=1e-6,
    param_noise=False, layer_norm=False, pn_update_freq=50, pn_reset_freq=10000):
    """Train DQN on a gym environment.

    Args:
        env_name (str): Name of openai gym environment eg. PongNoFrameskip-v4.
        tags (list(strs)): Labels to describe this experiment.
        load_dir (str): Absolute path to save directory eg. PongNoFrameskip-v4_2018-01-01-12-34-56.
        fresh_state (bool): If loading, reset replay buffer and training step.
        rnd_seed (int): Random seed for environment and session.
        total_steps (int): Number of iterations to train.
        save_freq (int): Save model and state every n steps. save_freq=None to skip saving.
        gamma (float): Discount factor.
        buffer_size (int): Size of experience replay buffer.
        buffer_append_freq (int): Append transition every n steps.
        epsilon_sched (list(tuples(int, float))): List of tuples (step, epsilon value at step).
        learning_rate (float): Network optimizer (adam) learning rate.
        learning_starts (int): Start training network after n steps of observation.
        batch_size (int): Size of training minibatch.
        train_freq (int): Train the network every n steps.
        target_update_freq (int): Update target network using weights of estimator network every n steps.
        hiddens (list(ints)): Specify number of nodes for each hidden layer.
        double_q (bool): Use double DQN (https://arxiv.org/abs/1509.06461).
        dueling_q (bool): Use dueling DQN (https://arxiv.org/abs/1511.06581).
        prioritized_replay (bool): Use prioritized replay buffer (https://arxiv.org/abs/1511.05952).
        pr_alpha (float): Alpha parameter for prioritized replay buffer.
        pr_beta0 (float): Initial beta parameter for prioritized replay buffer.
        pr_epsilon (float): Epsilon parameter for prioritized replay buffer.
        param_noise (bool): Use parameter noise for exploration (https://arxiv.org/abs/1706.01905).
        layer_norm (bool): Use layer normalization (needs to be true if using param_noise).
        pn_update_freq (int): Rescale param noise after n steps.
        pn_reset_freq (int): Max number of steps to take per episode before re-perturbing exploration.

    Returns:
        tuple(floats): The highest average 100 episode reward attained and
            the difference between that and the performance of random exploration.

    """
    # Directories and Logging
    save_dir = prep_paths(env_name)
    args = dict(locals())
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    # Create Environment
    env = make_env(env_name)
    if rnd_seed > 0 and rnd_seed is not None:
        set_global_seeds(rnd_seed)
        env.unwrapped.seed(rnd_seed)

    with U.make_session(4) as sess:
        # Create Model
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=hiddens,
            dueling=dueling_q,
            layer_norm=layer_norm
        )

        # Create Training Functions
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: Uint8Input(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            gamma=gamma,
            grad_norm_clipping=10,
            double_q=double_q,
            param_noise=param_noise
        )

        # Epsilon Schedule
        exploration = PiecewiseSchedule(epsilon_sched, outside_value=0.01)

        # Replay Buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, pr_alpha)
            beta_schedule = LinearSchedule(total_steps, initial_p=pr_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)

        # Initialize Variables
        U.initialize()
        update_target()
        epsilon_step = 0
        step_durations = []
        episode_rewards = [0.0]
        saved_reward = None
        baseline_reward = None
        obs = env.reset()
        steps_since_reset = 0
        reset = True
        model_dir = None

        # Load Old Model and State
        if load_dir is not None:
            state = load_model(load_dir)
            if state is not None and not fresh_state:
                step, replay_buffer = state['step'], state['replay_buffer']
                epsilon_step = step

        # Training Loop
        time0 = time.time()
        for step in range(1, total_steps + 1):
            # Determine Epsilon
            # Backtrack epsilon schedule if ave reward hasn't improved for awhile
            bad_progress = np.mean(np.ediff1d(episode_rewards[-100:])) <= 0
            if (bad_progress and exploration.value(epsilon_step) < 0.5):
                epsilon_step = max(0, epsilon_step - total_steps * 0.05)
            kwargs = {}
            if param_noise:
                if pn_reset_freq > 0 and steps_since_reset > pn_reset_freq:
                    # Reset param noise policy
                    reset = True
                update_eps = 0.01  # Ensures that we cannot get stuck completely
                # Compute the threshold such that the KL divergence between
                # perturbed and non-perturbed policy is comparable to
                # eps-greedy exploration with eps = exploration.value(t).
                update_param_noise_threshold = -np.log(
                    (1.0 - exploration.value(epsilon_step)) +
                    (exploration.value(epsilon_step) / float(env.action_space.n))
                )
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = (step % pn_update_freq == 0)
            else:
                update_eps = exploration.value(round(np.random.normal(epsilon_step, 1000)))

            # Get Transition from Env
            reset = False
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            new_obs, rew, done, _ = env.step(action)
            if step > 0 and step % buffer_append_freq == 0:
                replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            episode_rewards[-1] += rew

            if done:
                # Save Model and State
                mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 3)
                improved = saved_reward is None or mean_100ep_reward > saved_reward
                if save_freq is not None and step > 0 and (step % save_freq == 0 or step > total_steps) and improved:
                    saved_reward = mean_100ep_reward
                    if model_dir is not None and os.path.isdir(model_dir):
                        shutil.rmtree(model_dir)
                    model_dir = save_model(save_dir, {'replay_buffer': replay_buffer, 'step': step, 'reward': saved_reward})
                # End-of-Episode Stats
                if step_durations:
                    steps_left = total_steps - step
                    loop_seconds = np.mean(step_durations)
                    logger.record_tabular('loop duration', loop_seconds)
                    logger.record_tabular('ETA (hrs)', steps_left * loop_seconds / 3600)
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("epsilon", update_eps)
                logger.record_tabular("reward (100 ep mean)", mean_100ep_reward)
                logger.dump_tabular()
                # New Episode
                obs = env.reset()
                episode_rewards.append(0.0)
                steps_since_reset = 0
                reset = True

            if step == learning_starts:
                baseline_reward = mean_100ep_reward

            # Train the Network
            if step > learning_starts and step % train_freq == 0:
                # Sample Transitions
                if prioritized_replay:
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes
                    ) = replay_buffer.sample(
                        batch_size, beta=beta_schedule.value(epsilon_step))
                else:
                    (obses_t, actions, rewards, obses_tp1, dones) = replay_buffer.sample(batch_size)
                    weights = np.ones_like(rewards)
                # Minimize error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                # Update priorities in the replay buffer
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + pr_epsilon
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Update Target Network
            if step % target_update_freq == 0:
                update_target()

            # Increment epsilon schedule and param noise trackers
            epsilon_step += 1
            steps_since_reset += 1

            # Track loop execution speed
            time1 = time.time()
            step_durations.append(time1 - time0)
            time0 = time1
            if len(step_durations) > 1000:
                step_durations.pop(0)

    return saved_reward, (saved_reward - baseline_reward)


def prep_paths(env_name):
    timestamp = time.strftime('_%Y-%m-%d-%H-%M-%S')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, 'experiments', 'deepq', env_name + timestamp)
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(log_dir)
    return save_dir


def make_env(env_name):
    env = gym.make(env_name)
    monitored_env = bench.Monitor(env, logger.get_dir())
    env = wrap_cytomatrix(monitored_env)
    return env


def save_model(save_dir, state):
    try:
        model_dir = 'model-{}-{}'.format(state['step'], state['reward'])
        save_state(os.path.join(save_dir, model_dir, 'saved'))
        state_path = os.path.join(save_dir, 'training_state.pkl.zip')
        relatively_safe_pickle_dump(state, state_path, compression=True)
        logger.log('\nSaved model to {}\n'.format(model_dir))
        return os.path.join(save_dir, model_dir)
    except Exception as err:
        logger.log('\nFailed to save model\n')
        logger.log(str(err) + '\n')
        return None


def load_model(save_dir):
    try:
        state_path = os.path.join(os.path.join(save_dir, 'training_state.pkl.zip'))
        if os.path.exists(state_path):
            state = pickle_load(state_path, compression=True)
            model_dir = 'model-{}-{}'.format(state['step'], state['reward'])
            load_state(os.path.join(save_dir, model_dir, 'saved'))
            logger.log('Loaded model {} at {} steps'.format(save_dir, state['step']))
            return state
        else:
            logger.log('\nFailed to load model: path not found\n')
    except Exception as err:
        logger.log('\nFailed to load model\n')
        logger.log('Error: ' + str(err) + '\n')


def choose_param(param_vals):
    return param_vals[np.random.choice(len(param_vals))]


if __name__ == '__main__':
    gammas = [0.9, 0.99, 0.999]
    buffer_sizes = [10000, 100000, 200000]
    buffer_append_freqs = [1, 2, 4, 8]
    epsilon_chkpt1s = [0.9, 0.7, 0.5, 0.3]
    epsilon_diffs = [0.1, 0.2, 0.4, 0.5]
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    learning_starts = [10000, 50000, 100000]
    batch_sizes = [16, 32, 48, 64]
    train_freqs = [1, 2, 4, 8]
    target_update_freqs = [1000, 2000, 4000, 8000]
    hiddens = [[256], [512], [1024], [256, 256], [256, 512], [512, 256], [512, 512]]
    pr_alphas = [0.4, 0.6, 0.8]
    pr_beta0s = [0.2, 0.4, 0.6]
    pr_epsilons = [1e-5, 1e-6, 1e-7]
    param_noises = [True, False]
    layer_norms = [True, False]
    pn_update_freqs = [25, 50, 100]
    pn_reset_freqs = [1000, 10000, 20000]

    epsilon_chkpt1 = choose_param(epsilon_chkpt1s)
    epsilon_diff = choose_param(epsilon_diffs)
    epsilon_chkpt2 = max(0.01, (epsilon_chkpt1 - epsilon_diff))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.join(current_dir, 'experiments', 'deepq', 'Cytomatrix-v0_2018-02-04-10-02-05')

    _, performance = main(
        env_name='Cytomatrix-v0',
        tags=['10x10', '1 randwalk cancer', '30 invis randwalk cytes'],
        load_dir=None,
        fresh_state=True,
        total_steps=3000000,
        save_freq=10000,
        gamma=0.99,
        buffer_size=100000,
        buffer_append_freq=1,
        epsilon_sched=[(0, 1.0), (1000000, 0.1), (2000000, 0.01)],
        learning_rate=1e-4,
        learning_starts=100000,
        batch_size=32,
        train_freq=4,
        target_update_freq=1000,
        hiddens=[512],
        double_q=True,
        dueling_q=True,
        prioritized_replay=True,
        pr_alpha=0.6,
        pr_beta0=0.4,
        pr_epsilon=1e-6,
        param_noise=False,
        layer_norm=False,
        pn_update_freq=50,
        pn_reset_freq=10000
    )
