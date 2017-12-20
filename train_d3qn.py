import os
import tensorflow as tf
from cytomata.envs import CytomatrixEnv
from cytomata.agents import StateProcessor, Estimator, D3QNAgent


if __name__ == '__main__':
    tf.reset_default_graph()
    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format('cytomata'))
    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Create estimators
    q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")
    # State processor
    state_processor = StateProcessor()
    # Environment
    env = CytomatrixEnv()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in D3QNAgent(sess, env, q_estimator=q_estimator, target_estimator=target_estimator,
                            state_processor=state_processor, experiment_dir=experiment_dir,
                            num_episodes=1000, replay_memory_size=500000, replay_memory_init_size=10000,
                            update_target_estimator_every=10000, epsilon_start=1.0, epsilon_end=0.1,
                            epsilon_decay_steps=500000, discount_factor=0.99, batch_size=32):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
