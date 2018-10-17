import numpy as np
from gym_env import Env, Box, Discrete


class FOPDT(Env):
    """
    """
    def __init__(self, tp, y0):
        self.tp = tp
        self.i = 0
        self.t1 = self.tp[self.i]
        self.t2 = self.tp[self.i + i]
        self.yi = y0
        self.queue = []
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,))
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def model(self, t, y, K, tau, theta):
        """
        First Order Plus Dead Time Approximation.

        Args:
            t (float): Time
            y (float): Output
            K (float): Gain constant
            tau (float): Time constant
            theta (float): Delay constant

        Returns:
            dydt (float): Change in output
        """
        if (t - theta) <= 0:
            u = self.action_queue[0]
        else:
            u = self.action_queue[t - theta]
        dydt = (-(y - yp[0]) + K*(u - up[0]))/tau
        return dydt

    def simulate(self, k):
        K = k['K']
        tau = k['tau']
        theta = k['theta']
        result = solve_ivp(fun=lambda t, y: fopdt(t, y, K, tau, theta),
                           t_span=[self.t1, self.t2], y0=[self.yi], t_eval=[self.t2]
        )
        return result.y[0]

    def reset(self):
        return obs

    def step(self, action):
        return obs, reward, done, info

    def seed(self, seed):
        self.rng = rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='rgb_array'):
        img = self.get_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            return
