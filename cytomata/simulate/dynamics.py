import numpy as np
from gym_env import Env, Box, Discrete


class FOPDT(Env):
    """
    """
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,))
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def reset(self, t, y0, u0, k, dt):
        self.i = 0
        self.t = t
        self.dt = dt
        self.yi = y0
        self.ui = u0
        self.uv = []
        obs = self.simulate(k)
        return obs

    def simulate(self, k):
        K = k['K']
        tau = k['tau']
        theta = k['theta']
        t_eval = np.arange(self.t[i], self.t[i+1] + self.dt, self.dt)
        result = solve_ivp(fun=lambda t, y: self.model(t, y, K, tau, theta),
                           t_span=[self.t[i], self.t[i+1]], y0=[self.yi], t_eval=t_eval
        )
        self.i += 1
        return result.y[0]

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
        if self.uv and (t - self.uv[0][0] > theta):
            u = self.uv.pop(0)[1]
            self.ui = u
        else:
            u = self.ui
        dydt = (-(y - yp[0]) + K*(u - up[0]))/tau
        return dydt

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
