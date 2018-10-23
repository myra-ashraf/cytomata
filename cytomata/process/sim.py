import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from cytomata.utils.gym import Env, Box, Discrete


class FOPDT(Env):
    """
    First Order Plus Dead Time Approximation.
    """
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def reset(self, y0, K, tau, theta, max_iter):
        self.i = 0
        self.max_iter = max_iter
        self.K = K
        self.tau = tau
        self.theta = theta
        self.dt = theta/10.0
        self.t = [0.0]
        self.u = [0.0]
        self.y = [y0]
        return y0

    def step(self, action):
        reward = 0
        done = False
        t1 = self.t[-1]
        t2 = t1 + self.dt
        self.t.append(t2)
        self.u.append(action)
        self.uf = interp1d(self.t, self.u)
        result = solve_ivp(fun=lambda t, y: self.model(t, y, self.K, self.tau, self.theta),
                           t_span=[t1, t2], y0=[self.y[-1]], t_eval=[t1, t2], method='LSODA')
        obs = result.y[0][-1]
        self.y.append(obs)
        if np.diff(result.y[0])[0] < 0.02 and np.diff(result.y[0])[0] > 0.01:
            reward = 1
        if self.i >= self.max_iter:
            done = True
        info = None
        self.i += 1
        return obs, reward, done, info

    def model(self, t, y, K, tau, theta):
        """
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
            u = self.uf(0.0)
        else:
            u = self.uf(t - theta)
        dydt = (-(y - self.y[0]) + K*(u - self.u[0]))/tau
        return dydt

    def seed(self, seed):
        self.rng = rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='rgb_array'):
        img = self.get_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            return
