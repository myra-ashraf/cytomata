import os

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from cytomata.utils.gym import Env, Box, Discrete


class FOPDT(Env):
    """
    First Order Plus Dead Time Approximation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, K=None, tau=None, theta=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.display = False
        self.K = K
        self.tau = tau
        self.theta = theta
        self.sse = np.inf

    def model(self, t, y, uf, y0, K, tau, theta):
        if (t - theta) <= 0:
            u = uf(0.0)
        else:
            u = uf(t - theta)
        dydt = (-(y - y0) + K*(u - uf(0.0)))/tau
        return dydt

    def simulate(self, t, u, y0):
        uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y, uf, y0, self.K, self.tau, self.theta),
            t_span=[t[0], t[-1]], y0=[y0], t_eval=t, method='LSODA'
        )
        return result.y[0]

    def residual(self, params):
        self.K = params['K']
        self.tau = params['tau']
        self.theta = params['theta']
        self.y = self.simulate(self.tp, self.up, self.yp[0])
        return self.y - self.yp

    def progress(self, params, iter, resid):
        sse = np.sum(resid**2)
        if sse < self.sse and not self.prg_plot:
            self.sse = sse
            os.system('cls' if os.name == 'nt' else 'clear')
            print('SSE: ' + str(sse))
            print(params.valuesdict())
        if self.prg_plot:
            plt.clf()
            plt.plot(self.tp, self.yp)
            plt.plot(self.tp, self.y)
            plt.title('SSE: '+ str(np.round(sse, 3))
                + ' | K: ' + str(np.round(params['K'].value, 3))
                + ' | tau: ' + str(np.round(params['tau'].value, 3))
                + ' | theta: ' + str(np.round(params['theta'].value, 3))
            )
            plt.ylabel('Output')
            plt.xlabel('Time')
            plt.pause(1e-6)

    def fit_model(self, tp, up, yp, K0=None, tau0=None, theta0=None, method='powell', prg_plot=True):
        self.prg_plot = prg_plot
        self.tp = tp
        self.up = up
        self.yp = yp
        params = lm.Parameters()
        if K0 is None:
            K0 = np.max(self.yp)
        if tau0 is None:
            # timepoint where first occurence of 63% of max output
            tau0 = self.tp[np.where(self.yp == self.yp[np.abs(self.yp - np.max(self.yp)*0.632).argmin()])[0]]
        if theta0 is None:
            # time duration where input is nonzero but output is close to zero
            theta0 = len(np.where((self.tp < np.percentile(self.tp, 10)) & (self.up > 0))[0]) * np.mean(np.ediff1d(self.tp))
        params.add('K', value=K0, min=0.0, max=2*np.max(self.yp))
        params.add('tau', value=tau0, min=0.0, max=np.max(self.tp))
        params.add('theta', value=theta0, min=0.0, max=np.max(self.tp))
        self.opt_results = lm.minimize(
            self.residual, params, method=method, iter_cb=self.progress
        )
        self.K = self.opt_results.params['K'].value
        self.tau = self.opt_results.params['tau'].value
        self.theta = self.opt_results.params['theta'].value
        os.system('cls' if os.name == 'nt' else 'clear')
        print(lm.report_fit(self.opt_results))

    def reset(self, y0, dt, n, reward_func):
        self.n = n
        self.dt = dt
        self.t = [0.0]
        self.u = [0.0]
        self.y = [y0]
        self.reward_func = reward_func
        return y0

    def step(self, action):
        t0 = self.t[-1]
        t1 = t0 + self.dt
        self.t.append(t1)
        self.u.append(action)
        uf = interp1d(self.t, self.u)
        result = solve_ivp(fun=lambda t, y: self.model(t, y, uf, self.y[0], self.K, self.tau, self.theta),
                           t_span=[t0, t1], y0=[self.y[-1]], t_eval=[t0, t1], method='LSODA')
        obs = result.y[0][-1]
        self.y.append(obs)
        reward = self.reward_func(self.t, self.u, self.y)
        done = False
        if len(self.t) >= self.n:
            done = True
        info = None
        if self.display:
            plt.clf()
            plt.plot(self.t, self.y)
            plt.xlim((0.0, self.dt*self.n))
            plt.ylim((0.0, 1.1*self.K))
            plt.title('State: '+ str(len(self.t))
                + ' | Action: ' + str(action)
                + ' | Reward: ' + str(reward)
            )
            plt.ylabel('Output')
            plt.xlabel('Time')
            plt.pause(1e-2)

        return obs, reward, done, info

    def seed(self, seed):
        self.rng = rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
