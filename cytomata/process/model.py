import os
from collections import defaultdict

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
        dydt = (-(y - 0) + K*(u - 0))/tau
        return dydt

    def simulate(self, t, u, y0):
        uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y, uf, y0, self.K, self.tau, self.theta),
            t_span=[t[0], t[-1]], y0=[y0], t_eval=t, method='LSODA')
        return result.y[0]

    def residual(self, params):
        self.K = params['K']
        self.tau = params['tau']
        self.theta = params['theta']
        self.y = self.simulate(self.tp, self.up, self.yp[0])
        return self.y - self.yp

    def progress(self, params, iter, resid):
        sse = np.sum(resid**2)
        if sse < self.sse:
            self.sse = sse
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
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print('SSE: ' + str(sse))
            print(params.valuesdict())

    def fit_model(self, tp, up, yp, K0=None, tau0=None, theta0=None, method='powell', prg_plot=True):
        self.prg_plot = prg_plot
        self.tp = tp
        self.up = up
        self.yp = yp
        if K0 is None:
            K0 = np.max(self.yp)
        if tau0 is None:
            # timepoint where first occurence of 63% of max output
            tau0 = self.tp[np.where(self.yp == self.yp[np.abs(self.yp - np.max(self.yp)*0.632).argmin()])[0]]
        if theta0 is None:
            # time duration where input is nonzero but output is close to zero
            theta0 = len(np.where((self.tp < np.percentile(self.tp, 10)) & (self.up > 0))[0]) * np.mean(np.ediff1d(self.tp))
        params = lm.Parameters()
        params.add('K', value=K0, min=0.0, max=2*np.max(self.yp))
        params.add('tau', value=tau0, min=0.0, max=np.max(self.tp))
        params.add('theta', value=theta0, min=0.0, max=np.max(self.tp))
        self.opt_results = lm.minimize(
            self.residual, params, method=method, iter_cb=self.progress
        )
        self.close()
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
            self.close()
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
        plt.close()


class Regulator(Env):
    """
    An inducible gene expression model based on regulation of a
    transcription activator between inactive and active states.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params={}):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.display = False
        self.y = defaultdict(list)
        self.params = params
        self.sse = np.inf

    def model(self, t, y):
        I, A, P = y
        k_r = self.params['k_r']
        k_i = self.params['k_i']
        k_a = self.params['k_a']
        k_b = self.params['k_b']
        k_d = self.params['k_d']
        a = self.params['a']
        b = self.params['b']
        n = self.params['n']
        K = self.params['K']
        u = self.uf(t)
        dIdt = k_r + k_i*A - (k_a*u + k_b + k_d)*I
        dAdt = (k_a*u + k_b)*I - (k_i + k_d)*A
        dPdt = b + a*(A**n/(K**n + A**n)) - k_d*P
        if np.isnan(dPdt):
            print('K: ' + str(K))
            print('A: ' + str(A))
            print('n: ' + str(n))
        return [dIdt, dAdt, dPdt]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    def residual(self, params):
        self.params = params
        y = self.simulate(self.tp, self.up, [0.0, 0.0, self.yp[0]])
        self.y['I'] = y[0]
        self.y['A'] = y[1]
        self.y['P'] = y[2]
        if np.isnan(self.y['P']).any():
            return 1e6
        else:
            return self.y['P'] - self.yp

    def progress(self, params, iter, resid):
        sse = np.sum(resid**2)
        if sse < self.sse:
            self.sse = sse
        if self.prg_plot:
            plt.clf()
            plt.plot(self.tp, self.yp)
            plt.plot(self.tp, self.y['I'], label='I')
            plt.plot(self.tp, self.y['A'], label='A')
            plt.plot(self.tp, self.y['P'], label='P')
            title = 'SSE: ' + str(np.format_float_scientific(sse, precision=4)) + '\n'
            for i, (k, v) in enumerate(params.valuesdict().items()):
                title += ' | '+ k +': ' + str(np.round(v, 3))
                if i > 0 and i % 4 == 0:
                    title += '\n'
            plt.title(title)
            plt.ylabel('Output')
            plt.xlabel('Time')
            plt.legend(loc='best')
            plt.pause(1e-6)
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print('SSE: ' + str(sse))
            print(params.valuesdict())

    def fit_model(self, tp, up, yp, method='leastsq', prg_plot=True):
        self.prg_plot = prg_plot
        self.tp = tp
        self.up = up
        self.yp = yp
        params = lm.Parameters()
        params.add('k_r', value=1.0, min=1e-6, max=50)
        params.add('k_i', value=1.0, min=1e-6, max=50)
        params.add('k_a', value=1.0, min=1e-6, max=50)
        params.add('k_b', value=1.0, min=1e-6, max=50)
        params.add('k_d', value=1.0, min=1e-6, max=50)
        params.add('a', value=1.0, min=1e-6, max=50)
        params.add('b', value=1.0, min=1e-6, max=50)
        params.add('n', value=1.0, min=1, max=10)
        params.add('K', value=1.0, min=1e-6, max=50)
        self.opt_results = lm.minimize(
            self.residual, params, method=method, iter_cb=self.progress, nan_policy='propagate'
        )
        self.close()
        self.params = self.opt_results.params.valuesdict()
        os.system('cls' if os.name == 'nt' else 'clear')
        print(lm.report_fit(self.opt_results))

    def reset(self, y0, dt, n, reward_func):
        # self.n = n
        # self.dt = dt
        # self.t = [0.0]
        # self.u = [0.0]
        # self.y = [y0]
        # self.reward_func = reward_func
        y0 = None
        return y0

    def step(self, action):
        # t0 = self.t[-1]
        # t1 = t0 + self.dt
        # self.t.append(t1)
        # self.u.append(action)
        # uf = interp1d(self.t, self.u)
        # result = solve_ivp(fun=lambda t, y: self.model(t, y, uf, self.y[0], self.K, self.tau, self.theta),
        #                    t_span=[t0, t1], y0=[self.y[-1]], t_eval=[t0, t1], method='LSODA')
        # obs = result.y[0][-1]
        # self.y.append(obs)
        # reward = self.reward_func(self.t, self.u, self.y)
        # done = False
        # if len(self.t) >= self.n:
        #     done = True
        #     self.close()
        # info = None
        # if self.display:
        #     plt.clf()
        #     plt.plot(self.t, self.y)
        #     plt.xlim((0.0, self.dt*self.n))
        #     plt.ylim((0.0, 1.1*self.K))
        #     plt.title('State: '+ str(len(self.t))
        #         + ' | Action: ' + str(action)
        #         + ' | Reward: ' + str(reward)
        #     )
        #     plt.ylabel('Output')
        #     plt.xlabel('Time')
        #     plt.pause(1e-2)
        obs = None
        reward = None
        done = None
        info = None
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
        plt.close()
