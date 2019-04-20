import os
import time
import json
from collections import defaultdict

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
plt.rcParams['figure.figsize'] = 12, 12
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelpad'] = 0

from cytomata.utils.io import setup_dirs
from cytomata.utils.gym import Env, Box, Discrete


class FRC(Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.params = params

    def model(self, t, y):
        u = self.uf(t)
        [Ni, Na, P1i, P1a, P2i, P2a, M1, M2] = y
        kNa = self.params['kNa']
        kNi = self.params['kNi']
        kP1a = self.params['kP1a']
        kP1i = self.params['kP1i']
        kP2a = self.params['kP2a']
        kP2i = self.params['kP2i']
        kM1f = self.params['kM1f']
        kM1r = self.params['kM1r']
        kM2f = self.params['kM2f']
        kM2r = self.params['kM2r']
        dNi = kNi*Na - kNa*u*Ni
        dNa = kNa*u*Ni - kNi*Na + kM1r*M1 + kM2r*M2 - kM1f*Na*P1a - kM2f*Na*P2a
        dP1i = kP1i*P1a - kP1a*u*P1i
        dP1a = kP1a*u*P1i - kP1i*P1a + kM1r*M1 - kM1f*Na*P1a
        dP2i = kP2i*P2a - kP2a*u*P2i
        dP2a = kP2a*u*P2i - kP2i*P2a + kM2r*M2 - kM2f*Na*P2a
        dM1 = kM1f*Na*P1a - kM1r*M1
        dM2 = kM2f*Na*P2a - kM2r*M2
        return [dNi, dNa, dP1i, dP1a, dP2i, dP2a, dM1, dM2]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    def reset(self, y0, dt, n, reward_func):
        y0 = None
        return y0

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def seed(self, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
        plt.close()


class TwoSystem(Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.params = params

    def model(self, t, y):
        u = self.uf(t)
        [Ni, Na, P1i, P1a, P2i, P2a, T1, R1, T2] = y
        kNa = self.params['kNa']
        kNi = self.params['kNi']
        kP1a = self.params['kP1a']
        kP1i = self.params['kP1i']
        kP2a = self.params['kP2a']
        kP2i = self.params['kP2i']
        kT1f = self.params['kT1f']
        kT1r = self.params['kT1r']
        kR1f = self.params['kR1f']
        kR1r = self.params['kR1r']
        kT1a = self.params['kT1a']
        kT1b = self.params['kT1b']
        n1 = self.params['n1']
        kd1 = self.params['kd1']
        kR1a = self.params['kR1a']
        kR1b = self.params['kR1b']
        n2 = self.params['n2']
        dNi = kNi*Na - kNa*u*Ni
        dNa = kNa*u*Ni - kNi*Na + kT1r*T1 + kR1r*R1 - kT1f*Na*P1a - kR1f*Na*P2a
        dP1i = kP1i*P1a - kP1a*u*P1i
        dP1a = kP1a*u*P1i - kP1i*P1a + kT1r*T1 - kT1f*Na*P1a
        dP2i = kP2i*P2a - kP2a*u*P2i
        dP2a = kP2a*u*P2i - kP2i*P2a + kR1r*R1 - kR1f*Na*P2a
        dT1 = kT1f*Na*P1a - kT1r*T1
        dR1 = kR1f*Na*P2a - kR1r*R1
        dT2 = (kT1a*T1**n1)/(kT1b**n1 + T1**n1) - (kR1a*R1**n2)/(kR1b**n2 + R1**n2)
        return [dNi, dNa, dP1i, dP1a, dP2i, dP2a, dT1, dR1, dT2]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='RK45')
        return result.y

    def reset(self, y0, dt, n, reward_func):
        y0 = None
        return y0

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def seed(self, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
        plt.close()


class ThreeSystem(Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.params = params

    def model(self, t, y):
        u = self.uf(t)
        [Ni, Na, P1i, P1a, P2i, P2a, P3i, P3a, T1, R1, R2, T2] = y
        kNa = self.params['kNa']
        kNi = self.params['kNi']
        kP1a = self.params['kP1a']
        kP1i = self.params['kP1i']
        kP2a = self.params['kP2a']
        kP2i = self.params['kP2i']
        kP3a = self.params['kP3a']
        kP3i = self.params['kP3i']
        kT1f = self.params['kT1f']
        kT1r = self.params['kT1r']
        kR1f = self.params['kR1f']
        kR1r = self.params['kR1r']
        kR2f = self.params['kR2f']
        kR2r = self.params['kR2r']
        kT1a = self.params['kT1a']
        kT1b = self.params['kT1b']
        n1 = self.params['n1']
        kR1a = self.params['kR1a']
        kR1b = self.params['kR1b']
        n2 = self.params['n2']
        kR2a = self.params['kR2a']
        kR2b = self.params['kR2b']
        n3 = self.params['n3']
        dNi = kNi*Na - kNa*u*Ni
        dNa = kNa*u*Ni - kNi*Na + kT1r*T1 + kR1r*R1 + kR2r*R2 - kT1f*Na*P1a - kR1f*Na*P2a - kR2f*Na*P3a
        dP1i = kP1i*P1a - kP1a*u*P1i
        dP1a = kP1a*u*P1i - kP1i*P1a + kT1r*T1 - kT1f*Na*P1a
        dP2i = kP2i*P2a - kP2a*u*P2i
        dP2a = kP2a*u*P2i - kP2i*P2a + kR1r*R1 - kR1f*Na*P2a
        dP3i = kP3i*P3a - kP3a*u*P3i
        dP3a = kP3a*u*P3i - kP3i*P3a + kR2r*R2 - kR2f*Na*P3a
        dT1 = kT1f*Na*P1a - kT1r*T1
        dR1 = kR1f*Na*P2a - kR1r*R1
        dR2 = kR2f*Na*P3a - kR2r*R2
        dT2 = (kT1a*T1**n1)/(kT1b**n1 + T1**n1) - kR1a/(1 + (R1/kR1b)**n2) - kR2a/(1 + (R2/kR2b)**n3)
        return [dNi, dNa, dP1i, dP1a, dP2i, dP2a, dP3i, dP3a, dT1, dR1, dR2, dT2]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    def reset(self, y0, dt, n, reward_func):
        y0 = None
        return y0

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def seed(self, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
        plt.close()


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

    def fit(self, tp, up, yp, K0=None, tau0=None, theta0=None, method='powell', prg_plot=True):
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


class FrequencyResonance(Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.params = params

    def model(self, t, y):
        u = self.uf(t)
        [LCBc, LCBn, NCVc, NCVn, AC, POI] = y
        kf1 = self.params['kf1']
        kr1 = self.params['kr1']
        kf2 = self.params['kf2']
        kr2 = self.params['kr2']
        kf3 = self.params['kf3']
        kr3 = self.params['kr3']
        ka = self.params['ka']
        kb = self.params['kb']
        kc = self.params['kc']
        n = self.params['n']
        kd = self.params['kd']
        dLCBc = kr1*LCBn - kf1*u*LCBc
        dLCBn = kf1*u*LCBc - kr1*LCBn + kr3*AC
        dNCVc = kr2*NCVn - kf2*NCVc
        dNCVn = kf2*NCVc - kr2*NCVn + kr3*AC
        dAC = kf3*u*(LCBn + NCVn) - kr3*AC
        dPOI = ka + kb*(AC**n/(kc**n + AC**n)) - kd*POI
        return [dLCBc, dLCBn, dNCVc, dNCVn, dAC, dPOI]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    def reset(self, y0, dt, n, reward_func):
        y0 = None
        return y0

    def step(self, action):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def seed(self, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
        plt.close()


class Lintad(Env):
    """
    An blue light inducible gene expression model based on nuclear shuttling
    and dimerization of a transcription regulator pair.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.params = params
        # self.y = defaultdict(list)
        # self.display = False
        # self.sse_fit = [1e12]
        # self.time_fit = [time.time()]
        # self.top_params = None
        # self.clock = time.time()
        # self.save_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M%S'))
        # setup_dirs(self.save_dir)

    def model(self, t, y):
        u = self.uf(t)
        [LCBc, LCBn, NCVc, NCVn, AC, POI] = y
        kf1 = self.params['kf1']
        kr1 = self.params['kr1']
        kf2 = self.params['kf2']
        kr2 = self.params['kr2']
        kf3 = self.params['kf3']
        kr3 = self.params['kr3']
        ka = self.params['ka']
        kb = self.params['kb']
        kc = self.params['kc']
        n = self.params['n']
        kd = self.params['kd']
        dLCBc = kr1*LCBn - kf1*u*LCBc
        dLCBn = kf1*u*LCBc - kr1*LCBn + kr3*AC
        dNCVc = kr2*NCVn - kf2*NCVc
        dNCVn = kf2*NCVc - kr2*NCVn + kr3*AC
        dAC = kf3*u*(LCBn + NCVn) - kr3*AC
        dPOI = ka + kb*(AC**n/(kc**n + AC**n)) - kd*POI
        return [dLCBc, dLCBn, dNCVc, dNCVn, dAC, dPOI]

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    # def fit(self, tp, up, yp, y0=None, method='powell', method_kwargs={}):
    #     self.fit_figs_dir = os.path.join(self.save_dir, 'fit_figs')
    #     setup_dirs(self.fit_figs_dir)
    #     self.method = method
    #     self.tp = tp
    #     self.up = up
    #     self.yp = yp
    #     if y0:
    #         self.y0 = y0
    #     elif len(self.yp.shape) == 1:
    #         self.y0 = [0.0, 0.0, self.yp[0]]
    #     else:
    #         self.y0 = self.yp[0]
    #     params = lm.Parameters()
    #     params.add('k_r', value=1, min=0, max=20)
    #     params.add('k_i', value=1, min=0, max=20)
    #     params.add('k_a', value=1, min=0, max=20)
    #     params.add('k_b', value=1, min=0, max=20)
    #     params.add('k_d', value=1, min=0, max=20)
    #     params.add('a', value=1, min=0, max=20)
    #     params.add('b', value=1, min=0, max=20)
    #     params.add('n', value=1, min=0, max=10)
    #     params.add('K', value=1, min=0, max=20)
    #     self.opt_results = lm.minimize(
    #         self.residual, params, method=method, iter_cb=self.progress,
    #         nan_policy='propagate', **method_kwargs
    #     )
    #     self.close()
    #     sse_path = os.path.join(self.save_dir, 'sse.csv')
    #     sse_data = np.column_stack((self.time_fit, self.sse_fit))
    #     np.savetxt(sse_path, sse_data, delimiter=',', header='t,sse')
    #     self.params = self.opt_results.params.valuesdict()
    #     os.system('cls' if os.name == 'nt' else 'clear')
    #     print(lm.report_fit(self.opt_results))
    #
    # def residual(self, params):
    #     self.params = params
    #     y = self.simulate(self.tp, self.up, self.y0)
    #     self.y['I'] = np.nan_to_num(y[0])
    #     self.y['A'] = np.nan_to_num(y[1])
    #     self.y['P'] = np.nan_to_num(y[2])
    #     if len(self.yp.shape) == 1:
    #         return self.y['P'] - self.yp
    #     else:
    #         return y - self.yp
    #
    # def progress(self, params, iter, resid):
    #     sse = np.sum(resid**2)
    #     if sse < self.sse_fit[-1]:
    #         self.sse_fit.append(sse)
    #         self.time_fit.append(time.time())
    #         self.top_params = self.params.valuesdict()
    #         top_data = {'method': self.method, 'iter': iter, 'sse': sse,
    #             'time': time.strftime('%Y%m%d-%H%M%S')}
    #         for k, v in self.top_params.items():
    #             top_data[k] = v
    #         top_path = os.path.join(self.save_dir, 'params.json')
    #         with open(top_path, 'w') as fp:
    #             json.dump(top_data, fp)
    #     plt.clf()
    #     if len(self.yp.shape) == 1:
    #         plt.plot(self.tp, self.yp)
    #     else:
    #         for i in range(self.yp.shape[1]):
    #             plt.plot(self.tp, self.yp[:, i], label='p' + str(i))
    #     plt.plot(self.tp, self.y['I'], label='I')
    #     plt.plot(self.tp, self.y['A'], label='A')
    #     plt.plot(self.tp, self.y['P'], label='P')
    #     title = 'SSE: ' + str(np.format_float_scientific(sse, precision=3))
    #     for i, (k, v) in enumerate(params.valuesdict().items()):
    #         title += ' | '+ k +': ' + str(np.round(v, 3))
    #         if i > 0 and i % 4 == 0:
    #             title += '\n'
    #     plt.title(title)
    #     plt.ylabel('Output')
    #     plt.xlabel('Time')
    #     plt.legend(loc='best')
    #     if iter % 10 == 0:
    #         fig_path = os.path.join(self.fit_figs_dir, str(iter) + '.png')
    #         plt.savefig(fig_path)
    #     plt.pause(1e-6)
    #     if time.time() - self.clock > 43200:
    #         return True

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
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def render(self, mode='human'):
        if mode == 'human':
            self.display = True
        else:
            raise ValueError

    def close(self):
        self.display = False
        plt.close()


class Khammash(Env):
    """
    An inducible gene expression model accounting for:
        1. Stimuli dependent activation of a transcription factor (TF)
        2. TF-dependent transcription of mRNA and mRNA degradation
        3. Protein translation and degradation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None, save_dir='kh_model_results'):
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.display = False
        self.y = defaultdict(list)
        self.params = params
        self.sse_fit = [1e12]
        self.time_fit = [time.time()]
        self.top_params = None
        self.clock = time.time()
        self.save_dir = save_dir
        setup_dirs(save_dir)

    def fit(self, tp, up, yp, y0=None, method='powell', method_kwargs={}):
        self.fit_figs_dir = os.path.join(self.save_dir, 'fit_figs')
        setup_dirs(self.fit_figs_dir)
        self.method = method
        self.tp = tp
        self.up = up
        self.yp = yp
        if y0 is not None:
            self.y0 = y0
        elif len(self.yp.shape) == 1:
            self.y0 = [0.0, 0.0, self.yp[0]]
        else:
            self.y0 = self.yp[0]
        params = lm.Parameters()
        params.add('k_on', value=1, min=0, max=20)
        params.add('k_off', value=1, min=0, max=20)
        params.add('k_basal', value=1, min=0, max=20)
        params.add('k_max', value=1, min=0, max=20)
        params.add('k_degR', value=1, min=0, max=20)
        params.add('n', value=1, min=0, max=10)
        params.add('K_d', value=1, min=0, max=20)
        params.add('k_trans', value=1, min=0, max=20)
        params.add('k_degP', value=1, min=0, max=20)
        self.opt_results = lm.minimize(
            self.residual, params, method=method, iter_cb=self.progress,
            nan_policy='propagate', **method_kwargs
        )
        self.close()
        sse_path = os.path.join(self.save_dir, 'sse.csv')
        sse_data = np.column_stack((self.time_fit, self.sse_fit))
        np.savetxt(sse_path, sse_data, delimiter=',', header='t,sse')
        self.params = self.opt_results.params.valuesdict()
        os.system('cls' if os.name == 'nt' else 'clear')
        print(lm.report_fit(self.opt_results))

    def residual(self, params):
        self.params = params
        y = self.simulate(self.tp, self.up, self.y0)
        self.y['T'] = np.nan_to_num(y[0])
        self.y['M'] = np.nan_to_num(y[1])
        self.y['P'] = np.nan_to_num(y[2])
        if len(self.yp.shape) == 1:
            return self.y['P'] - self.yp
        else:
            return y - self.yp

    def simulate(self, t, u, y0):
        self.uf = interp1d(t, u)
        result = solve_ivp(
            fun=lambda t, y: self.model(t, y),
            t_span=[t[0], t[-1]], y0=y0, t_eval=t, method='LSODA')
        return result.y

    def model(self, t, y):
        T, M, P = y
        k_on = self.params['k_on']
        k_off = self.params['k_off']
        k_basal = self.params['k_basal']
        k_max = self.params['k_max']
        k_degR = self.params['k_degR']
        n = self.params['n']
        K_d = self.params['K_d']
        k_trans = self.params['k_trans']
        k_degP = self.params['k_degP']
        u = self.uf(t)
        T_tot = 2000
        dTdt = u*k_on*(T_tot - T) - k_off*T
        dMdt = k_basal + k_max*(T**n/(K_d**n + T**n)) - k_degR*M
        dPdt = k_trans*M - k_degP*P
        return [dTdt, dMdt, dPdt]

    def progress(self, params, iter, resid):
        sse = np.sum(resid**2)
        if sse < self.sse_fit[-1]:
            self.sse_fit.append(sse)
            self.time_fit.append(time.time())
            self.top_params = self.params.valuesdict()
            top_data = {'method': self.method, 'iter': iter, 'sse': sse,
                'time': time.strftime('%Y%m%d-%H%M%S')}
            for k, v in self.top_params.items():
                top_data[k] = v
            top_path = os.path.join(self.save_dir, 'params.json')
            with open(top_path, 'w') as fp:
                json.dump(top_data, fp)
        plt.clf()
        if len(self.yp.shape) == 1:
            plt.plot(self.tp, self.yp)
        else:
            for i in range(self.yp.shape[1]):
                plt.plot(self.tp, self.yp[:, i], label='p' + str(i))
        plt.plot(self.tp, self.y['T'], label='TF_on')
        plt.plot(self.tp, self.y['M'], label='mRNA')
        plt.plot(self.tp, self.y['P'], label='Protein')
        title = 'SSE: ' + str(np.format_float_scientific(sse, precision=3)) + '\n'
        for i, (k, v) in enumerate(params.valuesdict().items()):
            title += ' | '+ k +': ' + str(np.round(v, 3))
            if i > 0 and i % 4 == 0:
                title += '\n'
        plt.title(title)
        plt.ylabel('Output')
        plt.xlabel('Time')
        plt.legend(loc='best')
        if iter % 10 == 0:
            fig_path = os.path.join(self.fit_figs_dir, str(iter) + '.png')
            plt.savefig(fig_path)
        plt.pause(1e-6)
        if time.time() - self.clock > 43200:
            return True

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
