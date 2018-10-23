import os

import numpy as np
import lmfit as lm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class FOPDTFitter(object):
    """
    First Order Plus Dead Time Approximation.
    """
    def __init__(self, tp, up, yp):
        self.tp = tp
        self.up = up
        self.uf = interp1d(tp, up)
        self.yp = yp
        self.best_params = lm.Parameters()
        self.best_error = 1e6
        self.optimized = None

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
        dydt = (-(y - self.yp[0]) + K*(u - self.up[0]))/tau
        return dydt

    def simulate(self, k):
        K = k['K']
        tau = k['tau']
        theta = k['theta']
        result = solve_ivp(fun=lambda t, y: self.model(t, y, K, tau, theta),
                           t_span=[self.tp[0], self.tp[-1]], method='LSODA',
                           y0=[self.yp[0]], t_eval=self.tp
        )
        return result.y[0]

    def residual(self, k):
        return self.simulate(k) - self.yp

    def iter_cb(self, params, iter, resid):
        sse = np.sum(resid**2)
        if sse < 0.1:
            return True
        if sse < self.best_error:
            self.best_error = sse
            self.best_params = params
            os.system('cls' if os.name == 'nt' else 'clear')
            print('-'*40)
            print('SSE: ' + str(sse))
            print(params.valuesdict())
            print('-'*40)

    def optimize(self):
        params = lm.Parameters()
        params.add('K', value=1.0, min=0, max=100)
        params.add('tau', value=1.0, min=0, max=100)
        params.add('theta', value=1.0, min=0, max=100)
        self.optimized = lm.minimize(
            self.residual, params, method='powell', iter_cb=self.iter_cb
        )
        os.system('cls' if os.name == 'nt' else 'clear')
        print(lm.report_fit(self.optimized))
        return self.optimized
