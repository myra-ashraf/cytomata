import numpy as np
import lmfit as lm
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class FOPDT(object):
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
                           t_span=[self.tp[0], self.tp[-1]],
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
            print('-'*20)
            print('SSE: ' + str(sse))
            print(params.valuesdict())
            print('-'*20)

    def optimize(self):
        params = lm.Parameters()
        params.add('K', value=1.0, min=0, max=100)
        params.add('tau', value=1.0, min=0, max=100)
        params.add('theta', value=1.0, min=0, max=100)
        self.optimized = lm.minimize(
            self.residual, params, method='powell', iter_cb=self.iter_cb
        )
        return self.optimized

    # def plot(self):
    #     ym = self.simulate(self.optimized.params)
    #     partext = {k:np.round(v, 5) for (k, v) in self.optimized.params.valuesdict().items()}
    #     fig = plt.figure(dpi=150)
    #     plt.plot(self.tp, self.yp, 'o', label='Process Data')
    #     plt.xlabel('Time (hrs)')
    #     plt.ylabel('Intensity')
    #     plt.plot(self.tp, ym, label='Optimized Fit')
    #     plt.legend(loc='best')
    #     fig.text(x=0.121, y=0.96, fontsize=28, weight='bold', alpha=0.9,
    #              s='FOPDT Model')
    #     fig.text(x=0.121, y=0.92,fontsize=18, alpha=0.7,
    #              s='K={0} | tau={1} | theta={2}'.format(
    #                  partext['K'], partext['tau'], partext['theta']))
    #     plt.show()
    #     plt.savefig('fopdt.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    tp = df['time'].values/3600.
    up = df['light'].values
    yp = df['fluo'].values*10
    fopdt = FOPDT(tp, up, yp)
    fopdt.optimize()
    print(lm.report_fit(fopdt.optimized))
