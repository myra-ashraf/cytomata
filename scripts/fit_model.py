import os
import sys
import time
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from cytomata.model.ode import sim_ind_translo, sim_ind_dimer, sim_ind_gex
from cytomata.utils.visual import imshow, plot, custom_styles, custom_palette
from cytomata.utils.io import setup_dirs


def rescale(aa):
    return (aa - min(aa)) / (max(aa) - min(aa))


def sse(aa, bb):
    return np.sum((aa - bb)**2)


def preproc_translo_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    ycd = ydf['yc'].values
    ynd = ydf['yn'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    ycf = interp1d(td, ycd)
    ynf = interp1d(td, ynd)
    yc = np.array([ycf(ti) for ti in t])
    yn = np.array([ynf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    y = np.column_stack([yc, yn])
    return t, y, u

def fit_ind_translo(t, y, u):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    def residual(params):
        tm, ym = sim_ind_translo(t, y0, uf, params)
        return  (ym - y)**2 / np.var(y)
    def opt_iter(params, iter, resid):
        print('Iter: {} | Res: {}'.format(iter, resid))
        print(params.valuesdict())
    params = lm.Parameters()
    params.add('ku', value=0.1, min=0, max=1)
    params.add('kf', value=0.1, min=0, max=1)
    params.add('kr', value=0.1, min=0, max=1)
    params.add('a', value=4, min=2, max=6)
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate'
    )
    print(lm.report_fit(results))
    return results.params.valuesdict()


if __name__ == '__main__':
    y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    u_csv = '/home/phuong/data/LINTAD/LINuS-results/0/u.csv'
    t, y, u = preproc_translo_data(y_csv, u_csv)
    fit_ind_translo(t, y, u)
