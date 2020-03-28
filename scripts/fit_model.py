import os
import sys
import time
import itertools
import random as rnd
# rnd.seed(123)
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tellurium as te
import matplotlib
matplotlib.use('tkagg')
from scipy.stats import norm
from scipy.interpolate import interp1d

from cytomata.model.ode import sim_ind_translo, sim_ind_dimer, sim_ind_gex
from cytomata.utils.visual import imshow, plot, custom_styles, custom_palette
from cytomata.utils.io import setup_dirs


def wp_to_input(t, width, period):
    ui = [1]*width + [0]*(period - width)
    u =  ui * (len(t) // len(ui))
    if len(t) > len(u):
        n = len(t) - len(u)
        u += [0] * n
    elif len(t) < len(u):
        n = len(u) - len(t)
        del u[-n:]
    return u


def rescale(vals):
    return (vals - min(vals)) / (max(vals) - min(vals))


def standardize(vals):
    return (vals - np.mean(vals)) / np.std(vals)


def sse(ym, yd):
    return np.sum((ym-yd)**2)


def fit_ind_translo(t, y0, u, yd):
    def residual(params):
        ym = sim_ind_translo(t, y0, u, params)
        # res0 = np.sum(rescale((ym[0] - yd[0])**2))
        # res1 = np.sum(rescale((ym[1] - yd[1])**2))
        res0 = sse(ym[0], yd[0])
        res1 = sse(ym[1], yd[1])
        return  res0 + res1
    def opt_iter(params, iter, resid):
        print('Iter: {} | Res: {}'.format(iter, resid))
        print(params.valuesdict())
    params = lm.Parameters()
    params.add('kf', value=0.00085, min=0, max=1)
    params.add('ku', value=0.0005, min=0, max=1)
    params.add('kr', value=0.0009, min=0, max=1)
    params.add('a', value=3.9, min=0, max=10)
    results = lm.minimize(
        residual, params,
        method='powell', iter_cb=opt_iter, nan_policy='propagate'
    )
    final_params = results.params.valuesdict()
    print(lm.report_fit(results))



def fit_ind_dimer():
    pass


def fit_ind_gex():
    pass


if __name__ == '__main__':
    y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/data.csv'
    u_csv = '/home/phuong/data/LINTAD/LINuS-results/0/u.csv'
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
    y = [yc, yn]
    y0 = (0.248, 0.235)
    # fit_ind_translo(t, y0, u, y)

    params = {'kf': 0.01, 'ku': 0.02, 'kr': 0.0108, 'a': 4}
    cm, nm = sim_ind_translo(t, y0, u, params)
    plt.plot(t, cm)
    plt.plot(t, nm)
    plt.show()
