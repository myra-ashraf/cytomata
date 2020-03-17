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
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.model_selection import ParameterGrid


from cytomata.model.ode import sim_FRC
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


def fit_ind_dimer():
    pass


def fit_ind_translo():
    pass


def fit_ind_gex():
    pass


# def lintad_response(params):
#     """"""
#     periods = [60]
#     widths = np.arange(0.01, 1.0, 0.02)
#     ffolds = []
#     for period in periods:
#         for width in widths:
#             period = int(period)
#             width = int(round(width*period))
#             t = np.linspace(0.0, 6.0, 21600)
#             u = wp_to_input(t, width, period)
#             model = Lintad(params=params)
#             y0 = [1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0]
#             y = model.simulate(t, u, y0)
#             poi = np.transpose(y[-1])
#             ff = max(poi) - min(poi)
#             ffolds.append(ff)
#     return np.array(widths), np.array(ffolds)


# def lintad_residual(params):
#     widths, ffolds = lintad_response(params)
#     ffolds = rescale(ffolds)
#     xvals = rescale(widths)
#     rv = norm(loc=0.5, scale=0.05)
#     gvals = rescale(rv.pdf(xvals))
#     savep = os.path.join('lintad', time.strftime('%Y%m%d-%H%M%S') + '.jpg')
#     plot(xvals, np.column_stack((ffolds, gvals)), save_path=savep)
#     sse = np.sum(np.square(ffolds - gvals))
#     # xslice = len(ffolds) // 3
#     # yslice0 = ffolds[:xslice]
#     # yslice1 = ffolds[xslice:(2*xslice)]
#     # yslice2 = ffolds[(2*xslice):]
#     # nhigh0 = (yslice0 > 0.5).sum() / len(yslice0)
#     # nhigh1 = (yslice1 > 0.5).sum() / len(yslice1)
#     # nhigh2 = (yslice2 < 0.5).sum() / len(yslice2)
#     # n_high_vals = (ffolds > 0.3).sum() / len(ffolds)
#     return sse
#
#
# def opt_iter(params, iter, resid):
#     print('Iter: {} | Res: {}'.format(iter, resid))
#     print(params.valuesdict())
#
#
# def optimize_lintad():
#     setup_dirs('lintad')
#     params = lm.Parameters()
#     params.add('kf1', value=1, min=0, max=1)
#     params.add('kr1', value=0.1, min=0, max=1)
#     params.add('kf2', value=1, min=0, max=1)
#     params.add('kr2', value=1e-5, min=0, max=1)
#     params.add('kf3', value=1, min=0, max=1)
#     params.add('kr3', value=1, min=0, max=1)
#     params.add('ka', value=1, min=0, max=1, vary=False)
#     params.add('kb', value=1, min=0, max=1, vary=False)
#     params.add('kc', value=1, min=0, max=1, vary=False)
#     params.add('n', value=1, min=1e-3, max=4, vary=False)
#     params.add('kd', value=1e-3, min=0, max=1, vary=False)
#     results = lm.minimize(
#         lintad_residual, params,
#         method='differential_evolution', iter_cb=opt_iter, nan_policy='propagate'
#     )
#     final_params = results.params.valuesdict()
#     print(lm.report_fit(results))
#     return final_params


if __name__ == '__main__':
    pass