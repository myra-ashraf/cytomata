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


from cytomata.model.ode import sim_LINTAD
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


# def lintad_response(params, width=1, period=30):
#     """"""
#     t = np.linspace(0.0, 300, 300)
#     u = wp_to_input(t, width, period)
#     # [LCBc, LCBn, NCV, AC, POI] = y
#     y0 = [1000.0, 0.0, 1000.0, 0.0, 0.0]
#     y = sim_LINTAD(t, y0, u, params)
#     LCBc = np.transpose(y[0])
#     LCBn = np.transpose(y[1])
#     NCV = np.transpose(y[2])
#     AC = np.transpose(y[3])
#     POI = np.transpose(y[4])
#     return t, LCBc, LCBn, NCV, AC, POI
#
# params = {
#     'k1f': 0.092,
#     'k1r': 0.101,
#     'k2f': 0.087,
#     'k2r': 0.092,
#     'ka': 5e-5,
#     'kb': 3.66,
#     'kc': 6.33,
#     'n': 2.03,
#     'kd': 0.008,
# }
#
# t, LCBc, LCBn, NCV, AC, POI = lintad_response(params, width=1, period=30)
# plot(t, np.column_stack((LCBc, LCBn, NCV, AC, POI)), xlabel='Time (sec)', ylabel='Amount',
#     title='LINTAD Response (BL=1sec|30sec)', show=True, legend_loc='upper right',
#     labels=['LCBc', 'LCBn', 'NCV', 'AC', 'POI'])

# params = {
#     'k1f': 1,
#     'k1r': 0.001,
#     'k2f': 1,
#     'k2r': 0.001,
#     'ka': 10,
#     'kb': 1,
#     'kc': 0.001,
#     'n': 5,
# }
# y0 = [1, 0, 1, 0, 0]
# ydata = pd.read_csv('y.csv')
# udata = pd.read_csv('u.csv')
# y = rescale(np.array(ydata['y']))
# u = np.array(udata['u'])
# # u = np.ones(len(u))
# ut = np.array(udata['t'])
# # ut = ut/3600
# yt = np.linspace(ut[0], ut[-1], len(y))
# yf = interp1d(yt, y)
# ydata = yf(ut)
# t = np.sort(ut)
# ypred = sim_LINTAD(t, y0, u, params)
# LCBc = np.transpose(ypred[0])
# LCBn = np.transpose(ypred[1])
# NCV = np.transpose(ypred[2])
# AC = np.transpose(ypred[3])
# POI = np.transpose(ypred[4])
# plot(t, np.column_stack((AC, POI)), xlabel='Time (s)', ylabel='Amount',
#     title='LINTAD Response', show=True, legend_loc='upper right',
#     labels=['AC', 'POI'])


def lintad_residual(params, ydata, t, y0, u):
    y = sim_LINTAD(t, y0, u, params)
    ypred = np.transpose(y[4])
    sse = np.sum(np.square(ypred - ydata))
    if np.isnan(sse).all():
        sse = 1e12
    return sse


def opt_iter(params, iter, resid, *args):
    print('Iter: {} | Res: {}'.format(iter, resid))
    print(params.valuesdict())


def optimize_lintad():
    params = lm.Parameters()
    params.add('k1f', value=0.5, min=0, max=1)
    params.add('k1r', value=0.001, min=0, max=1)
    params.add('k2f', value=0.5, min=0, max=1)
    params.add('k2r', value=0.001, min=0, max=1)
    params.add('ka', value=1, min=0, max=1000)
    params.add('kb', value=1, min=0, max=1000)
    params.add('kc', value=0.001, min=0, max=10)
    params.add('n', value=3, min=1, max=6)
    y0 = [1.0, 0.0, 1.0, 0.0, 0.0]
    ydata = pd.read_csv('y.csv')
    udata = pd.read_csv('u.csv')
    y = rescale(np.array(ydata['y']))
    u = np.array(udata['u'])
    # u = np.ones(len(u))
    ut = np.array(udata['t'])
    # ut = ut/3600
    yt = np.linspace(ut[0], ut[-1], len(y))
    yf = interp1d(yt, y)
    ydata = yf(ut)
    t = np.sort(ut)
    results = lm.minimize(
        lintad_residual, params, args=(ydata, t, y0, u),
        method='powell', iter_cb=opt_iter, nan_policy='omit',
        tol=1e-5
    )
    final_params = results.params.valuesdict()
    print(lm.report_fit(results))
    y = sim_LINTAD(t, y0, u, final_params)
    LCBc = np.transpose(y[0])
    LCBn = np.transpose(y[1])
    NCV = np.transpose(y[2])
    AC = np.transpose(y[3])
    POI = np.transpose(y[4])
    plot(t, np.column_stack((POI, ydata)), xlabel='Time (s)', ylabel='Amount',
        title='LINTAD Response', show=True, legend_loc='upper right',
        labels=['Model', 'Data'])
    return final_params


optimize_lintad()
