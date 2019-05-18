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


from cytomata.process.model import sim_FRC
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


def frc_response(width=1, period=10):
    """"""
    t = np.linspace(0.0, 300, 300)
    u = wp_to_input(t, width, period)
    # [N, P1, P2, M1, M2, G1, I1, G2, I2] = y
    y0 = [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params = {
        'kM1f': 1.0,
        'kM1r': 0.01,
        'kM2f': 1.0,
        'kM2r': 0.1,
        'kGa': 1.0,
        'kGb': 100.0,
        'n1': 4.0,
        'kIa': 2.0,
        'kIb': 50.0,
        'n2': 4.0,
    }
    y = sim_FRC(t, y0, u, params)
    N = np.transpose(y[0])
    P1 = np.transpose(y[1])
    P2 = np.transpose(y[2])
    M1 = np.transpose(y[3])
    M2 = np.transpose(y[4])
    G1 = np.transpose(y[5])
    I1 = np.transpose(y[6])
    G2 = np.transpose(y[7])
    I2 = np.transpose(y[8])
    return t, N, P1, P2, M1, M2, G1, I1, G2, I2


t, N, P1, P2, M1, M2, G1, I1, G2, I2 = frc_response(width=1, period=5)
plot(t, np.column_stack((M1, M2, G1, G2)), xlabel='Time (sec)', ylabel='Amount',
    title='N=1, P1=1, P2=1 | W=1s | P=5s', show=True, legend_loc='upper right',
    labels=['Magnet1', 'Magnet2', 'Gene1', 'Gene2'])

# for w in range(1, 61):
#     t, P1a, P2a, T1, R1, T2 = frc_response(params, width=1, period=w)
#     plot(t, np.column_stack((P1a, P2a, T1, R1, T2)), xlabel='Time (sec)', ylabel='Amount',
#         title='Pulsatile Input: Width=1sec, Period={0}sec'.format(w), show=False,
#         save_path='frc_demo/{0}.png'.format(w), legend_loc='upper right', ylim=(0, 1000),
#         labels=['pMag1', 'pMag2', 'Activator', 'Repressor', 'Product'])
#     print(w)


# def frc_response(params, width=1, period=5):
#     """"""
#     t = np.linspace(0.0, 300, 300)
#     u = wp_to_input(t, width, period)
#     model = TwoSystem(params=params)
#     # [Ni, Na, P1i, P1a, P2i, P2a, T1, R1, T2, G] = y
#     y0 = [1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]
#     y = model.simulate(t, u, y0)
#     P1a = np.transpose(y[3])
#     P2a = np.transpose(y[5])
#     T1 = np.transpose(y[6])
#     R1 = np.transpose(y[7])
#     T2 = np.transpose(y[8])
#     return t, P1a, P2a, T1, R1, T2
#
# params = {
#     'kNa': 1.0,
#     'kNi': 0.01,
#     'kP1a': 1.0,
#     'kP1i': 0.05,
#     'kP2a': 1.0,
#     'kP2i': 2.0,
#     'kT1f': 0.01,
#     'kT1r': 0.01,
#     'kR1f': 0.2,
#     'kR1r': 0.05,
#     'kT1a': 1.0,
#     'kT1b': 100.0,
#     'n1': 4.0,
#     'kd1': 0.01,
#     'kR1a': 1.0,
#     'kR1b': 300.0,
#     'n2': 4.0,
# }
#
# for w in range(1, 61):
#     t, P1a, P2a, T1, R1, T2 = frc_response(params, width=1, period=w)
#     plot(t, np.column_stack((P1a, P2a, T1, R1, T2)), xlabel='Time (sec)', ylabel='Amount',
#         title='Pulsatile Input: Width=1sec, Period={0}sec'.format(w), show=False,
#         save_path='frc_demo/{0}.png'.format(w), legend_loc='upper right', ylim=(0, 1000),
#         labels=['pMag1', 'pMag2', 'Activator', 'Repressor', 'Product'])
#     print(w)


# def frc_response(params, width=1, period=5):
#     """"""
#     t = np.linspace(0.0, 300, 300)
#     u = wp_to_input(t, width, period)
#     model = ThreeSystem(params=params)
#     # [Ni, Na, P1i, P1a, P2i, P2a, P3i, P3a, T1, R1, R2, T2] = y
#     y0 = [1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     y = model.simulate(t, u, y0)
#     P1a = np.transpose(y[3])
#     P2a = np.transpose(y[5])
#     P3a = np.transpose(y[7])
#     T1 = np.transpose(y[8])
#     R1 = np.transpose(y[9])
#     R2 = np.transpose(y[10])
#     T2 = np.transpose(y[11])
#     return t, P1a, P2a, P3a, T1, R1, R2, T2
#
# params = {
#     'kNa': 1.0,
#     'kNi': 0.01,
#     'kP1a': 1.0,
#     'kP1i': 1.0,
#     'kP2a': 1.0,
#     'kP2i': 0.05,
#     'kP3a': 1.0,
#     'kP3i': 3.0,
#     'kT1f': 0.2,
#     'kT1r': 0.01,
#     'kR1f': 0.1,
#     'kR1r': 0.01,
#     'kR2f': 0.5,
#     'kR2r': 0.01,
#     'kT1a': 0.3,
#     'kT1b': 100.0,
#     'n1': 4.0,
#     'kR1a': 1.0,
#     'kR1b': 200.0,
#     'n2': 4.0,
#     'kR2a': 1.0,
#     'kR2b': 50.0,
#     'n3': 4.0
# }
#
# for w in range(1, 60):
#     t, P1a, P2a, P3a, T1, R1, R2, T2 = frc_response(params, width=1, period=w)
#     plot(t, np.column_stack((P1a, P2a, P3a, T1, R1, R2, T2)), xlabel='Time (sec)', ylabel='Amount',
#         title='Pulsatile Input: Width=1sec, Period={0}sec'.format(w), show=False,
#         save_path='frc_demo3/{0}.png'.format(w), legend_loc='upper right', ylim=(0, 1000),
#         labels=['pMag1-Activator', 'pMag2-Repressor1', 'pMag3-Repressor2', 'Activator1', 'Repressor1', 'Repressor2', 'Product'])
#     print(w)



# def frc_response(params):
#     """"""
#     period = 3
#     width = 1
#     t = np.linspace(0.0, 300, 300)
#     u = wp_to_input(t, width, period)
#     model = FRC(params=params)
#     # [Ni, Na, P1i, P1a, P2i, P2a, M1, M2] = y
#     y0 = [1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0]
#     y = model.simulate(t, u, y0)
#     M1 = np.transpose(y[-5])
#     M2 = np.transpose(y[-3])
#     return t, M1, M2
#
# params = {
#     'kNa': 1.0,
#     'kNi': 0.0001,
#     'kP1a': 1.0,
#     'kP1i': 0.0001,
#     'kP2a': 1.0,
#     'kP2i': 0.1,
#     'kM1f': 0.1,
#     'kM1r': 0.2,
#     'kM2f': 0.2,
#     'kM2r': 0.5
# }
#
# t, M1, M2 = frc_response(params)
#
# plot(t, np.column_stack((M1, M2)), xlabel='Time (sec)', ylabel='Amount', title='High Frequency Pulsatile Input', show=True, labels=['Activator', 'Repressor'])

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
#
#
# optimize_lintad()


# def optimize_lintad():
#     """"""
#     data_dir = time.strftime('%Y%m%d-%H%M%S')
#     setup_dirs(data_dir)
#     # Rate Constants
#     kconsts = ['kf1', 'kr1', 'kf2', 'kr2', 'kf3', 'kr3', 'ka', 'kb', 'kc', 'kd']
#     krates = [0.001, 0.01, 0.1, 1.0]
#     nrates = [0.1, 1.0, 2.0]
#     param_grid = {}
#     for kconst in kconsts:
#         rnd.shuffle(krates)
#         param_grid[kconst] = krates
#     rnd.shuffle(nrates)
#     param_grid['n'] = nrates
#     # Inputs
#     periods = [3600, 2400, 1200, 600, 300, 60, 30, 10, 5]
#     widths = [0.01, 0.03, 0.06, 0.12, 0.20, 0.36, 0.54, 0.78, 1.0]
#     pdata = []
#     fdata = {}
#     param_grid = ParameterGrid(param_grid)
#     input_grid = ParameterGrid({'width':widths, 'period':periods})
#     t0 = time.time()
#     ta = time.time()
#     td = 100.0
#     condition = False
#     for id, params in enumerate(param_grid):
#         os.system('cls' if os.name == 'nt' else 'clear')
#         n_combos = len(param_grid)
#         eta = (n_combos - id) * td / 3600
#         print('Simulating Case #{}/{} | {}s/iter | ETA: {}hrs'.format(id, n_combos, td, eta))
#         print(params)
#         prow = {'id': id, **params}
#         ffolds = []
#         fwidths = []
#         fperiods = []
#         for inp in input_grid:
#             period = int(inp['period'])
#             width = int(round(inp['width']*period))
#             t = np.linspace(0.0, 12.0, 43200)
#             u = [1]*width + [0]*(period - width)
#             if len(t) > len(u):
#                 n = len(t) - len(u)
#                 u += [0] * n
#             elif len(t) < len(u):
#                 n = len(u) - len(t)
#                 del u[-n:]
#             model = Lintad(params=params)
#             ynames = ['LCBc', 'LCBn', 'NCVc', 'NCVn', 'AC', 'POI']
#             y0 = [1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0]
#             y = model.simulate(t, u, y0)
#             gfp = np.transpose(y[-1])
#             ff = max(gfp) - min(gfp)
#             ffolds.append(ff)
#             fwidths.append(width)
#             fperiods.append(period)
#         fdata[str(id) + '_foldch'] = ffolds
#         fdata[str(id) + '_width'] = fwidths
#         fdata[str(id) + '_period'] = fperiods
#         thres = 0.5*(max(ffolds) - min(ffolds))
#         condition = len(np.where(np.array(ffolds) > thres)[0]) < (0.2 * len(ffolds))
#         prow['pass'] = int(condition)
#         pdata.append(prow)
#         pd.DataFrame(fdata).to_csv(os.path.join(data_dir, 'lintad_sims_freqs.csv'))
#         pd.DataFrame(pdata).to_csv(os.path.join(data_dir, 'lintad_sims_params.csv'))
#         tb = time.time()
#         td = tb - ta
#         ta = tb
#         if condition:
#             print('FOUND PARAMS')
#             print(params)
#             break
#     print('Total Duration: ' + str(time.time() - t0))

# optimize_lintad()


# plot(t, np.transpose(y[-1]),
#     xlabel='time (hr)', ylabel='amount',
#     title='Simulation', show=True
# )

# expt = '20181112-transf-30hrs-1sec30sec-nd8-4hrs'
# data_dir = os.path.join(os.path.expanduser('~'), 'data', expt)

## Pre-process Data
# u_path = os.path.join(data_dir, 'u.csv')
# df_u = pd.read_csv(u_path)
# ut = []
# uv = []
# for i, row in df_u.iterrows():
#     ut += [row['t'] + i for i in range(30)]
#     uv += [row['u']] + 29*[0]
# ut = np.array(ut) - ut[0]
# y_path = os.path.join(data_dir, '0.csv')
# df_y = pd.read_csv(y_path)
# yt = df_y['t'].values - df_y['t'].values[0]
# img_dir = os.path.join(data_dir, 'imgs', 'GFP', '0')
# proc_imgs_dir = os.path.join(data_dir, 'processed_imgs')
# yv = run_frame_ave_analysis(img_dir, save_dir=proc_imgs_dir, gauss_sigma=50)
# yf = interp1d(yt, yv, fill_value='extrapolate')
# yv = np.array([yf(t) for t in ut])
# ut /= 3600
# yv /= yv[0]
# cleaned_data_path = os.path.join(data_dir, 'processed.csv')
# pd.DataFrame({'t': ut, 'u': uv, 'y': yv}).to_csv(cleaned_data_path, encoding='utf-8', index=False)

## Load Data
# data_path = os.path.join(data_dir, 'processed.csv')
# df = pd.read_csv(data_path)
# t = df['t'].values
# u = df['u'].values
# y = df['y'].values

## Fit Model
# model = Khammash()
# model.fit(tp=t, up=u, yp=y, method='powell', method_kwargs={})


# # Optimize Pattern
# def residual(params):
#     light = params['light']
#     dark = params['dark']
#     period = light + dark
#     tp = np.array(range(0, 100000))/3600
#     up = np.array([0]*(100000%period) + ([1]*light + [0]*dark) * (100000//period))
#     y = fopdt.simulate(tp, up, 0.0)
#     plt.clf()
#     plt.plot(tp, y)
#     plt.ylabel('Output')
#     plt.xlabel('Time')
#     plt.pause(1e-6)
#     return np.sum((y - 10.0)**2)
# best_res = np.inf
# best_l = 1
# best_d = 1
# for l in range(1, 30):
#     for d in range(0, 30):
#         res = residual({'light': l, 'dark': d})
#         if res < best_res:
#             os.system('cls' if os.name == 'nt' else 'clear')
#             print('Lowest SSE: ' + str(res))
#             best_res = res
#             best_l = l
#             best_d = d
# plt.close()
# os.system('cls' if os.name == 'nt' else 'clear')
# print('Best On Duration: ' + str(best_l))
# print('Best Off Duration: ' + str(best_d))
#
# # Simulate
# # best_l = 10
# # best_d = 20
# n = 200
# dt = 0.1
# # Note: dt is in hours, if take dt of the tp array, it is 1/3600
# reward_func = lambda t, u, y: -abs(y[-1] - 10.0)
# obs = fopdt.reset(y0=0.0, dt=dt, n=n, reward_func=reward_func)
# fopdt.render()
# counter = False
# count = 0
# for i in range(n):
#     if i % best_d == 0:
#         action = 1.0
#         counter = True
#     if counter:
#         count += 1
#     if count == best_l:
#         count = 0
#         counter = False
#         action = 0.0
#     obs, rew, done, info = fopdt.step(action)
