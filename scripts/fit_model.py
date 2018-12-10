from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
    int, map, next, oct, open, pow, range, round, str, super, zip)

import os
import sys
import time
import itertools
sys.path.append(os.path.abspath('../'))

import numpy as np
import lmfit as lm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from cytomata.process.model import FOPDT, Regulator


# Load Data
df = pd.read_csv('data00.csv')
tp = df['t'].values
up = df['u'].values
yp = df['y'].values
# df = pd.read_csv('data.csv')
# tp = df['time'].values/3600
# yp = df['fluo'].values*10
# up = df['light'].values
# yf = interp1d(tp, yp)
# up = np.array([0]*5 + ([1]*1 + [0]*29) * 3857)
# tp = np.array(range(0, 115715))/3600
# yp = np.array([yf(t) for t in tp])
# pd.DataFrame({'t': tp, 'u': up, 'y': yp}).to_csv('data.csv')

# Fit Model
regulator = Regulator()
regulator.fit_model(tp=tp, up=up, yp=yp, method='nelder', method_kwargs={})
#
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
