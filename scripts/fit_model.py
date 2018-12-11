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

from cytomata.process.extract import images_to_median_frame_intensities
from cytomata.process.extract import interpolate_to_seconds
from cytomata.process.model import FOPDT, Regulator, Khammash


# Load Data
data_dir = 'data'
u_path = os.path.join(data_dir, 'u.csv')
df = pd.read_csv(u_path)
tu = df['time'].values/3600
u = df['input'].values
# up = np.array([0]*5 + ([1]*1 + [0]*29) * 3857)
ty_path = os.path.join(data_dir, '0.csv')
ty = pd.read_csv(ty_path)['t'].values/3600
img_dir = os.path.join(data_dir, 'imgs', 'GFP')
y = images_to_median_frame_intensities(img_dir, save_dir=None, gauss_sigma=40)
yf = interp1d(ty, y)
y = np.array([yf(t) for t in tu])
cleaned_data_path = os.path.join(data_dir, 'data.csv')
pd.DataFrame({'t': tu, 'u': u, 'y': y}).to_csv(cleaned_data_path)

# Fit Model
model = Khammash()
model.fit(tp=tu, up=u, yp=y, method='nelder', method_kwargs={})
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
