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

from cytomata.process.extract import images_to_ave_frame_intensities
from cytomata.process.model import FOPDT, Regulator, Khammash


expt = '20181112-transf-30hrs-1sec30sec-nd8-4hrs'
data_dir = os.path.join(os.path.expanduser('~'), 'data', expt)

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
# yv = images_to_ave_frame_intensities(img_dir, save_dir=proc_imgs_dir, gauss_sigma=50)
# yf = interp1d(yt, yv, fill_value='extrapolate')
# yv = np.array([yf(t) for t in ut])
# ut /= 3600
# yv /= yv[0]
# cleaned_data_path = os.path.join(data_dir, 'processed.csv')
# pd.DataFrame({'t': ut, 'u': uv, 'y': yv}).to_csv(cleaned_data_path, encoding='utf-8', index=False)

## Load Data
data_path = os.path.join(data_dir, 'processed.csv')
df = pd.read_csv(data_path)
t = df['t'].values
u = df['u'].values
y = df['y'].values

## Fit Model
model = Khammash()
model.fit(tp=t, up=u, yp=y, method='powell', method_kwargs={})


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
