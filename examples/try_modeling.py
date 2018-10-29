import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import lmfit as lm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from cytomata.process import FOPDT


# Load Data
df = pd.read_csv('data.csv')
tp = df['time'].values/3600
yp = df['fluo'].values*100
yf = interp1d(tp, yp)
# up = df['light'].values
up = np.array([0]*5 + ([1]*30) * 3857)
tp = np.array(range(0, 115715))/3600
yp = np.array([yf(t) for t in tp])

# Fit Model
fopdt = FOPDT()
t0 = time.time()
fopdt.fit_model(tp=tp, up=up, yp=yp, K0=None, tau0=None, theta0=None, method='differential_evolution')
t1 = time.time()

# Simulate
n = 100
reward_func = lambda t, u, y: -abs(y[-1] - np.mean(yp))
obs = fopdt.reset(y0=0.0, dt=1.0, n=n, reward_func=reward_func)
fopdt.render()
t2 = time.time()
step_dur = []
for i in range(n):
    t3 = time.time()
    action = 1.0
    if i > 50:
        action = 0.0
    obs, rew, done, info = fopdt.step(action)
    t4 = time.time()
    step_dur.append(t4 - t3)
print('Fitting Duration: ' + str(t1 - t0))
print('Reset Duration: ' + str(t2 - t1))
print('Step Duration: ' + str(np.mean(step_dur)))
print('Total Duration: ' + str(t4 - t0))
