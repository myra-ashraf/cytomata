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


from cytomata.model.ode import sim_I1FFL
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


params = {
    'kXf': 10,
    'kXr': 10,
    'kAa': 1,
    'kAb': 1,
    'kIa': 0.5,
    'kIb': 1,
    'kc': 0.1,
    'n': 5,
}
width = 20
period = 120
y0 = [1, 0, 0, 0]
t = np.linspace(0.0, 300, 300)
u = wp_to_input(t, width, period)
# u = np.zeros(len(t))
# u[100:200] = 1.0
u += np.absolute(np.random.normal(scale=0.1, size=len(t)))
y = sim_I1FFL(t, y0, u, params)
Xi = np.transpose(y[0])
Xa = np.transpose(y[1])
Y = np.transpose(y[2])
Z = np.transpose(y[3])
plot(t, np.column_stack((u, Y, Z)), xlabel='Time (sec)', ylabel='Amount',
    title='I1-FFL Response', show=True, legend_loc='upper right',
    labels=['u', 'Y', 'Z'])
