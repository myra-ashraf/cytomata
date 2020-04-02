import os
import sys
import time
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
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

def fit_ind_translo(t, y, u, results_dir):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # def residual(params):
    #     tm, ym = sim_ind_translo(t, y0, uf, params)
    #     res = (ym - y)**2 / np.var(y)
    #     return  res.flatten()
    # def opt_iter(params, iter, res):
    #     res = np.sum(res)
    #     print('Iter: {} | Res: {}'.format(iter, res))
    #     print(params.valuesdict())
    # params = lm.Parameters()
    # params.add('ku', value=0.001, min=0, max=1)
    # params.add('kf', value=0.001, min=0, max=1)
    # params.add('kr', value=0.001, min=0, max=1)
    # params.add('a', value=4, min=2, max=6)
    # results = lm.minimize(
    #     residual, params, method='nelder',
    #     iter_cb=opt_iter, nan_policy='propagate'
    # )
    # print(lm.report_fit(results))
    # opt_params = results.params.valuesdict()
    opt_params = {'ku': 0.002113, 'kf': 0.0003146, 'kr': 0.0006273, 'a': 3.9997968}
    tm, ym = sim_ind_translo(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(t, y[:, 0], color='#BBDEFB', label='Cytoplasm (Data)')
        ax.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm (Model)')
        ax.plot(t, y[:, 1], color='#ffcdd2', label='Nucleus (Data)')
        ax.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus (Model)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence Intensity')
        ax.legend(loc='best')
        # fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    plot(t, u, ylabel='BL State', yticks=[0, 1], xticks=[], figsize=(10, 2),
        save_path=os.path.join(results_dir, 'bl.png'))

if __name__ == '__main__':
    y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    u_csv = '/home/phuong/data/LINTAD/LINuS-results/0/u.csv'
    res_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    t, y, u = preproc_translo_data(y_csv, u_csv)
    # y0 = y[0, :]
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # params = {'ku': 0.0021, 'kf': 0.00031, 'kr': 0.00063, 'a': 4}
    # tm, ym = sim_ind_translo(t, y0, uf, params)
    # plt.plot(tm, ym)
    # plt.show()
    fit_ind_translo(t, y, u, res_dir)
