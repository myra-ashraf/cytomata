import os
import sys
import json
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import romb, simps
from scipy.interpolate import interp1d

from cytomata.model import sim_itranslo, sim_ssl
from fit_model import fit_itranslo
from cytomata.utils import setup_dirs, plot, custom_styles, custom_palette, clear_screen


def min_data_interval(y_csv, u_csv, results_dir):
    for skipn in range(1, 8):
        ydf = pd.read_csv(y_csv)
        udf = pd.read_csv(u_csv)
        td0 = np.around(ydf['t'].values)
        ycd0 = ydf['yc'].values
        ynd0 = ydf['yn'].values
        td = td0[::skipn]
        ycd = ycd0[::skipn]
        ynd = ynd0[::skipn]
        t = np.arange(td[0], td[-1])
        ycf = interp1d(td, ycd)
        ynf = interp1d(td, ynd)
        yc = np.array([ycf(ti) for ti in t])
        yn = np.array([ynf(ti) for ti in t])
        uta = np.around(udf['ta'].values)
        utb = np.around(udf['tb'].values)
        u = np.zeros_like(t)
        for ta, tb in zip(uta, utb):
            ia = list(t).index(ta)
            ib = list(t).index(tb)
            u[ia:ib] = 1
        y = np.column_stack([yc, yn])
        y0 = y[0, :]
        uf = interp1d(t, u, bounds_error=False, fill_value=0)
        # params_i = fit_itranslo(t, y, u, results_dir, save=False)
        # tm, ym = sim_itranslo(t, y0, uf, params_i)
        # if skipn == 1:
        #     tm1 = tm
        #     ym1 = ym
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            # fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
            # ax0.plot(t, u)
            # ax0.set_yticks([0, 1])
            # ax0.set_ylabel('BL')
            # ax1.plot(tm1, ym1[:, 0], color='#BBDEFB', label='Cytoplasm (Orig)')
            # ax1.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm (Skip)')
            # ax1.plot(tm1, ym1[:, 1], color='#ffcdd2', label='Nucleus (Orig)')
            # ax1.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus (Skip)')
            # ax1.set_xlabel('Time (s)')
            # ax1.set_ylabel('AU')
            # ax1.legend(loc='best')
            # fig.tight_layout()
            # fig.canvas.draw()
            # fig.savefig(os.path.join(results_dir, 'skip{}-model.png'.format(skipn-1)),
            #     dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
            # plt.close(fig)
            fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
            ax0.plot(t, u)
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
            ax1.plot(td0, ycd0, color='#BBDEFB', label='Cytoplasm (Orig)')
            ax1.plot(td, ycd, color='#1976D2', label='Cytoplasm (Skip)')
            ax1.plot(td0, ynd0, color='#ffcdd2', label='Nucleus (Orig)')
            ax1.plot(td, ynd, color='#d32f2f', label='Nucleus (Skip)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('AU')
            ax1.legend(loc='best')
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'skip{}-data.png'.format(skipn-1)),
                dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
            plt.close(fig)


def opti_activity(y_path, params_path, results_dir):
    auc_inputs = []
    auc_outputs = []
    t = np.arange(0, 600)
    u = np.zeros_like(t)
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    ydf = pd.read_csv(y_path)
    yc0 = ydf['yc'].values[0]
    yn0 = ydf['yn'].values[0]
    y0 = np.array([yc0, yn0])
    with open(params_path) as f:
      params = json.load(f)
    tm, ym = sim_itranslo(t, y0, uf, params)
    y0 = ym[-1]
    p = 30
    for w in range(p):
        u = np.zeros_like(t)
        for i in range(t[60], t[360], p):
            u[i:i+w] = 1
        uf = interp1d(t, u, bounds_error=False, fill_value=0)
        tm, ym = sim_itranslo(t, y0, uf, params)
        auc_inputs.append(simps(u))
        auc_outputs.append(np.max(ym[:, 1]))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots()
        ax.plot(auc_inputs, auc_outputs, color='#d32f2f')
        ax.set_xlabel('AUC of Input')
        ax.set_ylabel('AUC of Output')
        fig.savefig(os.path.join(results_dir, 'max_gex_in_vs_out.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def opti_ssl(results_dir):
    setup_dirs(results_dir)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    t = np.arange(0, 600)
    u1 = np.zeros_like(t)
    u1[100:400] = 1
    u2 = np.zeros_like(t)
    w = 1
    p = 25
    for i in range(t[100], t[400], p):
        u2[i:i+w] = 1
    u1f = interp1d(t, u1, bounds_error=False, fill_value=0)
    u2f = interp1d(t, u2, bounds_error=False, fill_value=0)
    y0 = [1, 10, 1, 0, 0, 0, 0]
    # def residual(params):
    #     tm1, ym1 = sim_ssl(t, y0, u1f, params)
    #     tm2, ym2 = sim_ssl(t, y0, u2f, params)
    #     auc_contin = (simps(ym1[:, 5]) - simps(ym1[:, 6]))/(simps(ym1[:, 5]) + simps(ym1[:, 6]))
    #     auc_pulse = (simps(ym2[:, 6]) - simps(ym2[:, 5]))/(simps(ym2[:, 6]) + simps(ym2[:, 5]))
    #     res = auc_contin + auc_pulse
    #     return  res
    # def opt_iter(params, iter, res):
    #     nonlocal min_res, best_params, iter_t
    #     clear_screen()
    #     ti = time.time()
    #     print('seconds/iter:', str(ti - iter_t))
    #     iter_t = ti
    #     print('Iter: {} | Res: {}'.format(iter, res))
    #     print(params.valuesdict())
    #     if res < min_res:
    #         min_res = res
    #         best_params = params.valuesdict()
    #     print('Best so far:')
    #     print('Res:', str(min_res))
    #     print(best_params)
    # params = lm.Parameters()
    # params.add('ku', value=1, min=0.1, max=2)
    # params.add('kd', value=0.0001, min=0, max=0.01)
    # params.add('kra', value=0.1, min=0, max=1)
    # params.add('krb', value=0.1, min=0, max=1)
    # params.add('kaa', value=0.1, min=0, max=10)
    # params.add('kab', value=0.1, min=0, max=10)
    # ta = time.time()
    # results = lm.minimize(
    #     residual, params, method='differential_evolution',
    #     iter_cb=opt_iter, nan_policy='propagate', tol=1e-3
    # )
    # print('Elapsed Time: ', str(time.time() - ta))
    # opt_params = results.params.valuesdict()
    # with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
    #     json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    opt_params = dict([('ku', 1), ('kd', 0.00999556289271125), ('kra', 0.04391719864186633), ('krb', 0.9999695272034472), ('kaa', 7.294808116908854), ('kab', 9.999761936784068)])
    tm1, ym1 = sim_ssl(t, y0, u1f, opt_params)
    tm2, ym2 = sim_ssl(t, y0, u2f, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u1)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        # ax1.plot(tm1, ym1[:, 0], color='#1976D2', label='Ai')
        # ax1.plot(tm1, ym1[:, 1], color='#D32F2F', label='Bi')
        # ax1.plot(tm1, ym1[:, 2], color='#388E3C', label='C')
        # ax1.plot(tm1, ym1[:, 3], color='#7B1FA2', label='Aa')
        # ax1.plot(tm1, ym1[:, 4], color='#F57C00', label='Ba')
        ax1.plot(tm1, ym1[:, 5], color='#1976D2', label='CA')
        ax1.plot(tm1, ym1[:, 6], color='#D32F2F', label='CB')
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'opti_ssl_contin.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u2)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        # ax1.plot(tm2, ym2[:, 0], color='#1976D2', label='Ai')
        # ax1.plot(tm2, ym2[:, 1], color='#D32F2F', label='Bi')
        # ax1.plot(tm2, ym2[:, 2], color='#388E3C', label='C')
        # ax1.plot(tm2, ym2[:, 3], color='#7B1FA2', label='Aa')
        # ax1.plot(tm2, ym2[:, 4], color='#F57C00', label='Ba')
        ax1.plot(tm2, ym2[:, 5], color='#1976D2', label='CA')
        ax1.plot(tm2, ym2[:, 6], color='#D32F2F', label='CB')
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'opti_ssl_pulse.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


if __name__ == '__main__':
    # # y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # results_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    # # min_data_interval(y_csv, u_csv, results_dir)
    # params_path = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params.json'
    # y_path = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # opti_activity(y_path, params_path, results_dir)

    results_dir = '/home/phuong/data/SSL'
    opti_ssl(results_dir)