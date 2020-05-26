import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from cytomata.model import sim_itranslo, sim_idimer, sim_iexpress, sim_ssl
from cytomata.utils import setup_dirs, clear_screen, plot, custom_styles, custom_palette, rescale


def prep_itranslo_data(y_csv, u_csv):
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


def fit_itranslo(t, y, u, results_dir):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_itranslo(t, y0, uf, params)
        res = np.mean(np.square(ym - y))
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    dyc = np.median(np.absolute(np.ediff1d(y[:, 0])))
    dyn = np.median(np.absolute(np.ediff1d(y[:, 1])))
    a0 = dyn/dyc
    kmax = 10**np.floor(np.log10(np.max(y)))
    dk = kmax/10
    params = lm.Parameters()
    params.add('ku', value=kmax/2, min=0, max=kmax)
    params.add('kf', value=kmax/2, min=0, max=kmax)
    params.add('kr', value=kmax/2, min=0, max=kmax)
    params.add('a', value=a0, min=1, max=10)
    # ta = time.time()
    # results = lm.minimize(
    #     residual, params, method='differential_evolution',
    #     iter_cb=opt_iter, nan_policy='propagate', tol=1e-1
    # )
    # print('Elapsed Time: ', str(time.time() - ta))
    # opt_params = results.params.valuesdict()
    opt_params = dict([('ku', 0.0008942295681229174), ('kf', 0.0005048121271898231), ('kr', 0.0006244090506147587), ('a', 4.632731164948481)])
    # with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
    #     json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_itranslo(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t, y[:, 0], color='#BBDEFB', label='Cytoplasm (Data)')
        ax1.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm (Model)')
        ax1.plot(t, y[:, 1], color='#ffcdd2', label='Nucleus (Data)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus (Model)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def prep_idimer_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    yad = ydf['ya'].values
    ydd = ydf['yd'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    yaf = interp1d(td, yad)
    ydf = interp1d(td, ydd)
    ya = np.array([yaf(ti) for ti in t])
    yd = np.array([ydf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    y = np.column_stack([ya, ya, yd])
    return t, y, u

def fit_idimer(t, y, u, results_dir):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_idimer(t, y0, uf, params)
        res = np.sum((ym - y)**2)
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    kmax = 10**np.floor(np.log10(np.max(y)))
    params = lm.Parameters()
    params.add('ku', value=0.1, min=0, max=1)
    params.add('kf', value=0.01, min=0, max=0.1)
    params.add('kr', value=0.01, min=0, max=0.1)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-7
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_idimer(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t, y[:, 2], color='#ffcdd2', label='AB (Data)')
        ax1.plot(tm, ym[:, 2], color='#d32f2f', label='AB (Model)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def prep_iexpress_data(y_csv, x_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    xdf = pd.read_csv(x_csv)
    udf = pd.read_csv(u_csv)
    t = np.around(ydf['t'].values/60)
    t = np.arange(t[0], t[-1])
    y = ydf['y'].values[:-1]
    x = xdf['yn'].values[::60]
    uta = np.around(udf['ta'].values/60)
    utb = np.around(udf['tb'].values/60)
    u = np.zeros_like(t)
    ia = list(t).index(uta[0])
    ib = list(t).index(utb[-1])
    u[ia:ib] = 1
    # plt.plot(t, u)
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.show()
    return t, y, x, u


def fit_iexpress(t, y, x, u, results_dir):
    y0 = [0, y[0]]
    xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_iexpress(t, y0, xf, params)
        res = np.mean((ym[:, 1] - y)**2)
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    ta = time.time()
    # params = lm.Parameters()
    # params.add('ka', min=0.01, max=1, brute_step=0.01)
    # params.add('kb', min=0.1, max=10, brute_step=0.1)
    # params.add('kc', min=0.001, max=0.1, brute_step=0.001)
    # params.add('n', min=1, vary=False)
    # results = lm.minimize(
    #     residual, params, method='brute',
    #     iter_cb=opt_iter, nan_policy='propagate',
    # )
    params = lm.Parameters()
    params.add('ka', value=0.1, min=0, max=1)
    params.add('kb', value=0.1, min=0, max=1)
    params.add('kc', value=0.1, min=0, max=1)
    params.add('n', value=1, min=0, max=1)
    params.add('kf', value=0.1, min=0, max=1)
    params.add('kg', value=0.1, min=0, max=1)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='dual_annealing',
        iter_cb=opt_iter, nan_policy='propagate'
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = params
    opt_params = results.params.valuesdict()
    with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    # opt_params = dict([('kf', 0.35964911635371705), ('kr', 1.9251102650885699e-07), ('ka', 0.00011779416929114106), ('kb', 0.0017298358328363683), ('kc', 0.00272869942333942), ('n', 9.999636729801574)])
    tm, ym = sim_iexpress(t, y0, xf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.scatter(t, y, color='#ffcdd2', label='POI (Data)')
        ax1.plot(tm, ym[:, 0], color='#2196F3', label='mRNA (Model)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='POI (Model)')
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


if __name__ == '__main__':
    # y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    # t, y, u = prep_itranslo_data(y_csv, u_csv)
    # y_csv = '/home/phuong/data/LINTAD/LINuS-1/y.csv'
    # res_dir = '/home/phuong/data/LINTAD/LINuS-1/'
    # df = pd.read_csv(y_csv)
    # t = df['t'].values
    # y = np.column_stack((df['yc'].values, df['yn'].values))
    # u = df['u'].values
    # fit_itranslo(t, y, u, res_dir)


    # y_csv = '/home/phuong/data/LINTAD/CAD-results/0/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/CAD/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/CAD-results/0/'
    # t, y, u = prep_idimer_data(y_csv, u_csv)
    # y0 = y[0, :]
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # params = {'ku': 0.09, 'kf': 0.0003, 'kr': 0.01}
    # tm, ym = sim_idimer(t, y0, uf, params)
    # plt.plot(tm, ym[:, 2])
    # plt.show()
    # fit_idimer(t, y, u, res_dir)


    # y_csv = '/home/phuong/data/LINTAD/LexA-results/0/y.csv'
    # x_csv = '/home/phuong/data/LINTAD/TF/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LexA/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LexA-results/0/'
    # t, y, x, u = prep_iexpress_data(y_csv, x_csv, u_csv)
    # fit_iexpress(t, y, x, u, res_dir)
    
    # xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    # y0 = [0, y[0]]
    # params = {'kf': 0.434, 'kr': 0.00174, 'ka': 0.000122, 'kb': 0.0041, 'kc': 0.00288, 'n': 10}
    # ta = time.time()
    # tm, ym = sim_iexpress(t, y0, xf, params)
    # print(time.time() - ta)
    # plt.plot(t, y)
    # plt.plot(tm, ym[:, 1])
    # plt.show()
    pass