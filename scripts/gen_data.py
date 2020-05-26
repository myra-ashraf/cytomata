import os
import sys
import json
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps

from cytomata.model import sim_itranslo, sim_iexpress
from cytomata.utils import setup_dirs, plot, custom_styles, custom_palette


def generate_itranslo(t, y0, uf, params, results_dir, rand_y=False, rand_k=False):
    if rand_y:
        new_y = y0 * (0.2*np.random.randn() + 1)
        while np.min(new_y) <= 0:
            new_y = y0 * (0.2*np.random.randn() + 1)
        y0 = new_y
    if rand_k:
        params = {k:v + 0.1*v*np.random.randn() for k, v in params.items()}
    tm, ym = sim_itranslo(t, y0, uf, params)
    setup_dirs(results_dir)
    data = np.column_stack((tm, ym, u))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,yc,yn,u', comments='')
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True,
            figsize=(16, 10),gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        # ax1.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def scan_freqs(y_csv, paramsf, paramsf1, results_dir):
    t = np.arange(0, 1200)
    u = np.zeros_like(t)
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    ydf = pd.read_csv(y_csv)
    yc0 = ydf['yc'].values[0]
    yn0 = ydf['yn'].values[0]
    y0 = np.array([yc0, yn0])
    with open(paramsf) as f:
      params = json.load(f)
    with open(paramsf1) as f1:
      params1 = json.load(f1)
    tm, ym = sim_itranslo(t, y0, uf, params)
    y0 = ym[-1]
    tm1, ym1 = sim_itranslo(t, y0, uf, params1)
    y01 = ym1[-1]
    auc_outputs = []
    auc_inputs = []
    periods = []
    auc_outputs1 = []
    for p in range(1, 60, 1):
        u = np.zeros_like(t)
        for i in range(t[100], t[700], p):
            u[i:i+1] = 1
        uf = interp1d(t, u, bounds_error=False, fill_value=0)
        tm, ym = sim_itranslo(t, y0, uf, params)
        tm1, ym1 = sim_itranslo(t, y01, uf, params1)
        if p == 30:
            plt.plot(ym[:, 1])
            plt.plot(ym1[:, 1])
            plt.show()
        auc_inputs.append(simps(u))
        auc_outputs.append(simps(ym[:, 1]))
        periods.append(p)
        auc_outputs1.append(simps(ym1[:, 1]))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots()
        ax.plot(auc_inputs, auc_outputs, color='#1976D2', label='Slow')
        ax.plot(auc_inputs, auc_outputs1, color='#d32f2f', label='Fast')
        ax.set_xlabel('Light Exposure')
        ax.set_ylabel('Total Activity')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'scan_freqs_u.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(periods, auc_outputs, color='#1976D2', label='Slow')
        ax.plot(periods, auc_outputs1, color='#d32f2f', label='Fast')
        ax.set_xlabel('Signal Period')
        ax.set_ylabel('Total Activity')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'scan_freqs_p.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(periods, np.array(auc_outputs) - np.array(auc_outputs1), color='#1976D2')
        ax.set_xlabel('Signal Period')
        ax.set_ylabel('Slow - Fast')
        fig.savefig(os.path.join(results_dir, 'scan_freqs_diff.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


if __name__ == '__main__':
    # y_path = '/home/phuong/data/LINTAD/LexA-results/0/y.csv'
    # u_path = '/home/phuong/data/LINTAD/LexA/u0.csv'
    # params_path = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params.json'
    # results_dir = '/home/phuong/data/LINTAD/TF'
    # ydf = pd.read_csv(y_path)
    # udf = pd.read_csv(u_path)
    # td = np.around(ydf['t'].values)
    # yd = ydf['y'].values
    # t = np.around(np.arange(td[0], td[-1]))
    # yf = interp1d(td, yd)
    # y = np.array([yf(ti) for ti in t])
    # uta = np.around(udf['ta'].values)
    # utb = np.around(udf['tb'].values)
    # u = np.zeros_like(t)
    # for ta, tb in zip(uta, utb):
    #     ia = list(t).index(ta)
    #     ib = list(t).index(tb)
    #     u[ia:ib] = 1
    # y_path = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # generate_itranslo(t, u, y_path, params_path, results_dir, rand_y=False, rand_k=False)

    # params_path = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params.json'
    # y_path = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # results_dir = '/home/phuong/data/LINTAD/LINuS-1'
    # t = np.arange(0, 600)
    # u = np.zeros_like(t)
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # ydf = pd.read_csv(y_path)
    # yc0 = ydf['yc'].values[0]
    # yn0 = ydf['yn'].values[0]
    # y0 = np.array([yc0, yn0])
    # with open(params_path) as f:
    #   params = json.load(f)
    # tm, ym = sim_itranslo(t, y0, uf, params)
    # y0 = ym[-1]
    # # for i in range(180, 1980, 30):
    # #     u[i] = 1
    # # plt.plot(t, u)
    # # plt.show()
    # u[60:180] = 1
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # generate_itranslo(t, y0, uf, params, results_dir, rand_y=False, rand_k=False)

    # params_path = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params_slow.json'
    # params_path1 = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params_faster.json'
    # y_path = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # results_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    # t = np.arange(0, 1200)
    # u = np.zeros_like(t)
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # ydf = pd.read_csv(y_csv)
    # yc0 = ydf['yc'].values[0]
    # yn0 = ydf['yn'].values[0]
    # y0 = np.array([yc0, yn0])
    # with open(params_path) as f:
    #   params = json.load(f)
    # with open(params_path1) as f1:
    #   params1 = json.load(f1)
    # tm, ym = sim_itranslo(t, y0, uf, params)
    # y0 = ym[-1]
    # tm1, ym1 = sim_itranslo(t, y0, uf, params1)
    # y01 = ym1[-1]
    # u = np.zeros_like(t)
    # p = 120
    # for i in range(t[100], t[700], p):
    #     u[i:i+1] = 1
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # tm, ym = sim_itranslo(t, y0, uf, params)
    # tm1, ym1 = sim_itranslo(t, y01, uf, params1)
    # plt.plot(ym[:, 1])
    # plt.plot(ym1[:, 1])
    # plt.show()
    # scan_freqs(y_path, params_path, params_path1, results_dir)
    

    y_path = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    params_path = '/home/phuong/data/LINTAD/LINuS-results/0/opt_params.json'
    t = np.arange(0, 600)
    ydf = pd.read_csv(y_path)
    yc0 = ydf['yc'].values[0]
    yn0 = ydf['yn'].values[0]
    y0 = np.array([yc0, yn0])
    u = np.zeros_like(t)
    for i in range(t[100], t[400], 10):
            u[i:i+5] = 1
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    with open(params_path) as f:
      params = json.load(f)
    ta = time.time()
    tm, ym = sim_itranslo(t, y0, uf, params)
    print(time.time() - ta)
    plt.plot(tm, ym)
    plt.show()