import os
import sys
import shutil
import warnings
from collections import deque, defaultdict
from itertools import cycle
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea, AnchoredSizeBar
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm
from natsort import natsorted
from scipy.stats import sem
from imageio import mimwrite
from scipy.interpolate import interp1d
from scipy.integrate import simps
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage import img_as_float
from skimage.io import imread

from cytomata.GHT import threshold_GHT
from cytomata.process import preprocess_img, segment_object, segment_clusters
from cytomata.utils import setup_dirs, list_img_files, rescale, plot, custom_styles, custom_palette


def test_preprocess_img(img_path):
    img, raw, bkg = preprocess_img(img_path)
    plt.imshow(img, cmap='turbo')
    plt.show()


def test_segment_object(img_path):
    img, raw, bkg, den = preprocess_img(img_path)
    thr = segment_object(den)
    plt.imshow(den, cmap='turbo')
    plt.contour(thr, linewidths=0.3, colors='w')
    plt.show()


def test_GHT(img_path):
    img = img_as_float(imread(img_path))
    thr = threshold_GHT(img)
    print(thr)
    plt.imshow((img > thr), cmap='gray')
    # plt.contour(thr, linewidths=0.3, colors='w')
    plt.show()


def test_segment_translo(bilinus_img_path, nucleus_img_path):
    nucl_img, nucl_tval = preprocess_img(nucleus_img_path)
    cell_img, cell_tval = preprocess_img(bilinus_img_path)
    nucl_thr = segment_object(nucl_img, cb=5, er=3)
    cell_thr = segment_object(cell_img, er=5)
    plt.imshow(cell_img, cmap='turbo')
    plt.contour(cell_thr, linewidths=0.3, colors='w')
    plt.contour(nucl_thr, linewidths=0.2, colors='r')
    plt.show()


def test_segment_clusters(cad_img_path):
    cell, img = segment_object(cad_img_path, offset=5, er=9)
    dots, _ = segment_clusters(cad_img_path)
    dots = np.logical_and(dots, cell)
    anti = np.logical_xor(dots, cell)
    plt.imshow(img, cmap='turbo')
    plt.contour(anti, linewidths=0.1, colors='w')
    plt.contour(dots, linewidths=0.1, colors='r')
    plt.show()


def process_translo(nucleus_dir, translo_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    yc = []
    yn = []
    nu_imgs = []
    tr_imgs = []
    nu_imgfs = list_img_files(nucleus_dir)
    tr_imgfs = list_img_files(translo_dir)
    n_imgs = min(len(nu_imgfs), len(tr_imgfs))
    nu_tval = None
    tr_tval = None
    for i, (nu_imgf, tr_imgf) in enumerate(tqdm(zip(nu_imgfs, tr_imgfs), total=n_imgs)):
        nu_img, nu_tval = preprocess_img(nu_imgf, tval=nu_tval)
        tr_img, tr_tval = preprocess_img(tr_imgf, tval=tr_tval)
        nucl = segment_object(nu_img, er=7, cb=5)
        cell = segment_object(tr_img, er=7)
        cell = np.logical_or(cell, nucl)
        cyto = np.logical_xor(cell, nucl)
        nroi = nucl*tr_img
        croi = cyto*tr_img
        nnz = nroi[np.nonzero(nroi)]
        cnz = croi[np.nonzero(croi)]
        nucl_int = np.median(nnz)
        cyto_int = np.median(cnz)
        fname = os.path.splitext(os.path.basename(tr_imgf))[0]
        t.append(np.float(fname))
        yn.append(nucl_int)
        yc.append(cyto_int)
        if i == 0:
            nucl_cmin = np.min(nu_img)
            nucl_cmax = 1.1*np.max(nu_img)
            cell_cmin = np.min(tr_img)
            cell_cmax = 1.1*np.max(tr_img)
        figh = nu_img.shape[0]/100
        figw = nu_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(nu_img, cmap='turbo')
        axim.set_clim(nucl_cmin, nucl_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        nu_imgs.append(cmimg)
        plt.close(fig)
        figh = tr_img.shape[0]/100
        figw = tr_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(tr_img, cmap='turbo')
        axim.set_clim(cell_cmin, cell_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        tr_imgs.append(cmimg)
        plt.close(fig)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(tr_img, cmap='turbo')
            axim.set_clim(cell_cmin, cell_cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(cyto, linewidths=1.0, colors='w')
                ax.contour(nucl, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, yc, yn))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,yc,yn', comments='')
    y = np.column_stack((yc, yn))
    plot(t, y, xlabel='Time (s)', ylabel='AU',
        labels=['Cytoplasm', 'Nucleus'], save_path=os.path.join(results_dir, 'plot.png'))
    mimwrite(os.path.join(results_dir, 'nucl.gif'), nu_imgs, fps=n_imgs//12)
    mimwrite(os.path.join(results_dir, 'cell.gif'), tr_imgs, fps=n_imgs//12)


def combine_translo(results_dir):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots()
        combined_yc = pd.DataFrame()
        combined_yn = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
            csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
            data = pd.read_csv(csv_fp)
            t = data['t'].values
            yc = data['yc'].values
            yn = data['yn'].values
            ycf = interp1d(t, yc, fill_value='extrapolate')
            ynf = interp1d(t, yn, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            yc = pd.Series([ycf(ti) for ti in t], index=t, name=i)
            yn = pd.Series([ynf(ti) for ti in t], index=t, name=i)
            combined_yc = pd.concat([combined_yc, yc], axis=1)
            combined_yn = pd.concat([combined_yn, yn], axis=1)
            ax.plot(yc, color='#BBDEFB', linewidth=3)
            ax.plot(yn, color='#ffcdd2', linewidth=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        fig.savefig(os.path.join(results_dir, 'yi.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        yc_ave = combined_yc.mean(axis=1).rename('yc_ave')
        yc_std = combined_yc.std(axis=1).rename('yc_std')
        yc_sem = combined_yc.sem(axis=1).rename('yc_sem')
        yn_ave = combined_yn.mean(axis=1).rename('yn_ave')
        yn_std = combined_yn.std(axis=1).rename('yn_std')
        yn_sem = combined_yn.sem(axis=1).rename('yn_sem')
        data = pd.concat([yc_ave, yc_std, yc_sem, yn_ave, yn_std, yn_sem], axis=1).dropna()
        data.to_csv(os.path.join(results_dir, 'y_ave.csv'))
        yc_ave = data['yc_ave']
        yn_ave = data['yn_ave']
        yc_ci = data['yc_sem']*1.96
        yn_ci = data['yn_sem']*1.96
        fig, ax = plt.subplots()
        ax.plot(yc_ave, color='#1976D2', label='Ave Cytoplasmic')
        ax.plot(yn_ave, color='#d32f2f', label='Ave Nuclear')
        ax.fill_between(t, (yc_ave - yc_ci), (yc_ave + yc_ci), color='#1976D2', alpha=.1)
        ax.fill_between(t, (yn_ave - yn_ci), (yn_ave + yn_ci), color='#d32f2f', alpha=.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'ave.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def process_cad(img_dir, results_dir):
    """Analyze cad dataset and generate figures."""
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    ya = []
    yd = []
    imgs = []
    tval = None
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, tval = preprocess_img(imgf, tval)
        cell = segment_object(img, er=7)
        dots = segment_clusters(img)
        dots = np.logical_and(dots, cell)
        anti = np.logical_xor(dots, cell)
        croi = cell * img
        aroi = anti * img
        droi = dots * img
        cnz = croi[np.nonzero(croi)]
        anz = aroi[np.nonzero(aroi)]
        dnz = droi[np.nonzero(droi)]
        cell_area = len(cnz)/39.0625
        anti_area = len(anz)/39.0625
        dots_area = len(dnz)/39.0625
        if i == 0:
            cmin = np.min(img)
            cmax = 1.1*np.max(img)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anti_int = np.nan_to_num(np.median(anz)) * (anti_area / cell_area)
            dots_int = np.nan_to_num(np.median(dnz) - np.median(anz)) * (dots_area / cell_area)
        t.append(np.float(fname))
        ya.append(anti_int)
        yd.append(dots_int)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img, cmap='turbo')
            axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(cell, linewidths=0.4, colors='w')
                ax.contour(dots, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            imgs.append(np.array(fig.canvas.renderer._renderer))
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, ya, yd))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,ya,yd', comments='')
    y = np.column_stack((ya, yd))
    plot(t, y, xlabel='Time (s)', ylabel='AU',
        labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot.png'))
    y = np.column_stack((rescale(ya), rescale(yd)))
    plot(t, y, xlabel='Time (s)', ylabel='AU',
        labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot01.png'))
    mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
    return t, ya, yd


def combine_cad(results_dir):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots()
        combined_ya = pd.DataFrame()
        combined_yd = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
            csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
            data = pd.read_csv(csv_fp)
            t = data['t'].values
            ya = data['ya'].values
            yd = data['yd'].values
            yaf = interp1d(t, ya, fill_value='extrapolate')
            ydf = interp1d(t, yd, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            ya = pd.Series([yaf(ti) for ti in t], index=t, name=i)
            yd = pd.Series([ydf(ti) for ti in t], index=t, name=i)
            combined_ya = pd.concat([combined_ya, ya], axis=1)
            combined_yd = pd.concat([combined_yd, yd], axis=1)
            ax.plot(ya, color='#BBDEFB')
            ax.plot(yd, color='#ffcdd2')
        ya_ave = combined_ya.mean(axis=1).rename('ya_ave')
        ya_std = combined_ya.std(axis=1).rename('ya_std')
        ya_sem = combined_ya.sem(axis=1).rename('ya_sem')
        ya_ci = 1.96*ya_sem
        yd_ave = combined_yd.mean(axis=1).rename('yd_ave')
        yd_std = combined_yd.std(axis=1).rename('yd_std')
        yd_sem = combined_yd.sem(axis=1).rename('yd_sem')
        yd_ci = 1.96*yd_sem
        combined_data = pd.concat([ya_ave, ya_std, ya_sem, yd_ave, yd_std, yd_sem], axis=1)
        combined_data.to_csv(os.path.join(results_dir, 'y.csv'))
        ax.plot(ya_ave, color='#1976D2', label='Anti Region Ave')
        ax.plot(yd_ave, color='#d32f2f', label='Dots Region Ave')
        ax.fill_between(t, (ya_ave - ya_ci), (ya_ave + ya_ci), color='#1976D2', alpha=.1)
        ax.fill_between(t, (yd_ave - yd_ci), (yd_ave + yd_ci), color='#d32f2f', alpha=.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'combined.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def combine_uy(root_dir, fold_change=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        combined_y = pd.DataFrame()
        combined_uta = pd.DataFrame()
        combined_utb = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv_fp = os.path.join(root_dir, data_dir, 'y.csv')
            y_data = pd.read_csv(y_csv_fp)
            t = y_data['t'].values
            y = y_data['y'].values
            yf = interp1d(t, y, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            y = pd.Series([yf(ti) for ti in t], index=t, name=i)
            if fold_change:
                y = y/y[0]
            combined_y = pd.concat([combined_y, y], axis=1)
            u_csv_fp = os.path.join(root_dir, data_dir, 'u.csv')
            u_data = pd.read_csv(u_csv_fp)
            uta = np.around(u_data['ta'].values, 1)
            utb = np.around(u_data['tb'].values, 1)
            uta = pd.Series(uta, name=i)
            utb = pd.Series(utb, name=i)
            combined_uta = pd.concat([combined_uta, uta], axis=1)
            combined_utb = pd.concat([combined_utb, utb], axis=1)
            ax.plot(y, color='#1976D2', alpha=0.5, linewidth=2)
        y_ave = combined_y.mean(axis=1).rename('y_ave')
        y_std = combined_y.std(axis=1).rename('y_std')
        y_sem = combined_y.sem(axis=1).rename('y_sem')
        data = pd.concat([y_ave, y_std, y_sem], axis=1).dropna()
        data.to_csv(os.path.join(root_dir, 'y_combined.csv'))
        y_ave = data['y_ave']
        y_ci = data['y_sem']*1.96
        uta_ave = combined_uta.mean(axis=1).rename('uta_ave')
        utb_ave = combined_utb.mean(axis=1).rename('utb_ave')
        tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
        u = np.zeros_like(tu)
        for ta, tb in zip(uta, utb):
            ia = list(tu).index(ta)
            ib = list(tu).index(tb)
            u[ia:ib] = 1
        ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#1976D2', alpha=.2, label='95% CI')
        ax.plot(y_ave, color='#1976D2', label='Ave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        if fold_change:
            ax.set_ylabel('Fold Change')
        ax.legend(loc='best')
        ax0.plot(tu, u, color='#1976D2')
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        if not fold_change:
            plot_name = 'y_combined_AU.png'
        else:
            plot_name = 'y_combined.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_nes_nls(img_dir, results_dir, u_csv=None):
    """Analyze gene expression dataset and generate figures."""
    setup_dirs(os.path.join(results_dir, 'imgs'))
    setup_dirs(os.path.join(results_dir, 'debug'))
    t = []
    yc = []
    yn = []
    imgs = []
    t_bl_on = []
    for imgf in list_img_files(img_dir):
        t.append(np.float(os.path.splitext(os.path.basename(imgf))[0]))
    if u_csv is not None:
        shutil.copyfile(u_csv, os.path.join(results_dir, 'u.csv'))
        udf = pd.read_csv(u_csv)
        t = np.around(t)
        tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
        uta = np.around(udf['ta'].values, 1)
        utb = np.around(udf['tb'].values, 1)
        u = np.zeros_like(tu)
        for ta, tb in zip(uta, utb):
            t_bl_on += list(np.arange(round(ta, 1), round(tb, 1) + 0.01, 0.1))
            ia = list(tu).index(ta)
            ib = list(tu).index(tb)
            u[ia:ib] = 1
    t_annotate_bl = []
    for tbl in t_bl_on:
        t_annotate_bl.append(min(t, key=lambda ti : abs(ti - tbl)))
    c_thr = None
    n_thr = None
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
            fname = os.path.splitext(os.path.basename(imgf))[0]
            img, raw, bkg = preprocess_img(imgf)
            if i == 0:
                c_thr = segment_object(img, method='triangle', rs=6000, fh=1000, offset=-5, er=13)
                n_thr = segment_object(img, method='otsu', offset=-50)
                cell = np.logical_or(c_thr, n_thr)
                c_thr = np.logical_xor(cell, n_thr)
            # if len(n_thrs) > 100:
            #     n_thrs.popleft()
            # n_thr = n_thri.copy()
            # if i > 0:
            #     for n_th in n_thrs:
            #         n_thr = n_thr * n_th
            # n_thrs.append(n_thri)
            # if len(c_thrs) > 100:
            #     c_thrs.popleft()
            # c_thr = c_thri.copy()
            # if i > 0:
            #     for c_th in c_thrs:
            #         c_thr = c_thr * c_th
            # c_thrs.append(c_thri)
            croi = c_thr*img
            nroi = n_thr*img
            cnz = croi[np.nonzero(croi)]
            nnz = nroi[np.nonzero(nroi)]
            c_ave_int = np.median(cnz)
            n_ave_int = np.median(nnz)
            yc.append(c_ave_int)
            yn.append(n_ave_int)
            if i == 0:
                cmin = np.min(img)
                cmax = 2*np.max(img)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img, cmap='turbo')
            axim.set_clim(cmin, cmax)
            t_text = 't = ' + fname + 's'
            ax.annotate(t_text, (16, 32), color='white', fontsize=20)
            fontprops = font_manager.FontProperties(size=20)
            asb = AnchoredSizeBar(ax.transData, 100, u'16\u03bcm',
                color='white', size_vertical=2, fontproperties=fontprops,
                loc='lower left', pad=0.1, borderpad=0.5, sep=5, frameon=False)
            ax.add_artist(asb)
            if u_csv and round(float(fname), 1) in t_annotate_bl:
                w, h = img.shape
                ax.add_patch(Rectangle((3, 3), w-7, h-7,
                    linewidth=5, edgecolor='#00B0FF', facecolor='none'))
            ax.grid(False)
            ax.axis('off')
            cb = fig.colorbar(axim, pad=0.01, format='%.3f',
                extend='both', extendrect=True, extendfrac=0.03)
            cb.outline.set_linewidth(0)
            fig.tight_layout(pad=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(c_thr, linewidths=0.8, colors='w')
                ax.contour(n_thr, linewidths=0.2, alpha=0.5, colors='k')
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', pad_inches=0)
            imgs.append(img_as_ubyte(np.array(fig.canvas.renderer._renderer)))
            plt.close(fig)
            bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
            row_i = np.random.choice(bg_rows.shape[0])
            bg_row = bg_rows[row_i]
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(raw[bg_row, :])
            ax.plot(bkg[bg_row, :])
            ax.set_title(str(bg_row))
            bg_path = os.path.join(results_dir, 'debug', '{}.png'.format(fname))
            fig.savefig(bg_path, dpi=100)
            plt.close(fig)
        if u_csv is None:
            fig, ax = plt.subplots(figsize=(10,8))
        else:
            fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
            ax0.plot(tu, u)
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        ax.plot(t, yc, color='#d32f2f', label='Cytoplasm')
        ax.plot(t, yn, color='#388E3C', label='Nucleus')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    data = np.column_stack((t, yc, yn))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,yc,yn', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
    return t, yc, yn


def compare_plots(root_dir, labels, fold_change=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        combined_y = pd.DataFrame()
        # combined_uta = pd.DataFrame()
        # combined_utb = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv_fp = os.path.join(root_dir, data_dir, 'y.csv')
            y_data = pd.read_csv(y_csv_fp)
            t = y_data['t'].values
            y = y_data['y'].values
            yf = interp1d(t, y, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            y = pd.Series([yf(ti) for ti in t], index=t, name=i)
            if fold_change:
                y = y/y[0]
            combined_y = pd.concat([combined_y, y], axis=1)
            # u_csv_fp = os.path.join(root_dir, data_dir, 'u.csv')
            # u_data = pd.read_csv(u_csv_fp)
            # uta = np.around(u_data['ta'].values, 1)
            # utb = np.around(u_data['tb'].values, 1)
            # uta = pd.Series(uta, name=i)
            # utb = pd.Series(utb, name=i)
            # combined_uta = pd.concat([combined_uta, uta], axis=1)
            # combined_utb = pd.concat([combined_utb, utb], axis=1)
            ax.plot(y, alpha=0.7)
        # uta_ave = combined_uta.mean(axis=1).rename('uta_ave')
        # utb_ave = combined_utb.mean(axis=1).rename('utb_ave')
        # tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
        # u = np.zeros_like(tu)
        # for ta, tb in zip(uta, utb):
        #     ia = list(tu).index(ta)
        #     ib = list(tu).index(tb)
        #     u[ia:ib] = 1
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        ax.legend(labels=labels, loc='best')
        if fold_change:
            ax.set_ylabel('Fold Change')
        # ax0.plot(tu, u, color='#1976D2')
        # ax0.set_yticks([0, 1])
        # ax0.set_ylabel('BL')
        fig.savefig(os.path.join(root_dir, 'y_compare.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def linescan(img_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    setup_dirs(os.path.join(results_dir, 'line'))
    setup_dirs(os.path.join(results_dir, 'debug'))
    t = []
    y = []
    imgs = []
    lines = []
    areas = []
    for imgf in list_img_files(img_dir):
        t.append(np.float(os.path.splitext(os.path.basename(imgf))[0]))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
            fname = os.path.splitext(os.path.basename(imgf))[0]
            img, raw, bkg = preprocess_img(imgf)
            if i == 0:
                img = rotate(img, angle=-10)
            if i == 1:
                img = rotate(img, angle=-5)
            if i == 2:
                img = rotate(img, angle=-18)
            if i == 3:
                img = rotate(img, angle=-35)
            if i == 4:
                img = rotate(img, angle=-50)
            if i == 5:
                img = rotate(img, angle=-55)
            thr = segment_object(img, offset=0)
            roi = thr*img
            nz = roi[np.nonzero(roi)]
            ave_int = np.median(nz)
            y.append(ave_int)
            labeled = label(thr)
            rprops = regionprops(labeled)
            line_row = int(np.round(rprops[0].centroid[0], 0))
            line_col = int(np.round(rprops[0].centroid[1], 0))
            area = rprops[0].area
            areas.append(area)
            if i == 0:
                cmin = np.min(img)
                cmax = 1*np.max(img)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img, cmap='turbo')
            ax.plot([line_col-40, line_col+40], [line_row, line_row], color='w')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(thr, linewidths=0.3, colors='w')
            axim.set_clim(cmin, cmax)
            ax.grid(False)
            ax.axis('off')
            cb = fig.colorbar(axim, pad=0.01, format='%.3f',
                extend='both', extendrect=True, extendfrac=0.03)
            cb.outline.set_linewidth(0)
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', pad_inches=0)
            imgs.append(img_as_ubyte(np.array(fig.canvas.renderer._renderer)))
            plt.close(fig)
            bg_row = np.argmax(np.var(img, axis=1))
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(raw[bg_row, :])
            ax.plot(bkg[bg_row, :])
            ax.set_title(str(bg_row))
            fig.savefig(os.path.join(results_dir, 'debug', '{}.png'.format(fname)),
                dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12,8))
            ax.plot(img[line_row, (line_col-40):(line_col+40)])
            ax.set_ylabel('Pixel Intensity')
            ax.set_xlabel('X Position')
            ax.set_ylim(0, 1)
            fig.savefig(os.path.join(results_dir, 'line', '{}.png'.format(fname)),
                dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
            lines.append(np.array(fig.canvas.renderer._renderer))
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(t, y, color='#d32f2f')
        ax.set_xlabel('Time (Frame)')
        ax.set_ylabel('AU')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'median_intensity.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(t, areas, color='#d32f2f')
        ax.set_xlabel('Time (Frame)')
        ax.set_ylabel('Area (sq. pixels)')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'area.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    data = np.column_stack((t, y, areas))
    np.savetxt(os.path.join(results_dir, 'data.csv'),
        data, delimiter=',', header='t,med_int,area', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(results_dir, 'imgs.gif'), imgs, fps=2)
        mimwrite(os.path.join(results_dir, 'line.gif'), lines, fps=2)
    return t, y


def process_pqr(fp0_img_dir, fp1_img_dir, results_dir):
    """Analyze PQR fluorescent intensities and generate figures."""
    setup_dirs(os.path.join(results_dir, 'fp0_imgs'))
    setup_dirs(os.path.join(results_dir, 'fp1_imgs'))
    y0 = []
    y1 = []
    imgfs0 = list_img_files(fp0_img_dir)
    imgfs1 = list_img_files(fp1_img_dir)
    for i, (imgf0, imgf1) in enumerate(tqdm(zip(imgfs0, imgfs1), total=len(imgfs0))):
        fname0 = os.path.splitext(os.path.basename(imgf0))[0]
        fname1 = os.path.splitext(os.path.basename(imgf1))[0]
        img0, tval0 = preprocess_img(imgf0)
        img1, tval1 = preprocess_img(imgf1)
        thr = segment_object(img0, offset=0, er=7)
        roi0 = thr*img0
        roi1 = thr*img1
        nz0 = roi0[np.nonzero(roi0)]
        nz1 = roi1[np.nonzero(roi1)]
        ave_int0 = np.median(nz0)
        ave_int1 = np.median(nz1)
        y0.append(ave_int0)
        y1.append(ave_int1)
        cmin = np.min([np.min(img0), np.min(img1)])
        cmax = np.max([np.max(img0), np.max(img1)])
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img0, cmap='turbo')
            # axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(thr, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'fp0_imgs', fname0 + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img1, cmap='turbo')
            # axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(thr, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='both', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'fp1_imgs', fname1 + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    y = np.array(y1)/np.array(y0)
    data = [y0, y1, list(y)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.violinplot(data=data, ax=ax, palette=['#1976D2', '#D32F2F', '#7B1FA2'], scale='count')
        ax.set_xticklabels(['FP0', 'FP1', 'FP1/FP0'])
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=200, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    data = np.column_stack((y0, y1, y))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='fp0,fp1,fp0/fp1', comments='')
    return y


def process_FRET(donor_dir, fret_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs', 'donor'))
    setup_dirs(os.path.join(results_dir, 'imgs', 'fret'))
    y0 = []
    y1 = []
    yn = []
    donor_imgfs = list_img_files(donor_dir)
    fret_imgfs = list_img_files(fret_dir)
    for i, (don_imgf, fre_imgf) in enumerate(tqdm(zip(donor_imgfs, fret_imgfs), total=len(donor_imgfs))):
        don_thr, don_img = segment_object(don_imgf)
        fre_thr, fre_img = segment_object(fre_imgf)
        don_roi = don_img * don_thr
        fre_roi = fre_img * fre_thr
        don_pix = don_roi[np.nonzero(don_roi)]
        fre_pix = fre_roi[np.nonzero(fre_roi)]
        don_ave_int = np.mean(don_pix)
        fre_ave_int = np.mean(fre_pix)
        y0.append(don_ave_int)
        y1.append(fre_ave_int)
        yn.append(fre_ave_int/don_ave_int)
        cmin = min(np.min(don_pix), np.min(fre_pix))
        cmax = 1.1*max(np.max(don_pix), np.max(fre_pix))
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(don_img, cmap='turbo')
            axim.set_clim(cmin, cmax)
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='min', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', 'donor', str(i) + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(fre_img, cmap='turbo')
            axim.set_clim(cmin, cmax)
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f',
                extend='min', extendrect=True, extendfrac=0.025)
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', 'fret', str(i) + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((y0, y1, yn))
    np.savetxt(os.path.join(results_dir, 'data.csv'),
        data, delimiter=',', header='d,f,f/d', comments='')


def compare_imgs(img_paths, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    setup_dirs(os.path.join(results_dir, 'debug'))
    fnames = []
    ave_ints = []
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        for imgf in img_paths:
            fname = os.path.splitext(os.path.basename(imgf))[0]
            fnames.append(fname)
            img, raw, bkg, den = preprocess_img(imgf)
            thr = segment_object(den, method=None, rs=100, fh=None, offset=0, er=None, cb=None)
            roi = thr*img
            nz = roi[np.nonzero(roi)]
            ave_int = np.median(nz)
            ave_ints.append(ave_int)
            cmin = np.min(img)
            cmax = np.max(img)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='turbo')
            axim.set_clim(cmin, cmax)
            fontprops = font_manager.FontProperties(size=20)
            asb = AnchoredSizeBar(ax.transData, 100, u'16\u03bcm',
                color='white', size_vertical=2, fontproperties=fontprops,
                loc='lower left', pad=0.1, borderpad=0.5, sep=5, frameon=False)
            ax.add_artist(asb)
            ax.grid(False)
            ax.axis('off')
            cb = fig.colorbar(axim, pad=0.01, format='%.3f',
                extend='both', extendrect=True, extendfrac=0.03)
            cb.outline.set_linewidth(0)
            fig.tight_layout(pad=0)
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     ax.contour(thr, linewidths=0.3, colors='w')
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            bg_rows = np.argsort(np.var(den, axis=1))[-100:-1:10]
            row_i = np.random.choice(bg_rows.shape[0])
            bg_row = bg_rows[row_i]
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(raw[bg_row, :])
            ax.plot(bkg[bg_row, :])
            ax.set_title(str(bg_row))
            bg_path = os.path.join(results_dir, 'debug', '{}.png'.format(fname))
            fig.savefig(bg_path, dpi=100)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(12,8))
        for i, (ave_int, fname) in enumerate(zip(ave_ints, fnames)):
            ax.plot([i], [ave_int], marker='o')
        plt.xticks(list(range(len(fnames))), fnames)
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=200, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_iexpress(img_dir, results_dir, u_csv=None, seg_params={'rs': 5000, 'fh': 400, 'cb': None, 'factor': 1}):
    """Analyze gene expression dataset and generate figures."""
    setup_dirs(os.path.join(results_dir, 'imgs'))
    setup_dirs(os.path.join(results_dir, 'debug'))
    t = []
    y = []
    imgs = []
    t_bl_on = []
    for imgf in list_img_files(img_dir):
        t.append(np.float(os.path.splitext(os.path.basename(imgf))[0]))
    if u_csv is not None:
        shutil.copyfile(u_csv, os.path.join(results_dir, 'u.csv'))
        udf = pd.read_csv(u_csv)
        t = np.around(t, 1)
        tu = np.around(np.arange(t[0], t[-1], 0.1), 1)
        uta = np.around(udf['ta'].values, 1)
        utb = np.around(udf['tb'].values, 1)
        u = np.zeros_like(tu)
        for ta, tb in zip(uta, utb):
            t_bl_on += list(np.arange(round(ta, 1), round(tb, 1) + 0.01, 0.1))
            ia = list(tu).index(ta)
            ib = list(tu).index(tb)
            u[ia:ib+1] = 1
    t_annotate_bl = []
    for tbl in t_bl_on:
        t_annotate_bl.append(min(t, key=lambda ti : abs(ti - tbl)))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
            fname = os.path.splitext(os.path.basename(imgf))[0]
            img, raw, bkg, den = preprocess_img(imgf)
            thr = segment_object(den, **seg_params)
            roi = thr*img
            nz = roi[np.nonzero(roi)]
            ave_int = np.median(nz)
            y.append(ave_int)
            if i == 0:
                cmin = np.min(img)
                cmax = 2*np.max(img)
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='turbo')
            axim.set_clim(cmin, cmax)
            t_text = 't = ' + fname + 's'
            ax.annotate(t_text, (16, 32), color='white', fontsize=20)
            fontprops = font_manager.FontProperties(size=20)
            asb = AnchoredSizeBar(ax.transData, 100, u'16\u03bcm',
                color='white', size_vertical=2, fontproperties=fontprops,
                loc='lower left', pad=0.1, borderpad=0.5, sep=5, frameon=False)
            ax.add_artist(asb)
            if u_csv and round(float(fname), 1) in t_annotate_bl:
                w, h = img.shape
                ax.add_patch(Rectangle((3, 3), w-7, h-7,
                    linewidth=5, edgecolor='#00B0FF', facecolor='none'))
            ax.grid(False)
            ax.axis('off')
            cb = fig.colorbar(axim, pad=0.01, format='%.3f',
                extend='both', extendrect=True, extendfrac=0.03)
            cb.outline.set_linewidth(0)
            fig.tight_layout(pad=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(thr, linewidths=0.3, colors='w')
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', pad_inches=0)
            imgs.append(img_as_ubyte(np.array(fig.canvas.renderer._renderer)))
            plt.close(fig)
            bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
            row_i = np.random.choice(bg_rows.shape[0])
            bg_row = bg_rows[row_i]
            fig, ax = plt.subplots(figsize=(10,8))
            ax.plot(raw[bg_row, :])
            ax.plot(bkg[bg_row, :])
            ax.set_title(str(bg_row))
            bg_path = os.path.join(results_dir, 'debug', '{}.png'.format(fname))
            fig.savefig(bg_path, dpi=100)
            plt.close(fig)
        if u_csv is None:
            fig, ax = plt.subplots(figsize=(16,8))
        else:
            fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
            ax0.plot(tu, u)
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        ax.plot(t, y, color='#d32f2f')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Median Intensity')
        ytiks, ystep = np.linspace(np.min(y), np.max(y), 6, endpoint=True, retstep=True)
        ylim = (ytiks[0] - ystep/4, ytiks[-1] + ystep/4)
        ax.set_yticks(ytiks)
        ax.set_ylim(ylim)
        ax1 = ax.twinx()
        ax1.plot(t, y/y[0], color='#d32f2f')
        # ax1.grid(False)
        ax1.set_yticks(ytiks/y[0])
        ax1.set_ylim(ylim/y[0])
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_ylabel('Fold Change')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'y.png'),
            dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,y', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
    return t, y


def compare_AUCs(root_dir, t_lim=(60, 120)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(16,8))
        freqs = []
        ave_aucs = []
        color_cycle = cycle(custom_palette)
        for freq_i in natsorted([x[1] for x in os.walk(root_dir)][0]):
            freqs.append(int(freq_i))
            freq_i_dir = os.path.join(root_dir, freq_i)
            color = next(color_cycle)
            freq_i_aucs = []
            for rep_j in natsorted([x[1] for x in os.walk(freq_i_dir)][0]):
                y_csv = os.path.join(freq_i_dir, rep_j, 'y.csv')
                df = pd.read_csv(y_csv)
                span = df[(df['t'] >= t_lim[0]) & (df['t'] <= t_lim[1])]
                tspan = span['t'].values
                yspan = rescale(span['y'].values)
                plt.plot(tspan, yspan)
                plt.show()
                auc = simps(yspan, tspan)
                freq_i_aucs.append(auc)
                ax.plot([int(freq_i)], [auc], marker='o', color=color)
            ave_aucs.append(np.mean(freq_i_aucs))
    color = next(color_cycle)
    ax.plot(freqs, ave_aucs, color=color)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Pulse Period')
    ax.set_xticks(freqs)
    fig.savefig(os.path.join(root_dir, 'freqscan.png'),
        dpi=200, bbox_inches='tight', transparent=False)
    plt.close(fig)


if __name__ == '__main__':
    # img_path = '/home/phuong/data/LINTAD/LexA/0/0.05.tiff'
    # img_path = '/home/phuong/data/LINTAD/CAD/0/66.65.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/0.05.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/373.63.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/391.18.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/nucleus/0/0.05.tiff'
    # img_path = '/home/phuong/data/ILID/ddFPRA-iLID16I/20200709-RA-iLID16I-BLmini-1/mCherry/0/282.0.tiff'
    # img_path = '/home/phuong/data/chujun/0/00.png'
    # img_path = '/home/phuong/data/ILID/ddFPRA-iLIDLS/freq-scan/20200726-RA-LS-pulse-2s/mCherry/1/36.0.tiff'
    # test_preprocess_img(img_path)
    # test_segment_object(img_path)


    # tr_img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/0.05.tiff'
    # nu_img_path = '/home/phuong/data/LINTAD/LINuS/nucleus/0/0.05.tiff'
    # test_segment_translo(tr_img_path, nu_img_path)

    # nu_root_dir = '/home/phuong/data/LINTAD/LINuS/nucleus'
    # tr_root_dir = '/home/phuong/data/LINTAD/LINuS/bilinus'
    # results_dir = '/home/phuong/data/LINTAD/LINuS-results'
    # for img_dir in natsorted([x[1] for x in os.walk(nu_root_dir)][0]):
    #     nucleus_dir = os.path.join(nu_root_dir, img_dir)
    #     translo_dir = os.path.join(tr_root_dir, img_dir)
    #     res_dir = os.path.join(results_dir, img_dir)
    #     process_translo(nucleus_dir, translo_dir, res_dir)
    # # results_dir = '/home/phuong/data/LINTAD/LINuS-mock/'
    # combine_translo(results_dir)


    # cad_img_path = '/home/phuong/data/LINTAD/CAD/0/66.65.tiff'
    # test_segment_clusters(cad_img_path)

    # root_dir = '/home/phuong/data/LINTAD/CAD/'
    # results_dir = '/home/phuong/data/LINTAD/CAD-results'
    # for img_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     img_dir = os.path.join(root_dir, img_dirname)
    #     res_dir = os.path.join(results_dir, img_dirname)
    #     process_cad(img_dir, res_dir)
    # combine_cad(results_dir)


    # root_dir = '/home/phuong/data/LINTAD/LexA/'
    # results_dir = '/home/phuong/data/LINTAD/LexA-results'
    # for img_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     img_dir = os.path.join(root_dir, img_dirname)
    #     res_dir = os.path.join(results_dir, img_dirname)
    #     process_iexpress(img_dir, res_dir)


    # don_dir = '/home/phuong/data/reed/dishB_YPet/yfp/'
    # fre_dir = '/home/phuong/data/reed/dishB_YPet/fret/'
    # res_dir = '/home/phuong/data/reed/dishB_YPet-results'
    # process_FRET(donor_dir=don_dir, fret_dir=fre_dir, results_dir=res_dir)


    # fp0_img_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/3NLS-mTq2/'
    # fp1_img_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/NES-mCh/'
    # results_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/'
    # process_pqr(fp0_img_dir, fp1_img_dir, results_dir)

    i = 0
    root_dir = '/home/phuong/data/20200731-pcDNA-LINTAD-4LexO_YB_mScI/'
    results_dir = os.path.join(root_dir, 'results', str(i))
    img_dir = os.path.join(root_dir, 'mCherry', str(i))
    u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    process_iexpress(img_dir, results_dir, u_csv,
        {'rs': None, 'fh': None, 'cb': None, 'er': None, 'factor': 0.1})


    # for j in [2, 5, 8, 13]:
    #     for i in range(3):
    #         root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/freq-scan/20200726-RA-LS-pulse-{}s/'.format(j)
    #         results_dir = os.path.join(root_dir, 'results', str(i))
    #         print(results_dir)
    #         img_dir = os.path.join(root_dir, 'mCherry', str(i))
    #         u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    #         process_iexpress(img_dir, results_dir, u_csv)

    # root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/20200721-LS-pulsatile/results/'
    # combine_uy(root_dir, fold_change=False)

    # root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/20200718-RA-LS-noBL-mch32/compare/'
    # compare_plots(root_dir, ['Imaging Only', 'With BL Induction'], fold_change=False)

    # img_paths = [
    #     '/home/phuong/data/Controls/L3K2/1.5.tif',
    #     '/home/phuong/data/Controls/L3K2/2.tif',
    #     '/home/phuong/data/Controls/L3K2/3.tif'
    # ]

    # results_dir = '/home/phuong/data/Controls/L3K2/results'
    # compare_imgs(img_paths, results_dir)


    # root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/freq-scan/freq_trend/'
    # compare_AUCs(root_dir)