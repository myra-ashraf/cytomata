import os
import sys
import time
import warnings
from itertools import cycle
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
from natsort import natsorted
from scipy.stats import sem
from imageio import mimwrite
from scipy.interpolate import interp1d
from scipy.integrate import simps
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.transform import rotate

from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, segment_object, segment_clusters, process_u_csv
from cytomata.utils import setup_dirs, list_img_files, rescale, custom_styles, custom_palette


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
    # plot(t, y, xlabel='Time (s)', ylabel='AU',
    #     labels=['Cytoplasm', 'Nucleus'], save_path=os.path.join(results_dir, 'plot.png'))
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


def process_fluo_timelapse(img_dir, save_dir, u_csv=None, cmax_mult=2,
    seg_params={'rs': 5000, 'fh': 400, 'cb': None, 'factor': 1}):
    """Analyze fluorescence timelapse images and generate figures."""
    t = [np.float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in list_img_files(img_dir)]
    y = []
    tu = []
    u = []
    imgs = []
    t_ann_img = []
    if u_csv:
        tu, u, t_ann_img = process_u_csv(t, u_csv, save_dir)
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        plot_bkg_profile(fname, raw, bkg, save_dir)
        thr = segment_object(den, **seg_params)
        roi = thr*img
        nz = roi[np.nonzero(roi)]
        ave_int = np.median(nz)
        y.append(ave_int)
        if i == 0:
            cmin = np.min(img)
            cmax = cmax_mult*np.max(img)
        sig_ann = round(float(fname), 1) in t_ann_img
        cell_img = plot_cell_img(fname, den, thr, cmin, cmax, save_dir, sig_ann)
        imgs.append(cell_img)
    plot_uy(t, y, tu, u, save_dir)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, 'y.csv'),
        data, delimiter=',', header='t,y', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(save_dir, 'cell.gif'), imgs, fps=len(imgs)//12)


def combine_uy(root_dir, fold_change=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, (ax0, ax) = plt.subplots(
            2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]}
        )
        combined_y = pd.DataFrame()
        combined_u = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv = os.path.join(root_dir, data_dir, 'y.csv')
            y_data = pd.read_csv(y_csv)
            t = y_data['t'].values
            y = y_data['y'].values
            yf = interp1d(t, y, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            y = pd.Series([yf(ti) for ti in t], index=t, name=i)
            if fold_change:
                y = y/y[0]
            combined_y = pd.concat([combined_y, y], axis=1)
            ax.plot(y, color='#1976D2', alpha=0.5, linewidth=2)
            u_csv = os.path.join(root_dir, data_dir, 'u.csv')
            u_data = pd.read_csv(u_csv)
            tu = u_data['t'].values
            u = pd.Series(u_data['u'].values, index=tu, name=i)
            combined_u = pd.concat([combined_u, u], axis=1)
        y_ave = combined_y.mean(axis=1).rename('y_ave')
        y_std = combined_y.std(axis=1).rename('y_std')
        y_sem = combined_y.sem(axis=1).rename('y_sem')
        data = pd.concat([y_ave, y_std, y_sem], axis=1).dropna()
        data.to_csv(os.path.join(root_dir, 'y_combined.csv'))
        y_ave = data['y_ave']
        y_ci = data['y_sem']*1.96
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
        plot_name = 'y_combined.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_ratio_fluo(fp0_img_dir, fp1_img_dir, results_dir):
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


def compare_fluo_imgs(img_paths, results_dir):
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


    # fp0_img_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/3NLS-mTq2/'
    # fp1_img_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/NES-mCh/'
    # results_dir = '/home/phuong/data/ILID/Controls/NES-mCh-PQR-3NLS-mTq2/'
    # process_pqr(fp0_img_dir, fp1_img_dir, results_dir)

    # i = 8
    # root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/20200721-LS-single/'
    # results_dir = os.path.join(root_dir, 'results', str(i))
    # img_dir = os.path.join(root_dir, 'mCherry', str(i))
    # u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    # process_fluo_timelapse(img_dir, results_dir, u_csv, cmax_mult=2,
    #     seg_params={'rs': 5000, 'fh': 400, 'cb': None, 'er': None, 'factor': 2})


    # for i in range(9):
    #     root_dir = '/home/phuong/data/ILID/ddFPRA-iLIDLS/20200721-LS-single/'
    #     results_dir = os.path.join(root_dir, 'results', str(i))
    #     img_dir = os.path.join(root_dir, 'mCherry', str(i))
    #     u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    #     process_fluo_timelapse(img_dir, results_dir, u_csv, cmax_mult=2,
    #         seg_params={'rs': 5000, 'fh': 400, 'cb': None, 'er': None, 'factor': 10})

    root_dir = '/home/phuong/data/ILID/RA-LS/20200721-LS-single/results/'
    combine_uy(root_dir, fold_change=True)

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