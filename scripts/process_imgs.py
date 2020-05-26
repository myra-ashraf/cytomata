import os
import sys
import warnings
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from scipy.stats import sem
from imageio import mimwrite
from scipy.interpolate import interp1d

from cytomata.process import preprocess_img, segment_object, segment_clusters
from cytomata.utils import setup_dirs, list_img_files, rescale, plot, custom_styles, custom_palette


def test_preprocess_img(img_path):
    img, tval = preprocess_img(img_path)
    plt.imshow(img, cmap='viridis')
    plt.show()


def test_segment_object(img_path):
    img, tval = preprocess_img(img_path)
    thr = segment_object(img)
    plt.imshow(img, cmap='viridis')
    plt.contour(thr, linewidths=0.3, colors='w')
    plt.show()


def test_segment_translo(bilinus_img_path, nucleus_img_path):
    nucl_img, nucl_tval = preprocess_img(nucleus_img_path)
    cell_img, cell_tval = preprocess_img(bilinus_img_path)
    nucl_thr = segment_object(nucl_img, cb=5, er=3)
    cell_thr = segment_object(cell_img, er=5)
    plt.imshow(cell_img, cmap='viridis')
    plt.contour(cell_thr, linewidths=0.3, colors='w')
    plt.contour(nucl_thr, linewidths=0.2, colors='r')
    plt.show()


def test_segment_clusters(cad_img_path):
    cell, img = segment_object(cad_img_path, offset=5, er=9)
    dots, _ = segment_clusters(cad_img_path)
    dots = np.logical_and(dots, cell)
    anti = np.logical_xor(dots, cell)
    plt.imshow(img, cmap='viridis')
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
        axim = ax.imshow(nu_img, cmap='viridis')
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
        axim = ax.imshow(tr_img, cmap='viridis')
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
            axim = ax.imshow(tr_img, cmap='viridis')
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
            axim = ax.imshow(img, cmap='viridis')
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


def process_iexpress(img_dir, results_dir):
    """Analyze gene expression dataset and generate figures."""
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    y = []
    imgs = []
    tval = None
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, tval = preprocess_img(imgf, tval)
        nz = img[np.nonzero(img)]
        if i == 0:
            cmin = np.min(img)
            cmax = 1.4*np.max(img)
        t.append(np.float(fname))
        y.append(np.mean(nz))
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img, cmap='viridis')
            axim.set_clim(cmin, cmax)
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
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,y', comments='')
    plot(t, y, xlabel='Time (s)', ylabel='Ave FL Intensity',
        save_path=os.path.join(results_dir, 'plot.png'))
    mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=len(imgs)//12)
    return t, y


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
            axim = ax.imshow(don_img, cmap='viridis')
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
            axim = ax.imshow(fre_img, cmap='viridis')
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


if __name__ == '__main__':
    # img_path = '/home/phuong/data/LINTAD/LexA/0/0.05.tiff'
    # img_path = '/home/phuong/data/LINTAD/CAD/0/66.65.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/0.05.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/373.63.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/391.18.tiff'
    # img_path = '/home/phuong/data/LINTAD/LINuS/nucleus/0/0.05.tiff'
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


    root_dir = '/home/phuong/data/LINTAD/LexA/'
    results_dir = '/home/phuong/data/LINTAD/LexA-results'
    for img_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
        img_dir = os.path.join(root_dir, img_dirname)
        res_dir = os.path.join(results_dir, img_dirname)
        process_iexpress(img_dir, res_dir)


    # don_dir = '/home/phuong/data/reed/dishB_YPet/yfp/'
    # fre_dir = '/home/phuong/data/reed/dishB_YPet/fret/'
    # res_dir = '/home/phuong/data/reed/dishB_YPet-results'
    # process_FRET(donor_dir=don_dir, fret_dir=fre_dir, results_dir=res_dir)
