import os
import sys
import time
from collections import deque
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import peak_local_max, blob_log
from skimage.filters import threshold_local, gaussian, rank, laplace
from skimage.morphology import disk, dilation, erosion, remove_small_objects, opening, remove_small_holes
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries
from skimage.color import label2rgb
from skimage.util import invert
from natsort import natsorted

from cytomata.utils.io import setup_dirs, list_img_files
from cytomata.process.detect import (
    preprocess_img, measure_regions, segment_clusters,
    segment_cell, segment_nucleus
)
from cytomata.utils.visual import imshow, plot, imgs_to_gif, custom_styles, custom_palette

def approx_half_life(t, y, phase='fall'):
    """Approximate half life of reaction process using cubic spline interpolation."""
    t = np.array(t)
    y = np.array(y)
    if phase == 'rise':
        tp = t[:y.argmax()]
        yp = y[:y.argmax()]
    elif phase == 'fall':
        tp = t[y.argmax():]
        yp = y[y.argmax():]
    y_half = (np.max(y) - np.min(y))/2
    yf = interp1d(tp, yp, 'cubic')
    ti = np.arange(tp[0], tp[-1], 1)
    yi = yf(ti)
    idx = np.argmin((yi - y_half)**2)
    t_half = ti[idx]
    return t_half


def process_cad(img_dir, save_dir, name='CIBN-CRY2PHR'):
    """Analyze cad dataset and generate figures."""
    setup_dirs(os.path.join(save_dir, 'imgs'))
    t = []
    y = []
    plot_imgs = []
    for i, imgf in enumerate(list_img_files(img_dir)):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        cell, den = segment_cell(imgf)
        dots, den = segment_clusters(imgf, mask=cell)
        if i == 0:
            cmin = np.min(den)
            cmax = 1.1*np.max(den)
        dots_area, _ = measure_regions(dots, den)
        dots_area = np.sum(dots_area/39.0625)
        cell_area, _ = measure_regions(cell, den)
        cell_area = np.max(cell_area/39.0625)
        t.append(np.float(fname))
        y.append(dots_area/cell_area)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='viridis')
            axim.set_clim(cmin, cmax)
            ax.contour(dots, linewidths=0.1, colors='r')
            ax.contour(cell, linewidths=0.1, colors='w')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            plot_imgs.append(np.array(fig.canvas.renderer._renderer))
            fig.savefig(os.path.join(save_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, 'data.csv'),
        data, delimiter=',', header='t,y', comments='')
    imgs_to_gif(plot_imgs, os.path.join(save_dir, 'cell.gif'), fps=10)
    t_half = approx_half_life(t, y)
    y_label = '(Cluster Area)/(Cell Area)'
    plot(t, y, xlabel='Time (s)', ylabel=y_label,
        title='%s Kinetics | $t_{1/2}$=%.0fsec' % (name, t_half),
        save_path=os.path.join(save_dir, 'plot.png'))


def process_bilinus(nucleus_dir, bilinus_dir, save_dir, name='BiLINuS'):
    setup_dirs(os.path.join(save_dir, 'imgs'))
    t = []
    y = []
    plot_imgs = []
    nucleus_imgfs = list_img_files(nucleus_dir)
    bilinus_imgfs = list_img_files(bilinus_dir)
    for i, (nu_imgf, bi_imgf) in enumerate(zip(nucleus_imgfs, bilinus_imgfs)):
        fname = os.path.splitext(os.path.basename(bi_imgf))[0]
        nucl, den_nucl = segment_nucleus(nu_imgf)
        cell, den = segment_cell(bi_imgf, mask=nucl)
        cell = cell | nucl
        cyto = cell ^ nucl
        if i == 0:
            cmin = np.min(den)
            cmax = 1.1*np.max(den)
        nucl_ave_int = np.mean(nucl*den)
        cyto_ave_int = np.mean(cyto*den)
        t.append(np.float(fname))
        y.append(nucl_ave_int/cyto_ave_int)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='viridis')
            axim.set_clim(cmin, cmax)
            ax.contour(nucl, linewidths=0.1, colors='r')
            ax.contour(cyto, linewidths=0.1, colors='w')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            plot_imgs.append(np.array(fig.canvas.renderer._renderer))
            fig.savefig(os.path.join(save_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, 'data.csv'),
        data, delimiter=',', header='t,y', comments='')
    imgs_to_gif(plot_imgs, os.path.join(save_dir, 'cell.gif'), fps=10)
    t_half = approx_half_life(t, y)
    y_label = '(Ave Nucleus Intensity)/(Ave Cytoplasm Intensity)'
    plot(t, y, xlabel='Time (s)', ylabel=y_label,
        title='%s Kinetics | $t_{1/2}$=%.0fsec' % (name, t_half),
        save_path=os.path.join(save_dir, 'plot.png'))


def aggregate_stats(results_dir, save_dir, xlabel, ylabel, title):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots()
        labels = []
        combined_y = pd.DataFrame()
        combined_t = pd.DataFrame()
        for res_dir in natsorted([x[1] for x in os.walk(results_dir)][0]):
            labels.append(res_dir)
            csv_fp = os.path.join(results_dir, res_dir, 'data.csv')
            data = pd.read_csv(csv_fp)
            combined_t[res_dir] = data['t']
            combined_y[res_dir] = data['y']
            ax.plot(data['t'], data['y'], color='#89bedc')
        combined_t_ave = combined_t.mean(axis=1)
        combined_y_ave = combined_y.mean(axis=1)
        labels.append('Ave')
        ave_curve = ax.plot(combined_t_ave, combined_y_ave, color='#0b559f')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left')
        ax.legend(handles=[Line2D([0], [0], color='#0b559f', lw=4)], labels=['Ave'], loc='best')
        fig.savefig(os.path.join(save_dir, 'kinetics.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def process_cad_multi():
    root_dir = '/home/phuong/data/LINTAD/CAD'
    results_dir = '/home/phuong/data/LINTAD/CAD-results'
    for img_dir in [x[1] for x in os.walk(root_dir)][0]:
        in_dir = os.path.join(root_dir, img_dir)
        out_dir = os.path.join(results_dir, img_dir)
        process_cad(in_dir, out_dir)
    xlabel = 'Time (s)'
    ylabel = '(Ave Cluster Area)/(Ave Cell Area)'
    title = 'CIBN-CRY2PHR Kinetics'
    aggregate_stats(results_dir, results_dir, xlabel, ylabel, title)

def process_bilinus_multi():
    nucleus_root_dir = '/home/phuong/data/LINTAD/LINuS/nucleus'
    bilinus_root_dir = '/home/phuong/data/LINTAD/LINuS/bilinus'
    results_dir = '/home/phuong/data/LINTAD/LINuS-results'
    for img_dir in [x[1] for x in os.walk(nucleus_root_dir)][0]:
        nucleus_dir = os.path.join(nucleus_root_dir, img_dir)
        bilinus_dir = os.path.join(bilinus_root_dir, img_dir)
        out_dir = os.path.join(save_dir, img_dir)
        process_bilinus(nucleus_dir, bilinus_dir, out_dir)
    xlabel = 'Time (s)'
    ylabel = '(Ave Nucleus Intensity)/(Ave Cytoplasm Intensity)'
    title = 'BiLINuS2 Kinetics'
    aggregate_stats(results_dir, results_dir, xlabel, ylabel, title)


if __name__ == '__main__':
    # nucleus_dir = '/home/phuong/data/LINTAD/LINuS/nucleus/0'
    # bilinus_dir = '/home/phuong/data/LINTAD/LINuS/bilinus/0'
    # save_dir = '/home/phuong/data/LINTAD/linus_results'
    # process_bilinus(nucleus_dir, bilinus_dir, save_dir)

    # nucleus_root_dir = '/home/phuong/data/LINTAD/LINuS/nucleus'
    # bilinus_root_dir = '/home/phuong/data/LINTAD/LINuS/bilinus'
    # save_dir = '/home/phuong/data/LINTAD/LINuS-results'
    # for img_dir in [x[1] for x in os.walk(nucleus_root_dir)][0]:
    #     nucleus_dir = os.path.join(nucleus_root_dir, img_dir)
    #     bilinus_dir = os.path.join(bilinus_root_dir, img_dir)
    #     out_dir = os.path.join(save_dir, img_dir)
    #     process_bilinus(nucleus_dir, bilinus_dir, out_dir)

    # root_dir = '/home/phuong/data/LINTAD/CAD'
    # save_dir = '/home/phuong/data/LINTAD/CAD-results'
    # for img_dir in [x[1] for x in os.walk(root_dir)][0]:
    #     in_dir = os.path.join(root_dir, img_dir)
    #     out_dir = os.path.join(save_dir, img_dir)
    #     process_cad(in_dir, out_dir)

    nucleus_root_dir = '/home/phuong/data/LINTAD/LINuS/nucleus'
    bilinus_root_dir = '/home/phuong/data/LINTAD/LINuS/bilinus'
    results_dir = '/home/phuong/data/LINTAD/LINuS-results'
    xlabel = 'Time (s)'
    ylabel = '(Ave Nucleus Intensity)/(Ave Cytoplasm Intensity)'
    title = 'BiLINuS2 Kinetics'
    aggregate_stats(results_dir, results_dir, xlabel, ylabel, title)

    # save_dir = '/home/phuong/data/LINTAD/CAD/20200131-CIBN-mTq2-CAD_CRY2PHR-mCh/3-results'
    # process_cad(img_dir, save_dir)
    # imgfs = list_img_files(img_dir)
    # imgf = imgfs[20]
    # cell, den = segment_whole_cell(imgf)
    # thr, den = segment_clusters(imgf, mask=cell)
    # # plt.hist(den.ravel(), bins=255)
    # plt.imshow(den, cmap='viridis')
    # plt.contour(cell, linewidths=0.2, colors='w')
    # plt.contour(thr, linewidths=0.2, colors='r')
    # plt.show()
