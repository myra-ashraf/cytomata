import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splev, splrep
import seaborn as sns
from matplotlib.patches import Rectangle
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

from cytomata.utils.io import setup_dirs, list_img_files
from cytomata.process.detect import (
    preprocess_img, measure_regions, segment_clusters, segment_mito,
    segment_whole_cell, segment_dim_nucleus, segment_bright_nucleus
)
from cytomata.utils.visual import imshow, plot, imgs_to_gif, custom_styles, custom_palette
from cytomata.utils.io import setup_dirs


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


def process_cad(img_dir, name='CRY2low', metric='area'):
    """Analyze cad dataset and generate figures."""
    save_dir = name + '-results'
    setup_dirs(os.path.join(save_dir, 'imgs'))
    y = []
    t = []
    plot_imgs = []
    for i, imgf in enumerate(list_img_files(img_dir)):
        dots, den = segment_clusters(imgf)
        dots_areas, dots_med_ints = measure_regions(dots, den)
        fname = os.path.splitext(os.path.basename(imgf))[0]
        if i == 0:
            cmin = np.min(den)
            cmax = 1.1*np.max(den)
        t.append(np.float(fname))
        if metric == 'area':
            y.append(np.sum(dots_areas)/39.0625)
            ylabel = r'Combined Area of Clusters (${\mu}m^2$)'
        elif metric == 'intensity':
            y.append(dots_med_ints)
            ylabel = 'Median Region Intensity'
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(den, cmap='viridis')
            axim.set_clim(cmin, cmax)
            ax.contour(dots, linewidths=0.1, colors='r')
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
    t_half = approx_half_life(t, y)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, name + '-kinetics.csv'),
        data, delimiter=',', header='t,y', comments='')
    imgs_to_gif(plot_imgs, os.path.join(save_dir, name + '-kinetics.gif'), fps=10)
    plot(t, y, xlabel='Time (s)', ylabel=ylabel,
        title='%s Kinetics | $t_{1/2}$=%.0fsec' % (name, t_half),
        save_path=os.path.join(save_dir, name + '-kinetics.png'))


def process_bilinus(img_dir):
    """Analyze BiLINuS dataset and generate figures."""
    pass


def process_lexa(img_dir):
    """Analyze LexA expression dataset and generate figures."""
    pass



if __name__ == '__main__':
    t0 = time.time()
    img_dir = os.path.join('data', 'pDN77')
    # process_cad(img_dir)
    imgfs = list_img_files(img_dir)
    imgf = imgfs[38]
    thr, den = segment_dim_nucleus(imgf)
    # # plt.hist(den.ravel(), bins=255)
    plt.imshow(den, cmap='viridis')
    plt.contour(thr, linewidths=0.2, colors='r')
    plt.show()
