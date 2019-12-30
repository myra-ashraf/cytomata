import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from cytomata.process.detect_classic import (
    preprocess_img, segment_clusters, segment_whole_cell, segment_dim_nucleus, segment_bright_nucleus
)
from cytomata.utils.visual import imshow, plot, custom_styles, custom_palette


if __name__ == '__main__':
    t0 = time.time()
    img_dir = os.path.join('data', 'iLID/mCherry/0')
    imgfs = list_img_files(img_dir)
    imgf = imgfs[1]
    thr, den = segment_clusters(imgf)
    # plt.hist(den.ravel(), bins=255)
    plt.imshow(den, cmap='gray')
    plt.contour(thr, linewidths=0.05, colors='r')
    plt.show()
    # data = []
    # for i, imgf in enumerate(imgfs):
    #     # imgf = imgfs[99]
    #     img = img_as_float(imread(imgf))
    #     area, den, thr = segment_clusters(imgf)
    #     data.append(area)
    #     fig, ax = plt.subplots()
    #     axim = ax.imshow(den, cmap='viridis')
    #     ax.contour(thr, linewidths=0.05, colors='r')
    #     ax.grid(False)
    #     ax.axis('off')
    #     cb = fig.colorbar(axim, pad=0.01)
    #     cb.outline.set_linewidth(0)
    #     fig.canvas.draw()
    #     fig.savefig('results/' + str(i) + '.png', dpi=100)
    #     plt.close(fig)
    #     # print(time.time() - t0)
    # plot(data, xlabel='Frame', ylabel='Combined Area of Clusters', title='CRY2-CIB1 Kinetics', show=True, save_path=None)

# with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
#     fig, ax = plt.subplots()
#     axim = ax.imshow(den, cmap='Blues')
#     ax.contour(thr, linewidths=0.05, colors='r')
#     ax.grid(False)
#     ax.axis('off')
#     cb = fig.colorbar(axim, pad=0.01)
#     cb.outline.set_linewidth(0)
#     fig.canvas.draw()
#     fig.savefig('results/' + str(i) + '.png', dpi=100, bbox_inches='tight')
#     plt.close(fig)
