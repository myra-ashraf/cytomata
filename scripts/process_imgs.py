import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from skimage.filters import gaussian, threshold_local
from skimage.restoration import estimate_sigma, denoise_nl_means
from skimage.morphology import disk, opening, white_tophat

from cytomata.utils.io import setup_dirs, list_img_files
from cytomata.process.extract import (
    get_ave_intensity,
    images_to_ave_frame_intensities,
    run_single_cell_analysis,
    )
from cytomata.utils.visual import imshow, plot


def norm_auc(frames, ints, t_spans):
    df = pd.DataFrame({'frame': frames, 'int': ints})
    calcs = []
    for i in range(len(t_spans) - 1):
        t0 = t_spans[i]
        t1 = t_spans[i + 1]
        interval = df.loc[(df['frame'] >= t0) & (df['frame'] < t1), 'int']
        calcs.append(interval.sum()/(t1 - t0))
    return calcs


if __name__ == '__main__':
    dataset = 'nate'
    img_dir = os.path.join('data', dataset)
    ## Global Frame Analysis
    save_dir = os.path.join('data', dataset + '_' + 'global_frame_analysis')
    # images_to_ave_frame_intensities(img_dir, save_dir, block=151, offset=0)
    imgfs = list_img_files(img_dir)
    img = img_as_float(imread(imgfs[25]))
    bkg = threshold_local(img, block_size=201, method='gaussian')
    sub = img - bkg
    sigma = estimate_sigma(sub)
    den = denoise_nl_means(sub, h=sigma, sigma=sigma, multichannel=False) - sigma*(0.0)
    den[den < 0.0] = 0.0
    print(len(img))
    # imshow(img, show=True)
    # plt.plot(img[95, :])
    # plt.plot(bkg[95, :])
    # plt.plot(sub[95, :])
    # plt.plot(den[95, :])
    # plt.show()

    # images_to_ave_frame_intensities(img_dir, save_dir, gauss_sigma=40, iter_cb=None)
    ## Single Cell Analysis
    # save_dir = os.path.join('data', dataset + '_' + 'single_cell_analysis')
    # t_spans = [0, 62, 73, 98, 109, 134, 145, 170, 181, 206, 217, 244]
    # save_single_cell_data(trajs, img_dir, save_dir, min_traj_length=100,
    #     calc_func=norm_auc, t_spans=t_spans)
    # run_single_cell_analysis(img_dir,
    #     save_dir, min_traj_length=100, overwrite=True, **reg_params)
