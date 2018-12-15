import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

from cytomata.utils.io import setup_dirs
from cytomata.process.extract import (
    images_to_ave_frame_intensities,
    run_single_cell_analysis,
    )


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
    dataset = '20181207-000000'
    img_dir = os.path.join('data', dataset)
    ## Global Frame Analysis
    save_dir = os.path.join('data', dataset + '_' + 'global_frame_analysis')
    images_to_ave_frame_intensities(img_dir, save_dir, gauss_sigma=40, iter_cb=None)
    ## Single Cell Analysis
    save_dir = os.path.join('data', dataset + '_' + 'single_cell_analysis')
    # t_spans = [0, 62, 73, 98, 109, 134, 145, 170, 181, 206, 217, 244]
    # save_single_cell_data(trajs, img_dir, save_dir, min_traj_length=100,
    #     calc_func=norm_auc, t_spans=t_spans)
    run_single_cell_analysis(img_dir,
        save_dir, min_traj_length=100, overwrite=True, **reg_params)
