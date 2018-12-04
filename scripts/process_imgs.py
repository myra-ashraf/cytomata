import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

from cytomata.utils.io import setup_dirs
from cytomata.process.extract import (
    get_median_intensity,
    get_regions,
    images_to_ave_single_cell_intensities,
    plot_single_cell_trajectories,
    plot_single_cell_trajectories_ave,
    save_single_cell_data
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
    data_dir = 'data'
    expt = 'nate'
    img_dir = os.path.join(data_dir, expt)
    save_dir = os.path.join(data_dir, expt + '_' + 'single_cell_results')
    setup_dirs(save_dir)
    trajs = images_to_ave_single_cell_intensities(img_dir, save_dir, regions_min_size=50)
    plot_single_cell_trajectories(trajs, save_dir)
    plot_single_cell_trajectories_ave(trajs, save_dir)
    # t_spans = [0, 62, 73, 98, 109, 134, 145, 170, 181, 206, 217, 244]
    # save_single_cell_data(img_dir, trajs, save_dir, calc_func=norm_auc, t_spans=t_spans)
    save_single_cell_data(img_dir, trajs, save_dir)
