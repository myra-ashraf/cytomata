from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
    int, map, next, oct, open, pow, range, round, str, super, zip)

import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import pims
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cytomata.utils.io import list_img_files
from cytomata.utils.visual import plot, imshow, imgs_to_mp4, imgs_to_gif, DynamicPlot


if __name__ == '__main__':
    # Styled Plots
    df = pd.read_csv('data_test.csv')
    tp = df['t'].values
    yp = df[['y', 'z']].values
    plot(tp, yp, xlabel='Time', ylabel='Intensity', title='LINTAD Dynamic Response',
        labels=['GFP', 'RFP'], show=True, save_path=None)

    # Dynamic Plots
    dplot = DynamicPlot(tp, yp, xlabel='Time', ylabel='Intensity',
        title='LINTAD Dynamic Response', labels=['GFP', 'RFP'], save_dir='.')
    t1 = time.time()
    for i in range(10):
        dplot.update(tp, yp)
    print(time.time() - t1)

    # Images to MP4
    imgs = pims.open(os.path.join('data/imgs/DIC/0', '*'))
    imgs_to_mp4(imgs, vid_path='result.mp4', fps=10)

    # Images to GIF
    imgs_to_gif(imgs, gif_path='result.gif', fps=10)

    # Show Image
    imshow(imgs[0], title=None, axes=False, colorbar=False, show=True, save_path=None)
