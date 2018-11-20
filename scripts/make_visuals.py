import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cytomata.utils.visual import plot, convert_to_png, load_images, frames_to_video

if __name__ == '__main__':
    # df = pd.read_csv('data_test.csv')
    # tp = df['t'].values
    # yp = df[['y', 'z']].values
    # # plot(tp, yp, labels=['GFP', 'RFP'], xlabel='Time', ylabel='Intensity', title='LINTAD Dynamic Response', save_path='./test.png')
    # #
    # dplot = DynamicPlot(tp, yp, labels=['GFP', 'RFP'], xlabel='Time', ylabel='Intensity', title='LINTAD Dynamic Response', save_dir='.')
    # t1 = time.time()
    # for i in range(10):
    #     dplot.update(tp, yp)
    # print(time.time() - t1)

    # img_dir = 'data/imgs/DIC/0'
    # png_dir = './png'
    # convert_to_png(img_dir, png_dir)
    # video_name = 'video2.mp4'
    # fps = 10.0
    # frames_to_video(png_dir, video_name, fps)
