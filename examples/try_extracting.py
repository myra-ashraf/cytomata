import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

from cytomata.utils.visual import list_img_names
from cytomata.process.extract import extract_intensity, extract_regions

if __name__ == '__main__':
    img_dir = os.path.join('data', 'imgs', 'DIC', '0')
    img_names = list_img_names(img_dir)
    img0 = imread(os.path.join(img_dir, img_names[0]))
    # intensity = extract_intensity(img0)
    # print(intensity)
    extract_regions(img0, denoise_h=0.005, gau_sigma=33, thres_block=55,
        thres_offset=0, peaks_min_dist=20, min_size=100, save_dir='imgs')
