import os
import shutil

import numpy as np
from skimage.util import invert
from skimage.io import imread, imsave
from skimage import img_as_uint, img_as_ubyte
import matplotlib.pyplot as plt
from scipy import stats


data_dir = 'data/dsb2018_val/'
new_dir = 'data/dsb2018_inspect/'
for folder in next(os.walk(data_dir))[1]:
    # Move raw images out of 'images' dir
    imgp = os.path.join(data_dir, folder, folder + '.png')
    shutil.copyfile(imgp, os.path.join(new_dir, folder + '.png'))
