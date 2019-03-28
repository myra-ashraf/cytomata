import os
import shutil

import numpy as np
from skimage.util import invert
from skimage.io import imread, imsave
from skimage import img_as_uint
import matplotlib.pyplot as plt
from scipy import stats

i = 0
data_dir = 'data/wahlby/'
for folder in next(os.walk(data_dir))[1]:
    well_dir = os.path.join(data_dir, folder)
    for imgf in next(os.walk(well_dir))[2]:
        img_path = os.path.join(well_dir, imgf)
        if '_w1_' in imgf or '_w2_' in imgf:
            shutil.move(img_path, os.path.join(data_dir, str(i) + '.tiff'))
            i += 1
    shutil.rmtree(well_dir)
