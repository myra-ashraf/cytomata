import os
import shutil

import numpy as np
from skimage.util import invert
from skimage.io import imread, imsave
from skimage import img_as_uint, img_as_ubyte
import matplotlib.pyplot as plt
from scipy import stats

data_dir = 'data/stage1_val/'
for folder in next(os.walk(data_dir))[1]:
    # Move raw images out of 'images' dir
    imgf = os.path.join(data_dir, folder, 'images', folder + '.png')
    shutil.move(imgf, os.path.join(data_dir, folder, folder + '.png'))
    os.rmdir(os.path.join(data_dir, folder, 'images'))

    # Grayscale 16bit Raw Images
    # imgf = os.path.join(data_dir, folder, folder + '.png')
    # img = imread(imgf, as_gray=True)
    # if np.median(img) > (img.max() - img.min())/2.0:
    #     img = invert(img)
    # imgf = os.path.join(data_dir, folder, folder + '.tiff')
    # imsave(imgf, img_as_uint(img))
    # imgf = os.path.join(data_dir, folder, folder + '.png')
    # os.remove(imgf)

    # Grayscale 16bit Mask Images
    # mask_dir = os.path.join(data_dir, folder, 'masks')
    # for maskf in next(os.walk(mask_dir))[2]:
    #     maskpng = os.path.join(mask_dir, maskf)
    #     mask = imread(maskpng, as_gray=True)
    #     masktiff = os.path.join(mask_dir, maskf.split('.')[0] + '.tiff')
    #     imsave(masktiff, img_as_uint(mask))
    #     os.remove(maskpng)
