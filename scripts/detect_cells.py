import os
import sys
import time
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import img_as_float
from skimage.io import imread
from skimage.filters import gaussian, threshold_local
from skimage.restoration import estimate_sigma, denoise_nl_means
from skimage.morphology import disk, opening, white_tophat

from cytomata.utils.io import setup_dirs, list_img_files
from cytomata.process.detect import train
from cytomata.utils.visual import imshow, plot, custom_styles, custom_palette


train_dir = 'data/dsb2018_train/'
val_dir = 'data/dsb2018_val/'
# train_dir = 'data/dsb2018_train/'
# val_dir = 'data/dsb2018_val/'
train(train_dir=train_dir, val_dir=val_dir, weights='data/results/mask_rcnn_cyto_train_8.h5')
