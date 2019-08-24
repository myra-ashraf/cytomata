import os
import time
from os.path import dirname, abspath, realpath


CONFIG_DIR = dirname(abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_config.cfg')
EXPT_DIR = os.path.join('expts', time.strftime('%Y%m%d-%H%M%S'))
CAM_PIXEL_SIZE = 0.16
CAM_IMG_WIDTH = 512
CAM_IMG_HEIGHT = 512
CAM_EXPOSURE = 100
CAM_GAIN = 1
OBJ_MAG = '100x'
STAGE_Z_LIMIT = [-100, 100]
STAGE_X_LIMIT = [-3600, 3600]
STAGE_Y_LIMIT = [-3600, 3600]