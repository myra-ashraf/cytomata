import os
import time
from os.path import dirname, abspath, realpath


CONFIG_DIR = dirname(abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_config.cfg')

SETTINGS = {
    'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    # 'xy_device': 'XYStage',
    'xy_device': None,
    'z_device': 'TIZDrive',
    'img_width': 512,
    'img_height': 512,
    'pixel_size': 0.16,
    'stage_z_limit': [-100, 100],
    'stage_x_limit': [-3600, 3600],
    'stage_y_limit': [-3600, 3600],
    #################################
    'cam_exposure': 50,
    'cam_gain': 1,
    'obj_mag': 4,
    'multi_position': False
}

INDUCTION = {
    't_info': [(1, 31, 5, 5)],
    'ch_ind': 'Blue-Light',
    'ch_dark': 'None'
    'mag': 2
}

IMAGING = {
    't_info': [(0, 300, 10)],
    'chs': ['DIC', 'YFP', 'mCherry']
}

AUTOFOCUS = {
#     't_info': [(0, 86400, 300)],
#     'ch': 'DIC'
}
