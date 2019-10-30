import os
import time
from os.path import dirname, abspath, realpath


CONFIG_DIR = dirname(abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')

SETTINGS = {
    'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'cam_device': 'QuantEM',
    'xy_device': 'XYStage',
    # 'xy_device': None,
    'z_device': 'TIZDrive',
    'img_width_px': 512,
    'img_width_um': 81.92,
    'img_height_px': 512,
    'img_height_um': 81.92,
    'pixel_size': 0.16,
    'stage_z_limit': [-100, 100],
    'stage_x_limit': [-3600, 3600],
    'stage_y_limit': [-3600, 3600],
    #################################
    'cam_exposure': 100,
    'cam_gain': 1,
    'obj_mag': 0,
    'multi_position': False,
    'mpos_ch': 'mCherry'
}

IMAGING = {
    't_info': [(0, 43200, 300)],
    'chs': ['DIC', 'GFP']
}

INDUCTION = {
    't_info': [(0, 43200, 30, 1)],
    'ch_ind': 'blue-light',
    'ch_dark': 'blue-dark',
    'mag': 5
}

AUTOFOCUS = {
    # 't_info': [(0, 1200, 60)],
    # 'ch': 'mCherry'
}
