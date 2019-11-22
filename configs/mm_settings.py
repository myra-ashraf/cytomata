import os
import time
from os.path import dirname, abspath, realpath


CONFIG_DIR = dirname(abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')

SETTINGS = {
    'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'lp_device': 'TILightPath',
    'cam_device': 'QuantEM',
    'xy_device': 'XYStage',
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
    'mpos': True,
    'mpos_ch': 'DIC',
    'mpos_mode': 'sequential'
}

IMAGING = {
    't_info': [(0, 120, 5)],
    'chs': ['DIC', 'mCherry']
}

INDUCTION = {
    't_info': [(0, 5, 5, 5)],
    'ch_ind': 'blue-light',
    'ch_dark': 'blue-dark',
}

AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
