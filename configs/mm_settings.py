import os
import time
from os.path import dirname, abspath, realpath


CONFIG_DIR = dirname(abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')

SETTINGS = {
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
    'cam_exposure': 200,
    'cam_gain': 1,
    'mpos': False,
    'mpos_ch': 'mCherry',
    'mpos_mode': 'sequential'
}


## Seconds-Timescale ##
# IMAGING = {
#     't_info': [(0, 61, 5), (65, 301, 5)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 65, 5, 5)],
#     'ch_ind': 'blue-light'
# }


## Minutes-Timescale ##
IMAGING = {
    't_info': [(0, 1381, 15)],
    'chs': ['mCherry', 'YFP']
}

INDUCTION = {
    't_info': [(180, 780, 15, 15)],
    'ch_ind': 'blue-light'
}


## Hours-Timescale ##
# IMAGING = {
#     't_info': [(0, 43201, 60)],
#     'chs': ['mCherry']
# }
#
# INDUCTION = {
#     't_info': [(7200, 10800, 60, 60)],
#     'ch_ind': 'blue-light'
# }


AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
