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
    'cam_exposure': 200,
    'cam_gain': 1,
    'mpos': False,
    'mpos_ch': 'mCherry',
    'mpos_mode': 'sequential'
}


## Seconds-Timescale ##
IMAGING = {
    't_info': [(0, 61, 5), (71, 301, 5)],
    'chs': ['mCherry']
}

INDUCTION = {
    't_info': [(61, 71, 10, 10)],
    'ch_ind': 'blue-light'
}


## Minutes-Timescale ##
# IMAGING = {
#     't_info': [(0, 61, 5), (61, 662, 10, 10), (662, 1263, 5)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(61, 662, 10, 10)],
#     'ch_ind': 'blue-light'
# }


## Hours-Timescale ##
# IMAGING = {
#     't_info': [(0, 3601, 60), (601, 4202, 60, 60), (4202, 43203, 60)],
#     'chs': ['mCherry']
# }
#
# INDUCTION = {
#     't_info': [(3601, 4202, 60, 60)],
#     'ch_ind': 'blue-light'
# }


AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
