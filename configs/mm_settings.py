import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')


SETTINGS = {
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'xy_device': 'XYStage',
    'z_device': 'TIZDrive',
    'img_width_px': 512,
    'img_width_um': 81.92,
    'img_height_px': 512,
    'img_height_um': 81.92,
    'pixel_size': 0.16,
    'stage_z_limit': [-240, 240],
    'stage_x_limit': [-3600, 3600],
    'stage_y_limit': [-3600, 3600],
    'mpos': False,
    'mpos_ch': 'mCherry',
    'mpos_mode': 'sequential',
}


## Seconds-Timescale ##
IMAGING = {
    't_info': [(0, 301, 5)],  # (start, stop, period)
    'chs': ['mCherry']
}

INDUCTION = {
    't_info': [(60, 180, 15, 0.1)],  # (start, stop, period, width)
    'ch_ind': 'blue-light'
}


## Minutes-Timescale ##
# IMAGING = {
#     't_info': [(0, 901, 15)],
#     'chs': ['mCherry', 'YFP']
# }

# INDUCTION = {
#     't_info': [(120, 300, 15, 14)],
#     'ch_ind': 'blue-light'
# }


## Hours-Timescale ##
# IMAGING = {
#     't_info': [(0, 43200, 60)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(7200, 7800, 60, 58)],
#     'ch_ind': 'blue-light'
# }


AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
