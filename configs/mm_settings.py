import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')


SETTINGS = {
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'xy_device': 'XYStage',
    'z_device': 'TIZDrive',
    'img_sync': ['XYStage', 'TIZDrive', 'Wheel-A', 'Wheel-B', 'Wheel-C'],
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


# ## Seconds-Timescale ##
# IMAGING = {
#     't_info': [(0, 601, 5)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 540, 5, 1)],
#     'ch_ind': 'blue-light'
# }


# Minutes-Timescale ##
IMAGING = {
    't_info': [(0, 901, 15)],
    'chs': ['mCherry', 'YFP']
}

INDUCTION = {
    't_info': [(180, 240, 60, 60)],
    'ch_ind': 'blue-light'
}


## Hours-Timescale ##
# IMAGING = {
#     't_info': [(0, 43201, 60)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     # 't_info': [(7200, 10800, 60, 5)],
#     # 'ch_ind': 'blue-light'
# }


AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
