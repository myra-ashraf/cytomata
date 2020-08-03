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
    'mpos_mode': 'sequential',  # "sequential" or "parallel"
}


# ## Seconds-Timescale ##
IMAGING = {
    't_info': [(0, 56, 5), (60, 62, 1), (65, 301, 5)],  # (start, stop, period)
    'chs': ['mCherry']
}

INDUCTION = {
    't_info': [(60, 61, 1, 1)],  # (start, stop, period, width)
    'ch_ind': 'BL'
}


# # Seconds-Timescale ## Pulsatile ##
# IMAGING = {
#     't_info': [(0, 361, 3)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 120, 2, 0.1), (120, 240, 20, 0.1)],  # (start, stop, period, width)
#     'ch_ind': 'BL'
# }


## Minutes-Timescale ##
# IMAGING = {
#     't_info': [(0, 901, 15)],
#     'chs': ['mCherry', 'YFP']
# }

# INDUCTION = {
#     't_info': [(120, 300, 15, 14)],
#     'ch_ind': 'BL'
# }


# Hours-Timescale ##
# IMAGING = {
#     't_info': [(0, 43200, 60)],
#     'chs': ['YFP', 'DIC']
# }

# INDUCTION = {
#     't_info': [(3600, 7201, 60, 15)],
#     'ch_ind': 'BL10x'
# }


AUTOFOCUS = {
    # 't_info': [(0, 43200, 300)],
    # 'ch': 'DIC',
    # bounds: [-5.0, 5.0],
    # max_iter: 5,
    # offset: 0
}
