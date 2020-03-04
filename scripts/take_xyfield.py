import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.utils.io import setup_dirs
from cytomata.interface.microscope import Microscope
from configs.mm_settings import MM_CFG_FILE, SETTINGS


expt_name = raw_input('Expt Directory Name: ')
expt_name = ''.join([x if x.isalnum() or x is '-' else '_' for x in expt_name])
SETTINGS['save_dir'] = os.path.join('expts', time.strftime('%Y%m%d-') + expt_name)

setup_dirs(SETTINGS['save_dir'])
setup_dirs(os.path.join(SETTINGS['save_dir'], 'tasks_log'))
with open(os.path.join(SETTINGS['save_dir'], 'settings.json'), 'w') as fp:
    json.dump(SETTINGS, fp)

mscope = Microscope(SETTINGS, MM_CFG_FILE)
w = SETTINGS['img_width_px'] * SETTINGS['pixel_size']
h = SETTINGS['img_height_px'] * SETTINGS['pixel_size']
step = min(w, h)
mscope.snap_xyfield('YFP', n=11, step=step)
