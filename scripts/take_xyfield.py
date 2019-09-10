import os
import sys
import time
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from configs.mm_settings import MM_CFG_FILE, SETTINGS


mscope = Microscope(SETTINGS, MM_CFG_FILE)
w = SETTINGS['img_width'] * SETTINGS['pixel_size']
h = SETTINGS['img_height'] * SETTINGS['pixel_size']
step = min(w, h)
mscope.snap_xyfield(n=5, step=step)
