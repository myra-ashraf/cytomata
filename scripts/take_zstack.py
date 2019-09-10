import os
import sys
import time
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from configs.mm_settings import MM_CFG_FILE, SETTINGS


mscope = Microscope(SETTINGS, MM_CFG_FILE)
mscope.snap_zstack(bounds=[4570, 4595], step=0.4)
