import os
import sys
import time
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope


mscope = Microscope(save_dir=os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')), tasks={})
mscope.snap_xyfield(ch='GFP', n=5, step=81.92)
