import os
import sys
import time
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope


mscope = Microscope(save_dir=os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')), tasks={})
mscope.snap_zstack(ch='GFP', bounds=[4570, 4595], step=0.4)