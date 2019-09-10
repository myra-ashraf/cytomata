import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from cytomata.utils.io import setup_dirs
from configs.mm_settings import MM_CFG_FILE, SETTINGS, INDUCTION, IMAGING, AUTOFOCUS


if __name__ == '__main__':
    # Init
    mscope = Microscope(SETTINGS, MM_CFG_FILE)
    if INDUCTION:
        mscope.queue_induction(**INDUCTION)
    if IMAGING:
        mscope.queue_imaging(**IMAGING)
    if AUTOFOCUS:
        mscope.queue_autofocus(**AUTOFOCUS)

    # Event Loop
    while True:
        done = mscope.run_tasks()
        if done:
            break
        else:
            time.sleep(0.1)
