import os
import sys
import time
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from configs.mm_settings import MM_CFG_FILE, SETTINGS, INDUCTION, IMAGING, AUTOFOCUS


notes = raw_input('Expt Notes (e.g. ND filters, expt group)?')
SETTINGS['notes'] = notes

mscope = Microscope(SETTINGS, MM_CFG_FILE)
if SETTINGS['multi_position']:
    mscope.add_coords_session()
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
