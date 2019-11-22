import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.utils.io import setup_dirs
from cytomata.interface.microscope import Microscope
from configs.mm_settings import MM_CFG_FILE, SETTINGS, INDUCTION, IMAGING, AUTOFOCUS


notes = raw_input('Notes: ')
SETTINGS['notes'] = notes

setup_dirs(SETTINGS['save_dir'])
setup_dirs(os.path.join(SETTINGS['save_dir'], 'tasks_log'))
with open(os.path.join(SETTINGS['save_dir'], 'settings.json'), 'w') as fp:
    json.dump(SETTINGS, fp)

mscope = Microscope(SETTINGS, MM_CFG_FILE)
if SETTINGS['mpos']:
    mscope.add_coords_session(SETTINGS['mpos_ch'])

# Event Loop
if SETTINGS['mpos'] and SETTINGS['mpos_mode'] == 'sequential':
    for cid in range(len(mscope.coords)):
        mscope.cid = cid
        mscope.t0 = time.time()
        if IMAGING:
            mscope.queue_imaging(**IMAGING)
        if INDUCTION:
            mscope.queue_induction(**INDUCTION)
        if AUTOFOCUS:
            mscope.queue_autofocus(**AUTOFOCUS)
        while True:
            done = mscope.run_tasks()
            if done:
                break
            else:
                time.sleep(0.001)
else:
    while True:
        done = mscope.run_tasks()
        if done:
            break
        else:
            time.sleep(0.001)
