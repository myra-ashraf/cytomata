import os
import sys
import time
import shutil
sys.path.append(os.path.abspath('../'))

from cytomata.utils import setup_dirs
from cytomata.mscope import Microscope
from configs.mm_settings import CONFIG_DIR, MM_CFG_FILE, SETTINGS, INDUCTION, IMAGING, AUTOFOCUS


expt_name = raw_input('Expt Directory Name: ')
expt_name = ''.join([x if x.isalnum() or x is '-' else '_' for x in expt_name])
SETTINGS['save_dir'] = os.path.join('expts', time.strftime('%Y%m%d-') + expt_name)
setup_dirs(SETTINGS['save_dir'])
settings_file = os.path.join(CONFIG_DIR, 'mm_settings.py')
log_file = os.path.join(SETTINGS['save_dir'], 'log.txt')
shutil.copyfile(settings_file, log_file)


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
