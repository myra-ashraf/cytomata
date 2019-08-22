import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from cytomata.utils.io import setup_dirs


if __name__ == '__main__':
    # Expt Parameters
    settings = {
        'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
        'ND_filter': 4,
        'pixel_size': 0.16,
        'img_width': 512,
        'img_height': 512,
        'exposure': 400,
        'gain': 1,
        'mag': 100,
        'z_bound': [-100, 100],
        'x_bound': [-720, 720],
        'y_bound': [-720, 720],
    }
    tasks = {
        'induction': {
            'func': 'pulse_light',
            't_starts': [0],
            't_stops': [86400],
            't_steps': [60],
            'kwargs': {
                'width': [5],
                'ch_dark': 'None',
                'ch_ind': 'Blue-Light',
                'mag': 2
            }
        },
        'imaging': {
            'func': 'image_coords',
            't_starts': [0],
            't_stops': [86400],
            't_steps': [300],
            'kwargs': {
                'chs': ['DIC', 'GFP', 'mCherry']
            }
        },
        'autofocus': {
            'func': 'autofocus',
            't_starts': [0],
            't_stops': [86400],
            't_steps': [300],
            'kwargs': {
                'ch': 'DIC',
                'algo': 'brent',
                'bounds': [-30.0, 30.0],
                'max_iter': 5
            }
        }
    }

    # Log Info
    setup_dirs(info['save_dir'])
    expt_log = settings.copy()
    expt_log.update(tasks)
    with open(os.path.join(info['save_dir'], 'expt_log.json'), 'w') as fp:
        json.dump(expt_log, fp)

    # Init
    mscope = Microscope(settings=settings, tasks=tasks)
    # Event Loop
    while True:
        done = mscope.run_events()
        if done:
            break
        else:
            time.sleep(0.1)
