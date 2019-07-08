import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from cytomata.utils.io import setup_dirs


if __name__ == '__main__':
    # Expt Parameters
    info = {
    'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
    'ND_filter': 4,
    'exposure': 400,
    'gain': 1,
    'mag': 100
    }
    tasks = {
        'induction': {
            'func': 'pulse_light',
            't_starts': [0],
            't_stops': [86400],
            't_steps': [30],
            'kwargs': {
                'width': [1],
                'ch_dark': 'None',
                'ch_ind': 'Blue-Light'
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
                'bounds': [-2.0, 2.0],
                'max_iter': 5
            }
        }
    }

    setup_dirs(info['save_dir'])
    expt_log = info.copy()
    expt_log.update(tasks)
    with open(os.path.join(info['save_dir'], 'params.json'), 'w') as fp:
        json.dump(expt_log, fp)

    # Init
    mscope = Microscope(save_dir=info['save_dir'], tasks=tasks)
    # Event Loop
    while True:
        done = mscope.run_events()
        if done:
            break
        else:
            time.sleep(0.1)
