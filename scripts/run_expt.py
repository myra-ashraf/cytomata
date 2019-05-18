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
    'exposure': 200,
    'gain': 1,
    'mag': 100
    }
    tasks = {
        'induction': {
            'func': 'pulse_light',
            't_starts': [0],
            't_stops': [1],
            't_steps': [1],
            'kwargs': {
                'width': [1],
                'ch_dark': 'None',
                'ch_ind': 'Blue-Light'
            }
        },
        'imaging': {
            'func': 'image_coords',
            't_starts': [1],
            't_stops': [60],
            't_steps': [1],
            'kwargs': {
                'chs': ['DIC', 'mNeonGreen', 'mScarlet']
            }
        }
    }

    setup_dirs(info['save_dir'])
    with open(os.path.join(info['save_dir'], 'params.json'), 'w') as fp:
        json.dump({**info, **tasks}, fp)

    # Init
    mscope = Microscope(save_dir=info['save_dir'], tasks=tasks)
    # Event Loop
    while True:
        done = mscope.run_events()
        if done:
            break
        else:
            time.sleep(0.1)
