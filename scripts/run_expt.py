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
            't_starts': [0, 901, 1803, 2706, 3611, 4522, 5432, 6353, 7287, 8242],
            't_stops': [1, 903, 1806, 2711, 3619, 4535, 5453, 6387, 7342, 8331],
            't_steps': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
            'kwargs': {
                't_widths': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
                'ch_dark': 'None',
                'ch_ind': 'Blue-Light'
            }
        },
        'imaging': {
            'func': 'image_coords',
            't_starts': [1, 903, 1806, 2711, 3619, 4532, 5453, 6387, 7342, 8331],
            't_stops': [301, 1203, 2106, 3011, 3919, 4832, 5753, 6687, 7642, 8631],
            't_steps': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'kwargs': {
                'chs': ['DIC', 'GFP']
            }
        },
        'autofocus': {
            'func': 'autofocus',
            't_starts': [1, 903, 1806, 2711, 3619, 4532, 5453, 6387, 7342, 8331],
            't_stops': [301, 1203, 2106, 3011, 3919, 4832, 5753, 6687, 7642, 8631],
            't_steps': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
            'kwargs': {
                'ch': 'DIC',
                'algo': 'brent',
                'bounds': [-2.0, 2.0],
                'max_iter': 5
            }
        }
    }

    # Init
    mscope = Microscope(info=info, tasks=tasks)
    # Event Loop
    while True:
        done = mscope.run_tasks()
        if done:
            break
