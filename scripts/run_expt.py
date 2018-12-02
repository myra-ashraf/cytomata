import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

import schedule

from cytomata.interface import Microscope
from cytomata.utils.io import setup_dirs


if __name__ == '__main__':
    # Experiment Parameters
    params = {
        'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
        'desc': '',
        'mag': 1,
        'chs_img': ['DIC', 'GFP'],
        'ch_dark': 'None',
        'ch_exc': 'Induction-460nm',
        'ch_af': 'DIC',
        'algo_af': 'hc',
        't_img_period': 300,
        't_total': 129600,
        't_exc_on': 43200,
        't_exc_off': 57600,
        't_exc_width': 1,
        't_exc_period': 30
    }

    # Record Parameters
    setup_dirs(params['save_dir'])
    with open(os.path.join(params['save_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    # Schedule Actions
    mic = Microscope(
        save_dir=params['save_dir'],
        mag=params['mag'],
        chs_img=params['chs_img'],
        ch_af=params['ch_af'],
        algo_af=params['algo_af'])
    mic.record_data()
    schedule.every(params['t_exc_period']).seconds.do(
        mic.control_excitation,
        params['ch_dark'],
        params['ch_exc'],
        params['t_exc_on'],
        params['t_exc_off'],
        params['t_exc_width'])
    schedule.every(params['t_img_period']).seconds.do(mic.record_data)
    while time.time() - mic.ts[0][0] < params['t_total']:
        schedule.run_pending()
        time.sleep(1)  # avoid high CPU usage
