import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

from cytomata.interface.microscope import Microscope
from cytomata.utils.io import setup_dirs


if __name__ == '__main__':
    # Expt Parameters
    params = {
        'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
        'desc': 'ND8-exposure200-gain1',
        'mag': 1,
        't_start': 0,
        't_stop': 86400,
        't_img_period': 300,
        'chs_img': ['DIC', 'GFP'],
        'ch_af': 'DIC',
        'algo_af': 'brent',
        'bounds_af': [-3.0, 3.0],
        'max_iter_af': 5,
        't_exc_on': 0,
        't_exc_off': 21600,
        't_exc_width': 1,
        't_exc_period': 30,
        'ch_dark': 'None',
        'ch_exc': 'Induction-460nm'
    }

    # Record Parameters
    setup_dirs(params['save_dir'])
    with open(os.path.join(params['save_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    # Schedule Tasks
    mscope = Microscope(save_dir=params['save_dir'], mag=params['mag'])
    mscope.add_task(
        mscope.image_coords,
        tstart=params['t_start'],
        tstop=params['t_stop'],
        tstep=params['t_img_period'],
        chs_img=params['chs_img'],
        ch_af=params['ch_af'],
        algo_af=params['algo_af'],
        bounds_af=params['bounds_af'],
        max_iter_af=params['max_iter_af']
        )
    mscope.add_task(
        mscope.pulse_light,
        tstart=params['t_start'],
        tstop=params['t_stop'],
        tstep=params['t_exc_period'],
        t_exc_on=params['t_exc_on'],
        t_exc_off=params['t_exc_off'],
        t_exc_width=params['t_exc_width'],
        t_exc_period=params['t_exc_period'],
        ch_dark=params['ch_dark'],
        ch_exc=params['ch_exc']
    )
    # Event Loop
    while True:
        done = mscope.run_tasks()
        if done:
            break
        time.sleep(0.1)
