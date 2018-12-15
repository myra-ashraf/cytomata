import os
import io
import sys
import time
import json
sys.path.append(os.path.abspath('../'))

import schedule

from cytomata.interface.microscope import Microscope
from cytomata.utils.io import setup_dirs


def control_excitation(mscope, ch_dark, ch_exc, t_exc_on, t_exc_off, t_exc_width, t_exc_period):
    t = time.time() - mscope.ts[0][0]
    mscope.ut += [time.time() + i for i in range(t_exc_period)]
    if t > t_exc_on and t < t_exc_off:
        mscope.us += [1.0] * t_exc_width + [0] * (t_exc_period - t_exc_width)
        mscope.set_channel(ch_exc)
        time.sleep(t_exc_width)
        mscope.set_channel(ch_dark)
    else:
        mscope.us += [0.0] * t_exc_period


if __name__ == '__main__':
    # Experiment Parameters
    params = {
        'save_dir': os.path.join('expts', time.strftime('%Y%m%d-%H%M%S')),
        'desc': 'ND8',
        'mag': 1,
        'chs_img': ['DIC', 'GFP'],
        'ch_dark': 'None',
        'ch_exc': 'Induction-460nm',
        'ch_af': 'DIC',
        'algo_af': 'brent',
        't_img_period': 300,
        't_total': 86400,
        't_exc_on': 43200,
        't_exc_off': 57600,
        't_exc_width': 1,
        't_exc_period': 30
    }

    # Record Parameters
    setup_dirs(params['save_dir'])
    # with io.open(os.path.join(params['save_dir'], 'params.json'), 'w', encoding="utf-8") as fp:
    #     fp.write(unicode(json.dumps(params, ensure_ascii=False)))
    with open(os.path.join(params['save_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    # Schedule Actions
    mscope = Microscope(
        save_dir=params['save_dir'],
        mag=params['mag'],
        chs_img=params['chs_img'],
        ch_af=params['ch_af'],
        algo_af=params['algo_af'])
    mscope.record_data()
    schedule.every(params['t_exc_period']).seconds.do(
        control_excitation,
        mscope,
        params['ch_dark'],
        params['ch_exc'],
        params['t_exc_on'],
        params['t_exc_off'],
        params['t_exc_width'],
        params['t_exc_period'])
    schedule.every(params['t_img_period']).seconds.do(mscope.record_data)

    # Event Loop
    while time.time() - mscope.ts[0][0] < params['t_total']:
        schedule.run_pending()
        time.sleep(1)  # avoid high CPU usage
