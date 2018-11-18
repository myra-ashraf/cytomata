import os
import time

import schedule

from cytomata.interface import Microscope


if __name__ == '__main__':
    # Parameters
    save_dir = os.path.join('experiments', time.strftime('%Y%m%d-%H%M%S'))
    mag = 1
    chs = ['DIC', 'GFP']
    ch_dark = 'None'
    ch_exc = 'Induction-460nm'
    img_int = 300
    t_total = 172800
    t_on = 43200
    t_off = 57600
    t_on_freq = 30
    t_on_dur = 1
    af_ch = 'DIC'
    af_method = 'hc'

    # Scheduling
    mic = Microscope(chs=chs, mag=mag, af_ch=af_ch, af_method=af_method, save_dir=save_dir)
    mic.record_data()
    schedule.every(t_on_freq).seconds.do(
        mic.control_light, ch_exc, ch_dark, t_on, t_off, t_on_dur).tag('light')
    schedule.every(img_int).seconds.do(mic.record_data).tag('data')
    while time.time() - mic.ts[0][0] < t_total:
        schedule.run_pending()
        time.sleep(1)  # avoid high CPU usage
