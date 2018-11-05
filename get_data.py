import faulthandler
faulthandler.enable()

import os
import time

import schedule

from cytomata.interface import Microscope


def step_input(save_dir, coords_file=None, mag=1, chs_img=['DIC', 'GFP'],
    ch_dark='None', ch_exc='Induction-460nm', pattern='pulsatile',
    t_total=259200, t_on=43200, t_off=259200, t_on_freq=30, t_on_dur=1, img_int=300):
    """
    Use a step input to characterize an optogenetic system.

    Starting from the dark state, the excitation light is turned on at t_on
    and turned off at t_off. Images are taken at regular time intervals via
    the microscope camera and processed to calculate fluorescence intensity
    as the output.

    Args:
        save_dir: File directory to save data for this experiment.
        mag (MM state): Microscope magnification (e.g. 4 = 100x).
        t_total (seconds): Duration of entire experiment.
        t_on (seconds): Timepoint to turn on excitation light.
        t_off (seconds): Timepoint to turn off excitation light.
        t_on_freq (seconds): How often to turn on excitation light.
        t_on_dur (seconds): How long to turn on excitation light.
        img_int (seconds): Time interval for capturing image.
        ch_dark (MM config): Micromanager channel for the dark state.
        ch_exc (MM config): Micromanager channel for induction light.
        chs_img (MM config): Micromanager channels for taking images.
    """
    mic = Microscope(save_dir=save_dir, coords_file=None, chs=chs_img, mag=mag,
        af_ch='DIC', af_method='hc'
    )
    mic.record_data()
    schedule.every(img_int).seconds.do(mic.record_data).tag('data')
    t0 = time.time()
    while time.time() - t0 < t_total:
        # Schedule light induction routine
        if (time.time() >= t0 + t_on and time.time() <= t0 + t_off):
            if 'light' not in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.every(t_on_freq).seconds.do(mic.control_light,
                pattern, ch_exc, ch_dark, t_on_dur).tag('light')
        # Remove light induction routine
        else:
            if 'light' in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.clear('light')
                mic.set_channel(ch_dark)
        schedule.run_pending()
        time.sleep(1)  # schedule needs pauses otherwise program crashes


if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    step_input(save_dir)
