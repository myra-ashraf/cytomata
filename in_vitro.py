import os
import time

import schedule

from cytomata.interface import Microscope


def step_up_down(save_dir, mag=1, t_total=259200, t_on=43200, t_off=259200,
    pattern='pulsatile', t_on_freq=30, t_on_dur=1, img_int=300, ch_dark='None',
    ch_exc='Induction-460nm', chs_img=['DIC', 'GFP']):
    """
    Use step functions to characterize an optogenetic system.

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
    # Create data and image directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_dir in [os.path.join(save_dir, 'imgs', ch) for ch in chs_img]:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    # Initialize Microscope controller
    mic = Microscope(ch=chs_img[0], mag=mag)
    # Acquire data and images for time = 0
    mic.record_data(save_dir, chs_img)
    # Schedule data recording routine
    schedule.every(img_int).seconds.do(
        mic.record_data, save_dir, chs_img).tag('data')
    t0 = time.time()
    # While in timeframe for experiment:
    while time.time() - t0 < t_total:
        # Schedule light induction routine
        if (time.time() >= t0 + t_on and time.time() <= t0 + t_off):
            if 'light' not in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.every(t_on_freq).seconds.do(mic.control_light,
                pattern, ch_exc, ch_dark, t_on_dur).tag('light')
                time.sleep(1)
        # Remove light induction routine
        else:
            if 'light' in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.clear('light')
                mic.set_channel(ch_dark)
                time.sleep(1)
        schedule.run_pending()
        time.sleep(1)  # schedule needs pauses otherwise program crashes


if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    step_up_down(save_dir)
