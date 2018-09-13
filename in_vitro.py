import os
import time
import datetime

import schedule
import numpy as np
from skimage.io import imsave

from cytomata.interface import Microscope


def step_up_down(save_dir, mag=1, t_total=129600, t_on=0, t_off=129600,
    t_on_freq=30, t_on_dur=1, img_int=300, ch_dark='None', ch_exc='Induction-460nm',
    chs_img=['DIC', 'GFP']):
    """Use step functions to characterize an optogenetic system.

    Starting from the dark state, the excitation light is turned on at t_on
    and turned off at t_off. Images are taken at regular time intervals via
    the microscope camera and processed to calculate fluorescence intensity
    as the output. At each timepoint, the input (light intensity = 0 in this
    case) and output (fluorescence intensity) are logged as csv files and
    images taken are saved as well.

    Args:
        save_dir: File directory to save data for this experiment.
        mag (MM state): Microscope default magnification (e.g. 4 = 100x).
        t_total (seconds): How long the enire experiment will last.
        t_on (seconds): Timepoint to turn on excitation light.
        t_off (seconds): Timepoint to turn off excitation light.
        t_on_freq (seconds): How often to turn on excitation light.
        t_on_dur (seconds): How long to turn on excitation light.
        img_int (seconds): Time interval for capturing image.
        ch_dark (MM config): Micromanager channel of the dark state.
        ch_exc (MM config): Micromanager channel of induction light.
        chs_img (MM config): Micromanager channels for taking images.
    """
    def control_light():
        mic.set_channel(ch_exc)
        time.sleep(t_on_dur)
        mic.set_channel(ch_dark)

    def record_data():
        d = [] # per-timepoint data vector
        # mic.set_position('XY', stage_pos[:2])
        # mic.set_position('Z', stage_pos[2])
        # Forget old autofocusing data
        if len(mic.af_positions) > 25 and len(mic.af_focuses) > 25:
            mic.af_positions.pop(0)
            mic.af_focuses.pop(0)
        best_z, af_func, af_optim = mic.autofocus()
        for ch in chs_img:
            mic.set_channel(ch)
            img = mic.take_snapshot()
            if ch == 'DIC':  # autofocus is based on DIC channel
                pos = mic.get_position('Z')
                foc = mic.measure_focus(img)
                mic.af_positions.append(pos)
                mic.af_focuses.append(foc)
            ts = time.time()
            img_path = os.path.join(
                save_dir, 'imgs', ch, str(data_count) + '.tiff')
            imsave(img_path, img)
            # roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
            # d += [roi_int, bg_int]
            d += [pos, foc, best_z]
        current_pos = list(mic.get_position('XY')) + [mic.get_position('Z')]
        data.append([data_count, ts] + d + current_pos)
        data_count += 1
        data_path = os.path.join(save_dir, 'step_up_down.csv')
        # column_names = ', '.join(['time (s)'] + [
        #     ch + d for ch in chs_img for d in ['_fluo_int', '_bg_int']])
        column_names = ', '.join([
            'index', 'time (s)', 'position', 'focus', 'best_z'
            'stage-x (um)', 'stage-y (um)', 'stage-z (um)'
        ])
        np.savetxt(data_path, np.array(data), delimiter=',',
            header=column_names, comments='')

    # BEGIN Experiment
    # Create data and image directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_dirs = [os.path.join(save_dir, 'imgs', ch) for ch in chs_img]
    for d in img_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # Initialize Microscope controller
    mic = Microscope()
    mic.set_magnification(mag)
    # Acquire data to be used in autofocusing routine
    mic.sample_pos_focus(num=10, step=5)
    mic.set_channel(ch_dark)
    # stage_pos = list(mic.get_position('XY')) + [mic.get_position('Z')]

    data = []
    data_count = 0
    # Acquire data and images for time = 0
    record_data()
    # Schedule data recording routine
    schedule.every(img_int).seconds.do(record_data).tag('data')
    t0 = time.time()
    # While in timeframe for experiment:
    while time.time() - t0 < t_total:
        schedule.run_pending()
        # Schedule light induction routine when within certain timeframes
        if (time.time() >= t0 + t_on and time.time() <= t0 + t_off):
            if 'light' not in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.every(t_on_freq).seconds.do(control_light).tag('light')
        # Remove light induction routine when out of certain timeframes
        else:
            if 'light' in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.clear('light')
            mic.set_channel(ch_dark)
            time.sleep(1) # schedule needs pauses otherwise program crashes
        time.sleep(1)



if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    step_up_down(save_dir)
