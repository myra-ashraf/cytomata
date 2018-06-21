import os
import time
import datetime

import cv2
import numpy as np
import schedule

from cytomata.control import BangBang, PID
from cytomata.interface import Microscope


def step_up_down(save_dir, mag=4, t_total=36000, img_int=300, t_on=300,
    t_on_freq=30, t_on_dur=2, t_off=21600, ch_dark='None',
    chs_img=['GFP', 'mCherry'], ch_exc='Induction-460nm'):
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
        img_int (seconds): Time interval for capturing image.
        t_on (seconds): Timepoint to turn on excitation light.
        t_on_freq (seconds): How often to turn on excitation light.
        t_on_dur (seconds): How long to turn on excitation light.
        t_off (seconds): Timepoint to turn off excitation light.
        ch_dark (MM config): Micromanager channel of the dark state.
        chs_img (MM config): Micromanager channels for taking images.
        ch_exc (MM config): Micromanager channel of induction light.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_dirs = [os.path.join(save_dir, 'imgs', ch) for ch in chs_img]
    for d in img_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    mic = Microscope()
    mic.set_channel(ch_dark)
    mic.set_magnification(mag)
    data = []

    def control_light():
        print('control_light')
        mic.set_channel(ch_exc)
        time.sleep(t_on_dur)
        mic.set_channel(ch_dark)

    def record_data():
        print('record_data')
        d = []
        for ch in chs_img:
            mic.set_channel(ch)
            img = mic.take_snapshot()
            ts = time.time()
            img_path = os.path.join(
                save_dir, 'imgs', ch, str(round(ts)) + '.png')
            cv2.imwrite(img_path, img)
            roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
            d += [roi_int, bg_int]
        data.append([ts] + d)
        data_path = os.path.join(save_dir, 'step_up_down.csv')
        column_names = ', '.join(['time (s)'] + [
            ch + d for ch in chs_img for d in ['_fluo_int', '_bg_int']])
        np.savetxt(data_path, np.array(data), delimiter=',',
            header=column_names, comments='')

    schedule.every(img_int).seconds.do(record_data).tag('data')
    t0 = time.time()
    while time.time() - t0 < t_total:
        schedule.run_pending()
        if (time.time() >= t0 + t_on and time.time() <= t0 + t_off):
            if 'light' not in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.every(t_on_freq).seconds.do(control_light).tag('light')
            print('light on')
        else:
            if 'light' in [list(j.tags)[0] for j in schedule.jobs]:
                schedule.clear('light')
            mic.set_channel(ch_dark)
        time.sleep(1)



if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    step_up_down(save_dir)
