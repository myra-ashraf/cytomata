import os
import time
import datetime

import cv2
import numpy as np
import schedule

from cytomata.control import BangBang, PID
from cytomata.interface import Microscope


def no_light(save_dir, t_total=21600, img_int=600, ch_dark='None'):
    """Negative control for microscope light induction experiments.

    No light stimulus is applied but images are still taken at regular time
    intervals via the microscope camera and processed to calculate fluorescence
    intensity as the output. At each timepoint, the input (light intensity = 0
    in this case) and output (fluorescence intensity) are logged as csv files
    and images taken are saved as well.

    Args:
        save_dir: File directory to save data for this experiment.
        t_total (seconds): How long the enire experiment will last.
        img_int (seconds): Duration of time before repeating script.
        ch_dark (MM config): Micromanager channel of the dark state.
    """
    mic = Microscope()
    mic.set_channel(ch_dark)
    data = []

    def tasks():
        img = mic.take_snapshot()
        roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
        data.append([roi_int, bg_int])

    schedule.every(img_int).seconds.do(tasks)
    t0 = time.time()
    while time.time() - t0 < t_total:
        schedule.run_pending()

    np.savetxt('no_light.csv', np.array(data), delimiter=',',
        header='fluo_intensity, bkg_intensity', comments='')


def step_up_down(save_dir, t_total=21600, img_int=600, t_on=7200,
    t_off=14400, ch_dark='None', chs_img=['GFP', 'mCherry'],
    ch_exc='Induction-460nm'):
    """Use step functions to characterize an optogenetic system.

    Starting from the dark state, the excitation light is turned on at t_on
    and turned off at t_off. Images are taken at regular time intervals via
    the microscope camera and processed to calculate fluorescence intensity
    as the output. At each timepoint, the input (light intensity = 0 in this
    case) and output (fluorescence intensity) are logged as csv files and
    images taken are saved as well.

    Args:
        save_dir: File directory to save data for this experiment.
        t_total (seconds): How long the enire experiment will last.
        img_int (seconds): Time interval for capturing image.
        t_on (seconds): Timepoint to turn on excitation light.
        t_off (seconds): Timepoint to turn off excitation light.
        ch_dark (MM config): Micromanager channel of the dark state.
        chs_img (MM config): Micromanager channels for taking images.
        ch_exc (MM config): Micromanager channel of induction light.
    """
    os.makedirs(save_dir, exist_ok=True)

    mic = Microscope()
    mic.set_channel(ch_dark)
    data = []

    def record_data():
        d = []
        for ch in chs_img:
            mic.set_channel(ch)
            img = mic.take_snapshot()
            ts = time.time()
            img_path = os.path.join(
                save_dir, 'imgs', ch, str(round(ts)) + '.png')
            cv2.imwrite(img_path, img)
            roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
            d.append(roi_int, bg_int)
        data.append([ts] + d)

    schedule.every(img_int).seconds.do(record_data)
    t0 = time.time()
    while time.time() - t0 < t_total:
        schedule.run_pending()
        if (time.time() > t0 + t_on and time.time() < t0 + t_off
            and mic.get_channel() != ch_exc):
            mic.set_channel(ch_exc)
        else:
            mic.set_channel(ch_dark)

    data_path = os.path.join(save_dir, 'step_up_down.csv')
    column_names = ', '.join(['time (s)'] + [
        ch + d for ch in chs_img for d in ['_fluo_int', '_bg_int']])
    np.savetxt(data_path, np.array(data), delimiter=',',
        header=column_names, comments='')


if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    step_up_down(save_dir)
