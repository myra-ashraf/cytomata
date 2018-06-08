import time
import numpy as np
import cv2
import schedule
from cytomata.control import BangBang, PID
from cytomata.interface import Microscope


def no_light(t_total=21600, img_int=600, ch_dark='None'):
    """Negative control for microscope light induction experiments.

    No light stimulus is applied but images are still taken at regular time
    intervals via the microscope camera and processed to calculate fluorescence
    intensity as the output. At each timepoint, the input (light intensity = 0
    in this case) and output (fluorescence intensity) are logged as csv files
    and images taken are saved as well.

    Args:
        t_total (seconds): How long the enire experiment will last.
        img_int (seconds): Duration of time before repeating script.
    """
    mic = Microscope()
    mic.set_channel(ch_dark)
    data = []

    def tasks():
        img = mic.take_snapshot()
        roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
        data.append([roi_int, bg_int])

    schedule.every(time_interval).seconds.do(tasks)
    t0 = time.time()
    while time.time() - t0 < total_time:
        schedule.run_pending()

    np.savetxt('no_light.csv', np.array(data), delimiter=',',
        header='fluo_intensity, bkg_intensity', comments='')


def step_up_down(t_total=21600, img_int=600, t_on=7200,
    t_off=14400, ch_dark='None', ch_img='GFP', ch_exc='Induction-460nm'):
    """Use step functions to characterize an optogenetic system.

    Starting from the dark state, the excitation light is turned on at t_on
    and turned off at t_off. Images are taken at regular time intervals via
    the microscope camera and processed to calculate fluorescence intensity
    as the output. At each timepoint, the input (light intensity = 0 in this
    case) and output (fluorescence intensity) are logged as csv files and
    images taken are saved as well.

    Args:
        t_total (seconds): How long the enire experiment will last.
        img_int (seconds): Time interval for capturing image.
        t_on (seconds): Timepoint to turn on excitation light.
        t_off (seconds): Timepoint to turn off excitation light.
        ch_dark (MM config): Micromanager channel of the dark state.
        ch_img (MM config): Micromanager channel for taking images.
        ch_exc (MM config): Micromanager channel of induction light.
    """
    mic = Microscope()
    mic.set_channel(ch_dark)
    data = []

    def take_img():
        mic.set_channel(ch_img)
        img = mic.take_snapshot()
        roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
        data.append([roi_int, bg_int])

    schedule.every(time_interval).seconds.do(take_img)
    t0 = time.time()
    while time.time() - t0 < total_time:
        schedule.run_pending()
        if (time.time() > t0 + t_on and time.time() < t0 + t_off
            and mic.get_channel() != ch_exc):
            mic.set_channel(ch_exc)
        else:
            mic.set_channel(ch_dark)

    np.savetxt('step_up_down.csv', np.array(data), delimiter=',',
        header='fluo_intensity, bkg_intensity', comments='')
