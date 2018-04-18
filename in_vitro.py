import time
import numpy as np
import cv2
import schedule
from cytomata.control import BangBang, PID
from cytomata.interface import Microscope


def no_light_negative_control(total_time=43200, time_interval=60):
    """Negative control for microscope light induction experiments.

    No light stimulus is applied but images are still taken at regular time
    intervals via the microscope camera and processed to calculate fluorescenc
    intensity as the output. At each timepoint, the input (light intensity = 0
    in this case) and output (fluorescence intensity) are logged as csv files
    and images taken are saved as well.

    Args:
        total_time (seconds): How long the enire experiment will last.
        time_interval (seconds): Duration of time before repeating script.
    """
    mic = Microscope()
    mic.set_channel('None')
    data = []

    def tasks():
        img = mic.take_snapshot()
        roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
        data.append([0.0, roi_int, bg_int])

    schedule.every(time_interval).seconds.do(tasks)
    t0 = time.time()
    while time.time() - t0 < total_time:
        schedule.run_pending()

    np.savetxt('no_light.csv', np.array(data), delimiter=',',
        header='light_intensity, fluo_intensity, bkg_intensity', comments='')

def constant_light_positive_control():
    pass
