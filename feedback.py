import time
import schedule
import matplotlib.pyplot as plt
import cv2
from cytomata.interface import Camera


def task():
    print('hello')


with Camera('QuantEM', 'PrincetonInstruments', 'Camera-1') as cam:
    schedule.every(10).seconds.do(task)
    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        img = cam.get_img()
        roi_int, roi, bg_int, bg = cam.measure_fluorescence(img)
        cv2.imshow('Camera Feed', img)
        # Once every 10 seconds:
        schedule.run_pending()
