import time
import schedule
import cv2
from cytomata.interface import Microscope, PID


mic = Microscope()
pid = PID(Kp=80.0, Ki=10.0, Kd=0.0, SP=5000.0, windup_limit=10.0)


def update():
    mic.set_channel('mCherry')
    img = mic.take_snapshot()
    roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
    mv = pid.step(roi_int)
    mic.set_channel('Induction-460nm')
    time.sleep(mv)
    mic.set_channel('DIC')


schedule.every(1).minutes.do(update)
while True:
    schedule.run_pending()
