import numpy as np
import cv2
import mahotas as mh


class Camera(object):
    """Image acquisition from microscope camera"""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()
        cv2.destroyAllWindows()

    def read_raw_img(self):
        ret, frame = self.cap.read()
        return frame

    def measure_fluorescence(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img)
        blur = cv2.GaussianBlur(equ, (5, 5), 0)
        th_bg = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 10)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret, th_roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bg = img * (th_bg > 0)
        bg_intensity = np.percentile(bg[np.nonzero(bg)], 25)
        roi = img * (th_roi > 0)
        roi_intensity = np.median(roi[np.nonzero(roi)]) - bg_intensity
        return roi_intensity, roi, bg_intensity, bg


with Camera(0) as cam:
    while(True):
        img = cam.read_raw_img()
        roi_int, roi, bg_int, bg = cam.measure_fluorescence(img)
        cv2.imshow('Camera Feed', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
