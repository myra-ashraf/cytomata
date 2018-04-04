import numpy as np
import cv2
import MMCorePy


class Camera(object):
    """Image acquisition from microscope camera"""
    
    def __init__(self, name='QuantEM', lib='PrincetonInstruments', adapter='Camera-1'):
        self.mmc = MMCorePy.CMMCore()
        self.mmc.loadDevice(name, lib, adapter)
        self.mmc.initializeDevice(name)
        self.mmc.setCameraDevice(name)
        self.mmc.startContinuousSequenceAcquisition(1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cv2.destroyAllWindows()
        self.mmc.stopSequenceAcquisition()
        self.mmc.reset()

    def get_img(self):
        if self.mmc.getRemainingImageCount() > 0:
            return self.mmc.getLastImage()
        return None

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
