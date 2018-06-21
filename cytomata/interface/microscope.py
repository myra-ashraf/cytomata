import os
import numpy as np
import cv2
import MMCorePy

dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """MicroManager wrapper for the acquisition and processing of images
    to extract system output information"""

    def __init__(self, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.core.waitForSystem()

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig('Channel'):
            self.core.setConfig('Channel', chname)
            self.core.waitForConfig('Channel', chname)

    def set_magnification(self, mag):
        if mag != self.core.getState('TINosePiece'):
            self.core.setState('TINosePiece', mag)
            self.core.waitForDevice('TINosePiece')

    def take_snapshot(self):
        self.core.snapImage()
        return self.core.getImage()

    def measure_fluorescence(self, img):
        img = img.astype(np.uint8)
        if img.ndim > 2:
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
