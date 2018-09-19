import os
import time

import MMCorePy
import numpy as np
from scipy import optimize
from skimage.filters import laplace

dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """MicroManager wrapper for the acquisition and processing of images
    to extract system output information"""

    def __init__(self, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.core.setExposure(200)
        self.core.waitForSystem()
        self.af_positions = []
        self.af_focuses = []

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig('Channel'):
            self.core.setConfig('Channel', chname)
            self.core.waitForConfig('Channel', chname)

    def set_magnification(self, mag):
        if mag != self.core.getState('TINosePiece'):
            self.core.setState('TINosePiece', mag)
            self.core.waitForDevice('TINosePiece')

    def get_position(self, axis):
        if axis == 'XY':
            return self.core.getXPosition('XYStage'), self.core.getYPosition('XYStage')
        elif axis == 'Z':
            return self.core.getPosition('TIZDrive')
        else:
            print('Error getting stage position')

    def set_position(self, axis, value):
        if axis == 'XY':
            self.core.setXYPosition('XYStage', value[0], value[1])
        elif axis == 'Z':
            self.core.setPosition('TIZDrive', value)
        else:
            print('Error setting stage position')
        time.sleep(1)

    def shift_position(self, axis, value):
        if axis == 'XY':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
        elif axis == 'Z':
            self.core.setRelativePosition('TIZDrive', value)
        else:
            print('Error shifting stage position')
        time.sleep(1)

    def take_snapshot(self):
        self.core.snapImage()
        return self.core.getImage()

    def measure_fluorescence(self, img):
        pass

    def measure_focus(self, img, metric='lap'):
        if metric == 'lap':
            return np.var(laplace(img))
        elif metric == 'vol':


    def sample_focus(self):
        self.af_positions.append(self.get_position('Z'))
        self.af_focuses.append(self.measure_focus(self.take_snapshot()))
        return self.af_positions, self.af_focuses

    def sample_focus_multi(self, num=2, step=3, pos0=None, clear_data=True):
        """
        Sample different stage positions about the current position
        and calculate variance of laplace for each image.

        Args:
            num (int): Number of positions above or below the current position
            step (int): Difference in stage position in microns
            pos0 (float): Specified position instead of current position

        Returns:
            new_positions (list(floats)): Stage positions for each image
            new_focuses (list(floats)): Variance of laplace for each image
        """
        if pos0 is None:
            pos0 = self.get_position('Z')
        self.set_position('Z', pos0 - num*step)
        self.set_channel('DIC')
        af_focuses = []
        af_positions = []
        for z in range(num*2 + 1):
            img = self.take_snapshot()
            af_focuses.append(self.measure_focus(img))
            af_positions.append(self.get_position('Z'))
            self.shift_position('Z', step)
        self.set_position('Z', pos0)
        return af_positions, af_focuses

    def autofocus_qf(self, bounds=[-50.0, 50.0]):
        positions, focuses = self.sample_focus_multi()
        coeffs = np.polyfit(positions, focuses, 2)
        func = np.poly1d(-coeffs)
        results = optimize.minimize(func, x0=positions[-1])
        best_z = results.x[0]
        if best_z > self.init_z + bounds[0] and best_z < self.init_z + bounds[1]:
            self.set_position('Z', best_z)
        return best_z


    def autofocus_lf(self, step=5, bounds=[50.0, 50.0], max_iter=20):
        if not self.af_positions or not self.af_focuses:
            self.sample_focus()
            return self.af_focuses[-1]
        self.sample_focus()
        while (self.af_focuses[-1] < np.max(self.af_focuses)*0.95
            and self.af_positions[-1] > self.af_positions[0] + bounds[0]
            and self.af_positions[-1] < self.af_positions[0] + bounds[1]
            and i < max_iter):
            self.shift_position('Z', step)
            self.sample_focus()
            coeffs = np.polyfit(self.af_positions[-2:], self.af_focuses[-2:], 1)
            func = np.poly1d(coeffs)
            new_pos = (func - np.max(self.af_focuses)).roots[0]
            step = new_pos - self.af_positions[-1]
        return self.af_focuses[-1]

    def autofocus_bi(self, step=10, bounds=[50.0, 50.0], max_iter=20):
        self.sample_focus()
        while (self.af_focuses[-1] < np.max(self.af_focuses)*0.95
            and self.af_positions[-1] > self.af_positions[0] + bounds[0]
            and self.af_positions[-1] < self.af_positions[0] + bounds[1]
            and i < max_iter):
            self.shift_position('Z', step)
            self.sample_focus()
            if self.af_focuses[-1] < self.af_focuses[-2]:
                step *= -0.5
