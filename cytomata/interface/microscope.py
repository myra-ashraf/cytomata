import os

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

    def shift_position(self, axis, value):
        if axis == 'XY':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
        elif axis == 'Z':
            self.core.setRelativePosition('TIZDrive', value)
        else:
            print('Error shifting stage position')

    def take_snapshot(self):
        self.core.snapImage()
        return self.core.getImage()

    def measure_fluorescence(self, img):
        pass

    def measure_focus(self, img):
        return np.var(laplace(img))

    def sample_pos_focus(self, num=10, step=5, pos0=None, clear_data=True):
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
        if clear_data:
            self.af_positions = []
            self.af_focuses = []
        for z in range(num*2):
            img = self.take_snapshot()
            self.af_focuses.append(self.measure_focus(img))
            self.af_positions.append(self.get_position('Z'))
            self.shift_position('Z', step)
        self.set_position('Z', pos0)
        return positions, focuses

    def autofocus(self, positions=self.af_positions, focuses=self.af_focuses):
        coeffs = np.polyfit(positions, focuses, 2)
        func = np.poly1d(-coeffs)
        results = opt.minimize(func, x0=positions[-1])
        best_z = results.x[0]
        self.set_position('Z', best_z)
        return best_z, func, results
