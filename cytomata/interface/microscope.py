import os
import time

import cv2
import MMCorePy
import numpy as np
from scipy import optimize
from skimage import img_as_float
from skimage.filters import laplace


dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """
    MicroManager wrapper for microscope automation, image acquisition,
    image processing, and data recording.
    """
    def __init__(self, ch, mag, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.core.waitForSystem()
        self.set_channel(ch)
        self.set_magnification(mag)
        self.count = 0
        self.times = []
        self.stage_xs = []
        self.stage_ys = []
        self.stage_zs = []
        self.af_positions = []
        self.af_focuses = []
        self.fluo_ints = []

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig('Channel'):
            self.core.setConfig('Channel', chname)
            self.core.waitForConfig('Channel', chname)

    def set_magnification(self, mag):
        if mag != self.core.getState('TINosePiece'):
            self.core.setState('TINosePiece', mag)
            self.core.waitForDevice('TINosePiece')

    def get_position(self, axis):
        if axis.lower() == 'x':
            return self.core.getXPosition('XYStage')
        elif axis.lower() == 'y':
            return self.core.getYPosition('XYStage')
        elif axis.lower() == 'z':
            return self.core.getPosition('TIZDrive')
        else:
            raise ValueError('Invalid axis argument.')

    def set_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setXYPosition('XYStage', value[0], value[1])
        elif axis.lower() == 'z':
            self.core.setPosition('TIZDrive', value)
        else:
            raise ValueError('Invalid axis argument.')
        time.sleep(0.25)

    def shift_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
        elif axis.lower() == 'z':
            self.core.setRelativePosition('TIZDrive', value)
        else:
            raise ValueError('Invalid axis argument.')
        time.sleep(0.25)

    def take_snapshot(self):
        self.core.snapImage()
        return self.core.getImage()

    def record_data(self, save_dir, chs_img, af_channel='DIC', af_method='tr'):
        self.stage_xs.append(self.get_position('x'))
        self.stage_ys.append(self.get_position('y'))
        self.stage_zs.append(self.get_position('z'))
        self.autofocus(ch=af_channel, method=af_method)
        self.times.append(time.time())
        for ch in chs_img:
            self.set_channel(ch)
            img = self.take_snapshot()
            # self.fluo_ints.append(self.measure_fluorescence(img))
            if ch == 'DIC':
                self.af_positions.append(self.get_position('z'))
                self.af_focuses.append(self.measure_focus(img))
            img_path = os.path.join(save_dir, 'imgs', ch, str(self.count) + '.tiff')
            cv2.imwrite(img_path, img)
        data_path = os.path.join(save_dir, 'step_up_down.csv')
        column_names = ', '.join([
            'time (s)', 'x', 'y', 'z', 'af_position', 'af_focus',
        ])
        data = np.column_stack((
            self.times, self.stage_xs, self.stage_ys, self.stage_zs,
            self.af_positions, self.af_focuses,
        ))
        np.savetxt(data_path, data, delimiter=',', header=column_names)
        self.count += 1

    def measure_fluorescence(self, img):
        pass

    def measure_focus(self, img, metric='lap'):
        if metric == 'lap':
            return np.var(laplace(img))
        elif metric == 'vol':
            pass
        elif metric == 'bren':
            pass
        elif metric == 'ten':
            pass
        else:
            raise ValueError('Invalid focus metric.')

    def control_light(self, pattern, ch_exc, ch_dark, duration):
        self.set_channel(ch_exc)
        if pattern == 'pulsatile':
            time.sleep(duration)
            self.set_channel(ch_dark)

    def sample_focus(self):
        pos = self.get_position('z')
        foc = self.measure_focus(self.take_snapshot())
        return pos, foc

    def sample_focus_multi(self, num=4, step=1):
        positions = []
        focuses = []
        pos0 = self.get_position('z')
        self.set_position('z', pos0 - num*step)
        for z in range(num*2 + 1):
            pos, foc = self.sample_focus()
            positions.append(pos)
            focuses.append(foc)
            self.shift_position('z', step)
        self.set_position('z', pos0)
        return positions, focuses

    def autofocus(self, ch='DIC', method='tr', step=5,
        maxiter=9, bounds=[-50.0, 50.0]):
        self.set_channel(ch)
        if method == 'tr': # Top Ranked
            positions, focuses = self.sample_focus_multi()
            best_foc = max(focuses)
            best_pos = positions[focuses.index(max(focuses))]
            if (best_pos > self.stage_zs[0] + bounds[0]
                and best_pos < self.stage_zs[0] + bounds[1]):
                self.set_position('z', best_pos)
        elif method == 'qf':  # Quadratic Fitting
            positions, focuses = self.sample_focus_multi()
            coeffs = np.polyfit(positions, focuses, 2)
            func = np.poly1d(-coeffs)
            results = optimize.minimize(
                func, x0=positions[int(len(positions)//2)])
            best_pos = results.x[0]
            best_foc = func(best_pos)
            if (best_pos > self.stage_zs[0] + bounds[0]
                and best_pos < self.stage_zs[0] + bounds[1]):
                self.set_position('z', best_pos)
        elif method == 'bs':  # Binary Search Hill Climb
            i = 0
            pos, foc = self.sample_focus()
            af_positions = [pos]
            af_focuses = [foc]
            while (af_focuses[-1] < np.max(af_focuses)*0.90
                and af_positions[-1] > af_positions[0] + bounds[0]
                and af_positions[-1] < af_positions[0] + bounds[1]
                and i < maxiter):
                self.shift_position('z', step)
                pos, foc = self.sample_focus()
                if af_focuses[-1] < af_focuses[-2]:
                    step *= -0.5
                i += 1
            best_pos = af_positions[-1]
            best_foc = af_focuses[-1]
        elif method == 'lf':  # Linear Fitting Hill Climb
            if not self.af_positions or not self.af_focuses:
                return self.sample_focus()
            i = 0
            pos, foc = self.sample_focus()
            af_positions = [pos]
            af_focuses = [foc]
            while (af_focuses[-1] < np.max(af_focuses)*0.90
                and af_positions[-1] > af_positions[0] + bounds[0]
                and af_positions[-1] < af_positions[0] + bounds[1]
                and i < maxiter):
                self.shift_position('z', step)
                pos, foc = self.sample_focus()
                af_positions.append(pos)
                af_focuses.append(foc)
                coeffs = np.polyfit(af_positions[-2:], af_focuses[-2:], 1)
                func = np.poly1d(coeffs)
                new_pos = (func - np.max(af_focuses)).roots[0]
                step = new_pos - af_positions[-1]
                i += 1
            best_pos = af_positions[-1]
            best_foc = af_focuses[-1]
        else:  # Reset stage position to initial state
            self.set_position(self.stage_zs[0])
            best_pos = self.stage_zs[0]
            best_foc = 0.0
        return best_pos, best_foc
