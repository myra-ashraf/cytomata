from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
    int, map, next, oct, open, pow, range, round, str, super, zip)

import os
import time
import warnings
from collections import defaultdict

import numpy as np
from skimage.io import imsave
from skimage.filters import laplace, sobel_h, sobel_v

import MMCorePy
from cytomata.utils.io import setup_dirs


dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """
    MicroManager wrapper for microscope automation and data recording.
    """
    def __init__(self, save_dir, mag, chs_img, ch_af=None,
        algo_af=None, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.set_channel(chs_img[0])
        self.set_magnification(mag)
        self.core.waitForSystem()
        self.save_dir = save_dir
        self.chs_img = chs_img
        self.ch_af = ch_af
        self.algo_af = algo_af
        self.coords = np.array([[
            self.get_position('x'),
            self.get_position('y'),
            self.get_position('z')
        ]])
        self.count = 0
        self.ts = defaultdict(list)
        self.xs = defaultdict(list)
        self.ys = defaultdict(list)
        self.zs = defaultdict(list)
        self.ut = []
        self.us = []

    def get_position(self, axis):
        if axis.lower() == 'x':
            return self.core.getXPosition('XYStage')
        elif axis.lower() == 'y':
            return self.core.getYPosition('XYStage')
        elif axis.lower() == 'z':
            return self.core.getPosition('TIZDrive')
        else:
            raise ValueError('Invalid axis arg in Microscope.get_position(axis).')

    def set_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setXYPosition('XYStage', value[0], value[1])
            self.core.waitForDevice('XYStage')
        elif axis.lower() == 'z':
            self.core.setPosition('TIZDrive', value)
            self.core.waitForDevice('TIZDrive')
        else:
            raise ValueError('Invalid axis arg in Microscope.set_position(axis).')

    def shift_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
            self.core.waitForDevice('XYStage')
        elif axis.lower() == 'z':
            self.core.setRelativePosition('TIZDrive', value)
            self.core.waitForDevice('TIZDrive')
        else:
            raise ValueError('Invalid axis arg in Microscope.shift_position(axis).')

    def add_position(self):
        x = self.get_position('x')
        y = self.get_position('y')
        z = self.get_position('z')
        self.coords = np.vstack((self.coords, [x, y, z]))

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

    def measure_focus(self, img, metric='lap'):
        if metric == 'var': # Variance of image
            return np.var(img)
        elif metric == 'nvar':  # Normalized variance of image
            return np.var(img)/np.mean(img)
        elif metric == 'lap':  # Variance of laplacian
            return np.var(laplace(img))
        elif metric == 'vol':  # Vollath's F4
            return np.mean(img[:-1, :]*img[1:, :]) - np.mean(img[:-2, :]*img[2:, :])
        elif metric == 'bren':  # Brenner Gradient
            return np.mean((img - np.roll(img, 2, 0))**2)
        elif metric == 'ten':  # Tenengrad
            return np.mean((sobel_h(img)**2) + (sobel_v(img)**2))
        else:
            raise ValueError('Invalid metric arg in Microscope.measure_focus(img, metric).')

    def sample_focus(self):
        pos = self.get_position('z')
        foc = self.measure_focus(self.take_snapshot())
        return pos, foc

    def sample_focus_multi(self, num=3, step=0.5):
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

    def autofocus(self, step=-1.0, min_step=0.2, max_iter=7, bounds=[-100.0, 50.0]):
        if self.ch_af is not None and self.algo_af is not None:
            self.set_channel(self.ch_af)
            z0 = self.coords[0, 2]
            if self.algo_af == 'ts':  # Top Sampled
                positions, focuses = self.sample_focus_multi()
                best_foc = max(focuses)
                best_pos = positions[focuses.index(max(focuses))]
            elif self.algo_af == 'hc':  # Hill Climb
                pos, foc = self.sample_focus()
                positions = [pos]
                focuses = [foc]
                iter = 0
                while (abs(step) > min_step and iter < max_iter
                    and positions[-1] > z0 + bounds[0]
                    and positions[-1] < z0 + bounds[1]):
                    self.shift_position('z', step)
                    pos, foc = self.sample_focus()
                    if foc < focuses[-1]:
                        step *= -0.4
                    positions.append(pos)
                    focuses.append(foc)
                    iter += 1
                best_foc = max(focuses)
                best_pos = positions[focuses.index(max(focuses))]
            else:  # Reset stage position to initial position
                best_pos = z0
                best_foc = 0.0
        if (best_pos > z0 + bounds[0] and best_pos < z0 + bounds[1]):
            return best_pos, best_foc
        else:
            return z0, 0.0

    def record_data(self):
        sample_dirs = [
            os.path.join(self.save_dir, 'imgs', ch, str(i))
            for ch in self.chs_img
            for i in range(len(self.coords))]
        for sample_dir in sample_dirs:
            setup_dirs(sample_dir)
        best_pos, best_foc = self.autofocus()
        self.coords[:, 2] += (best_pos - self.coords[0, 2]) # Update all coords based on 1st coord
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            self.ts[i].append(time.time())
            self.xs[i].append(x)
            self.ys[i].append(y)
            self.zs[i].append(z)
            for ch in self.chs_img:
                self.set_channel(ch)
                img = self.take_snapshot()
                img_path = os.path.join(self.save_dir, 'imgs', ch, str(i), str(self.count) + '.tif')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img)
            data_path = os.path.join(self.save_dir, str(i) + '.csv')
            header = ','.join(['t', 'x', 'y', 'z'])
            data = np.column_stack((self.ts[i], self.xs[i], self.ys[i], self.zs[i]))
            np.savetxt(data_path, data, delimiter=',', header=header, comments='')
        u_path = os.path.join(self.save_dir, 'u.csv')
        np.savetxt(u_path, np.column_stack((self.ut, self.us)),
            delimiter=',', header='t,u', comments='')
        self.count += 1
