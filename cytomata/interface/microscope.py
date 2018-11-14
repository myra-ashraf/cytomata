import os
import time
from collections import defaultdict

import numpy as np
from skimage import img_as_float
from skimage.io import imsave
from skimage.restoration import denoise_nl_means
from skimage.filters import gaussian, laplace, sobel_h, sobel_v

import MMCorePy


dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """
    MicroManager wrapper for microscope automation, image acquisition,
    image processing, and data recording.
    """
    def __init__(self, save_dir, coords_file=None, chs=['DIC'], mag=1, af_ch='DIC',
        af_method='hc', config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.set_channel(chs[0])
        self.set_magnification(mag)
        self.core.waitForSystem()
        self.save_dir = save_dir
        self.chs = chs
        self.af_ch = af_ch
        self.af_method = af_method
        if coords_file:
            self.coords = np.load(coords_file)
        else:
            self.coords = np.array([[
                self.get_position('x'),
                self.get_position('y'),
                self.get_position('z')
            ]])
        for sample_dir in [os.path.join(save_dir, 'imgs', ch, str(i)) for ch in chs for i in range(len(self.coords))]:
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
        self.count = 0
        self.ts = defaultdict(list)
        self.xs = defaultdict(list)
        self.ys = defaultdict(list)
        self.zs = defaultdict(list)
        self.fls = defaultdict(list)
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
            raise ValueError('Invalid axis argument.')

    def set_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setXYPosition('XYStage', value[0], value[1])
            self.core.waitForDevice('XYStage')
        elif axis.lower() == 'z':
            self.core.setPosition('TIZDrive', value)
            self.core.waitForDevice('TIZDrive')
        else:
            raise ValueError('Invalid axis argument.')

    def shift_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
            self.core.waitForDevice('XYStage')
        elif axis.lower() == 'z':
            self.core.setRelativePosition('TIZDrive', value)
            self.core.waitForDevice('TIZDrive')
        else:
            raise ValueError('Invalid axis argument.')

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig('Channel'):
            self.core.setConfig('Channel', chname)
            self.core.waitForConfig('Channel', chname)
            time.sleep(0.1)

    def set_magnification(self, mag):
        if mag != self.core.getState('TINosePiece'):
            self.core.setState('TINosePiece', mag)
            self.core.waitForDevice('TINosePiece')

    def take_snapshot(self):
        self.core.snapImage()
        return self.core.getImage()

    def measure_fluorescence(self, img):
        den = denoise_nl_means(img_as_float(img), h=0.005, multichannel=False)
        gau = gaussian(den, sigma=30)
        sub = den - gau
        sub[sub < 0] = 0
        return np.mean(sub[sub.nonzero()])

    def control_light(self, ch_exc, ch_dark, t_on, t_off, duration):
        self.ut.append(time.time())
        t = time.time() - self.ts[0][0]
        if t > t_on and t < t_off:
            print('control_light0')
            self.us.append(1.0)
            self.set_channel(ch_exc)
            time.sleep(duration)
            self.set_channel(ch_dark)
            print('control_light1')
        else:
            self.us.append(0.0)

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
            raise ValueError('Invalid focus metric.')

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

    def autofocus(self, step=-1.0, maxiter=7, bounds=[-100.0, 50.0]):
        self.set_channel(self.af_ch)
        z0 = self.coords[0, 2]
        if self.af_method == 'ts':  # Top Sampled
            positions, focuses = self.sample_focus_multi()
            best_foc = max(focuses)
            best_pos = positions[focuses.index(max(focuses))]
        elif self.af_method == 'hc':  # Hill Climb
            iter = 0
            pos, foc = self.sample_focus()
            positions = [pos]
            focuses = [foc]
            while (abs(step) > 0.3 and iter < maxiter
                and positions[-1] > z0 + bounds[0]
                and positions[-1] < z0 + bounds[1]):
                self.shift_position('z', step)
                pos, foc = self.sample_focus()
                if foc < focuses[-1]:
                    step *= -0.5
                positions.append(pos)
                focuses.append(foc)
                iter += 1
            best_foc = max(focuses)
            best_pos = positions[focuses.index(max(focuses))]
        else:  # Reset stage position to initial state
            best_pos = z0
            best_foc = 0.0
        if (best_pos > z0 + bounds[0] and best_pos < z0 + bounds[1]):
            return best_pos, best_foc
        else:
            return z0, 0.0

    def record_position(self):
        x = self.get_position('x')
        y = self.get_position('y')
        z = self.get_position('z')
        data_path = os.path.join(self.save_dir, 'positions.npy')
        if not os.path.isfile(data_path):
            data = np.array([[x, y, z]])
            np.save(data_path, data)
        else:
            data = np.load(data_path)
            data = np.vstack((data, [x, y, z]))
            np.save(data_path, data)

    def save_data(self):
        for i in range(len(self.coords)):
            data_path = os.path.join(self.save_dir, str(i) + '.csv')
            header = ','.join(['t', 'x', 'y', 'z', 'fl'])
            data = np.column_stack((self.ts[i], self.xs[i], self.ys[i], self.zs[i], self.fls[i]))
            np.savetxt(data_path, data, delimiter=',', header=header, comments='')
        u_path = os.path.join(self.save_dir, 'u.csv')
        np.savetxt(u_path, np.column_stack((self.ut, self.us)), delimiter=',', header='t, u', comments='')

    def record_data(self):
        print('autofocus0')
        best_pos, best_foc = self.autofocus()
        print('autofocus1')
        self.coords[:, 2] += (best_pos - self.coords[0, 2])
        print('save_data0')
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            self.ts[i].append(time.time())
            self.xs[i].append(x)
            self.ys[i].append(y)
            self.zs[i].append(z)
            print('save_img0')
            for ch in self.chs:
                self.set_channel(ch)
                img = self.take_snapshot()
                if ch != 'DIC':
                    self.fls[i].append(self.measure_fluorescence(img))
                img_path = os.path.join(self.save_dir, 'imgs', ch, str(i), str(self.count) + '.tiff')
                imsave(img_path, img)
            print('save_img1')
            self.save_data()
        print('save_data1')
        self.count += 1
