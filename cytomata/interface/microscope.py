import os
import time
import warnings
from collections import defaultdict, deque

import numpy as np
from skimage.io import imsave
from skimage.filters import laplace, sobel_h, sobel_v
from scipy.optimize import minimize_scalar

import MMCorePy
from cytomata.utils.io import setup_dirs


dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """
    Microscope task automation and data recording.
    """
    def __init__(self, save_dir, mag=1, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.set_magnification(mag)
        self.core.waitForSystem()
        self.save_dir = save_dir
        self.saved = False
        self.tasks = []
        self.coords = np.array([[
            self.get_position('x'),
            self.get_position('y'),
            self.get_position('z')
        ]])
        self.xt = defaultdict(list)
        self.xx = defaultdict(list)
        self.xy = defaultdict(list)
        self.xz = defaultdict(list)
        self.ut = defaultdict(list)
        self.uu = defaultdict(list)
        self.t0 = time.time()

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

    def add_coord(self):
        x = self.get_position('x')
        y = self.get_position('y')
        z = self.get_position('z')
        self.coords = np.vstack((self.coords, [x, y, z]))

    def snap_image(self):
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
        foc = self.measure_focus(self.snap_image())
        return pos, foc

    def sample_focus_stack(self, bounds=[-3.0, 3.0], num=7):
        z0 = self.coords[0, 2]
        zi = self.get_position('z')
        zl = np.max([zi + bounds[0], z0 - 50.0])
        zu = np.min([zi + bounds[1], z0 + 50.0])
        positions = list(np.linspace(zl, zu, num))
        focuses = []
        for z in positions:
            self.set_position('z', z)
            _, foc = self.sample_focus()
            focuses.append(foc)
        self.set_position('z', zi)
        return positions, focuses

    def autofocus(self, ch_af='DIC', algo_af='brent', bounds_af=[-3.0, 3.0], max_iter_af=5):
        self.set_channel(ch_af)
        z0 = self.coords[0, 2]
        if algo_af == 'ts':  # Grid Search
            positions, focuses = self.sample_focus_stack(bounds=bounds_af, num=max_iter_af)
            best_foc = max(focuses)
            best_pos = positions[focuses.index(max(focuses))]
        elif algo_af == 'brent':  # Brent's Method
            def residual(z):
                self.set_position('z', z)
                foc = self.sample_focus()[1]
                return -foc
            zi = self.get_position('z')
            zl = np.max([zi - 5, z0 + bounds_af[0]])
            zu = np.min([zi + 5, z0 + bounds_af[1]])
            result = minimize_scalar(residual, method='bounded',
                bounds=(zl, zu), options={'maxiter': max_iter_af, 'xatol': 3.0})
            best_pos, best_foc = result.x, -result.fun
        else:  # Reset Position
            best_pos = z0
            best_foc = 0.0
        return best_pos, best_foc

    def image_coords(self, chs_img, **kwargs):
        if not self.saved:
            for ch in chs_img:
                for i in range(len(self.coords)):
                    setup_dirs(os.path.join(self.save_dir, ch, str(i)))
            self.saved = True
        best_pos, best_foc = self.autofocus(**kwargs)
        self.coords[:, 2] += (best_pos - self.coords[0, 2]) # Update all coords based on 1st coord
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            self.xt[i].append(time.time() - self.t0)
            self.xx[i].append(x)
            self.xy[i].append(y)
            self.xz[i].append(z)
            for ch in chs_img:
                self.set_channel(ch)
                img = self.snap_image()
                img_name = str(round(self.xt[i][-1]))
                img_path = os.path.join(self.save_dir, ch, str(i), img_name + '.tiff')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img)
            x_path = os.path.join(self.save_dir, str(i) + '.csv')
            x_data = np.column_stack((self.xt[i], self.xx[i], self.xy[i], self.xz[i]))
            np.savetxt(x_path, x_data, delimiter=',', header='t,x,y,z', comments='')
            u_path = os.path.join(self.save_dir, 'u.csv')
            u_data = np.column_stack((self.ut[i], self.uu[i]))
            np.savetxt(u_path, u_data, delimiter=',', header='t,u', comments='')

    def pulse_light(self, t_exc_on, t_exc_off, t_exc_width, t_exc_period, ch_dark, ch_exc):
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            t = time.time() - self.t0
            self.ut[i] += [t + i for i in range(t_exc_period)]
            if t > t_exc_on and t < t_exc_off:
                self.set_channel(ch_exc)
                time.sleep(t_exc_width)
                self.set_channel(ch_dark)
                self.uu[i] += [1.0] * t_exc_width + [0.0] * (t_exc_period - t_exc_width)
            else:
                self.uu[i] += [0.0] * t_exc_period

    def add_task(self, func, tstart, tstop, tstep, **kwargs):
        tstart += self.t0
        tstop += self.t0
        times = deque(np.arange(tstart, tstop + tstep, tstep))
        task = {'func': func, 'times': times, 'kwargs': kwargs}
        self.tasks.append(task)

    def run_tasks(self):
        if self.tasks:
            for i, task in enumerate(self.tasks):
                if task['times']:
                    if time.time() > task['times'][0]:
                        task['func'](**task['kwargs'])
                        self.tasks[i]['times'].popleft()
                else:
                    self.tasks.pop(i)
        else:
            return True
